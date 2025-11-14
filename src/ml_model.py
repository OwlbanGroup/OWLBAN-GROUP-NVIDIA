"""
Machine Learning model module for anomaly detection with NVIDIA GPU support
"""
import numpy as np
import os
from typing import Optional, Dict, Any, Tuple
from .logger import telemetry_logger

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.device = None
        self.gpu_available = False
        self.gpu_count = 0
        self.cuda_version = None
        self.gpu_memory_info = {}

        # Configure GPU usage
        self.setup_gpu()

    def setup_gpu(self):
        """Configure NVIDIA GPU support with PyTorch and TensorFlow"""
        try:
            # Check for PyTorch CUDA support (preferred for ML)
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_count = torch.cuda.device_count()
                self.cuda_version = torch.version.cuda
                self.device = torch.device('cuda:0')

                # Log GPU information
                for i in range(self.gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    self.gpu_memory_info[i] = {
                        'name': gpu_name,
                        'memory_gb': gpu_memory,
                        'current_memory_allocated': 0
                    }
                    telemetry_logger.get_logger().info(
                        f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB) - CUDA {self.cuda_version}"
                    )

                # Set memory management
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                telemetry_logger.get_logger().info(f"PyTorch GPU configured with {self.gpu_count} GPU(s)")
            else:
                self.device = torch.device('cpu')
                telemetry_logger.get_logger().info("PyTorch CUDA not available, using CPU")

        except ImportError:
            telemetry_logger.get_logger().info("PyTorch not available, checking TensorFlow...")

            # Fallback to TensorFlow
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.gpu_available = True
                    self.gpu_count = len(gpus)
                    self.device = 'gpu'

                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        telemetry_logger.get_logger().info(f"TensorFlow GPU configured with {len(gpus)} GPU(s)")
                    except RuntimeError as e:
                        telemetry_logger.get_logger().error(f"TensorFlow GPU configuration error: {e}")
                else:
                    self.device = 'cpu'
                    telemetry_logger.get_logger().info("No GPU found, using CPU")
            except ImportError:
                self.device = 'cpu'
                telemetry_logger.get_logger().info("Neither PyTorch nor TensorFlow available, using CPU")

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        stats = {
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count,
            'cuda_version': self.cuda_version,
            'device': str(self.device),
            'gpu_memory_info': self.gpu_memory_info
        }

        if self.gpu_available:
            try:
                import torch
                for i in range(self.gpu_count):
                    current_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    max_memory = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                    stats[f'gpu_{i}_current_memory_gb'] = current_memory
                    stats[f'gpu_{i}_max_memory_gb'] = max_memory
                    self.gpu_memory_info[i]['current_memory_allocated'] = current_memory
            except ImportError:
                pass

        return stats

    def train(self, X, contamination=0.1):
        """
        Train the anomaly detection model using GPU-accelerated cuML if available
        """
        if X.shape[0] < 10:
            raise ValueError("Need at least 10 samples to train")

        try:
            # Try to use cuML for GPU acceleration
            from cuml.ensemble import IsolationForest as cuML_IsolationForest
            import cudf

            # Convert to cuDF DataFrame for GPU processing
            if hasattr(X, 'values'):
                X_gpu = cudf.DataFrame(X.values)
            else:
                X_gpu = cudf.DataFrame(X)

            self.model = cuML_IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X_gpu)
            self.is_trained = True
            telemetry_logger.get_logger().info("GPU-accelerated anomaly detection model trained with cuML")

        except ImportError:
            telemetry_logger.get_logger().info("cuML not available, falling back to sklearn")
            # Fallback to sklearn
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(contamination=contamination, random_state=42)
            self.model.fit(X)
            self.is_trained = True
            telemetry_logger.get_logger().info("CPU-based anomaly detection model trained with sklearn")

        except Exception as e:
            telemetry_logger.get_logger().error(f"Error training GPU model: {e}, falling back to sklearn")
            # Fallback to sklearn on any error
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(contamination=contamination, random_state=42)
            self.model.fit(X)
            self.is_trained = True
            telemetry_logger.get_logger().info("CPU-based anomaly detection model trained with sklearn (fallback)")

    def predict(self, X):
        """
        Predict anomalies in the data with batch processing and GPU memory optimization
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")

        try:
            # Optimize for large datasets with batch processing
            batch_size = 1000  # Process in batches to avoid memory issues
            all_predictions = []

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]

                # Check if using cuML model
                if hasattr(self.model, '_predict_gpu'):
                    # cuML model - convert input to GPU format
                    import cudf
                    if hasattr(batch_X, 'values'):
                        X_gpu = cudf.DataFrame(batch_X.values)
                    else:
                        X_gpu = cudf.DataFrame(batch_X)

                    # Clear GPU cache before prediction to optimize memory
                    if hasattr(self.model, '_predict_gpu'):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    predictions = self.model.predict(X_gpu)

                    # Convert to numpy array if needed
                    if hasattr(predictions, 'to_numpy'):
                        predictions = predictions.to_numpy()
                    elif hasattr(predictions, 'values'):
                        predictions = predictions.values
                else:
                    # sklearn model
                    predictions = self.model.predict(batch_X)

                all_predictions.extend(predictions)

            # Convert to numpy array
            all_predictions = np.array(all_predictions)

            # Convert to 0 (normal) and 1 (anomaly)
            # cuML returns 1 for outliers, -1 for inliers (opposite of sklearn)
            if hasattr(self.model, '_predict_gpu'):
                # cuML: 1 = anomaly, -1 = normal
                anomalies = np.where(all_predictions == 1, 1, 0)
            else:
                # sklearn: -1 = anomaly, 1 = normal
                anomalies = np.where(all_predictions == -1, 1, 0)

            return anomalies

        except Exception as e:
            telemetry_logger.get_logger().error(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {e}")

    def get_anomaly_score(self, X):
        """
        Get anomaly scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")

        try:
            # Check if using cuML model
            if hasattr(self.model, '_predict_gpu'):
                # cuML model - convert input to GPU format
                import cudf
                if hasattr(X, 'values'):
                    X_gpu = cudf.DataFrame(X.values)
                else:
                    X_gpu = cudf.DataFrame(X)
                scores = self.model.decision_function(X_gpu)
                # Convert to numpy array if needed
                if hasattr(scores, 'to_numpy'):
                    scores = scores.to_numpy()
                elif hasattr(scores, 'values'):
                    scores = scores.values
            else:
                # sklearn model
                scores = self.model.decision_function(X)

            return scores

        except Exception as e:
            telemetry_logger.get_logger().error(f"Error getting anomaly scores: {e}")
            raise ValueError(f"Anomaly score calculation failed: {e}")
