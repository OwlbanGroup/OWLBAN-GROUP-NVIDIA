"""
OWLBAN AI Module - NVIDIA GPU-Accelerated AI System

This module provides a comprehensive AI system optimized for NVIDIA GPUs,
featuring TensorRT acceleration, multi-GPU support, and robust error handling.
"""

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# NVIDIA-specific imports
try:
    import tensorrt as trt  # type: ignore
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None
    TENSORRT_AVAILABLE = False


class OptimizedModel(nn.Module):
    """NVIDIA-optimized neural network with cuDNN acceleration"""

    def __init__(self, input_size: int = 10, hidden_size: int = 50, output_size: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # cuDNN optimized
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OwlbanAI:
    """Main class for OWLBAN AI system with NVIDIA GPU acceleration."""

    def __init__(self):
        self.models_loaded = False
        self.models: Dict[str, nn.Module] = {}
        self.tensorrt_engines = {}
        self.logger = logging.getLogger("OwlbanAI")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0
        self.logger.info("Using device: %s, GPU count: %d", self.device, self.gpu_count)

    def load_models(self) -> None:
        """Load and initialize NVIDIA-optimized AI models."""
        self.logger.info("Loading OWLBAN AI models with NVIDIA GPU acceleration...")
        try:
            # Create NVIDIA-optimized models
            self.models["prediction_model"] = OptimizedModel().to(self.device)
            self.models["anomaly_model"] = OptimizedModel().to(self.device)
            self.models["telehealth_model"] = OptimizedModel().to(self.device)

            # Enable cuDNN optimization
            if self.cuda_available:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

            # Set models to evaluation mode
            for model in self.models.values():
                model.eval()

            # Optimize for multi-GPU if available
            if self.gpu_count > 1:
                self._setup_multi_gpu()

            self.models_loaded = True
            self.logger.info("Models loaded successfully with NVIDIA GPU acceleration.")
            print("OWLBAN AI models loaded with NVIDIA GPU support.")
        except (RuntimeError, ValueError) as e:
            self.logger.error("Error loading models: %s", e)
            self.models_loaded = False

    def _prepare_input_data(self, data):
        """Prepare input data by converting to tensor."""
        if isinstance(data, dict):
            input_data = torch.tensor(np.array(list(data.values())), dtype=torch.float32).to(self.device)
        elif isinstance(data, list):
            input_data = torch.tensor(np.array(data), dtype=torch.float32).to(self.device)
        else:
            input_data = torch.tensor(np.array([data]), dtype=torch.float32).to(self.device)
        return input_data

    def _check_gpu_health(self, input_data):
        """Check GPU health and handle memory issues."""
        if self.cuda_available:
            try:
                memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if memory_used > 0.95:
                    self.logger.warning("GPU memory critical, clearing cache")
                    torch.cuda.empty_cache()
            except (RuntimeError, ValueError) as gpu_error:
                self.logger.error("GPU health check failed: %s", gpu_error)
                # Fallback to CPU
                self.device = torch.device("cpu")
                self.cuda_available = False
                self.gpu_count = 0
                input_data = input_data.to(self.device)
                for model in self.models.values():
                    model.to(self.device)
        return input_data

    def _perform_inference(self, input_data):
        """Perform inference on the prediction model."""
        with torch.no_grad():
            output = self.models["prediction_model"](input_data.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()

        result = {
            "prediction": "positive" if prediction == 1 else "negative",
            "confidence": confidence,
            "device_used": str(self.device),
        }
        self.logger.info("NVIDIA GPU inference completed: %s", result)
        return result

    def _handle_inference_error(self, e, data):
        """Handle inference errors with recovery attempts."""
        self.logger.error("Inference error: %s", e)
        try:
            self.logger.info("Attempting inference recovery...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.cuda_available:
                self.device = torch.device("cpu")
                self.cuda_available = False
                self.gpu_count = 0
                self.logger.info("Switching to CPU for recovery")
                return self.run_inference(data)
        except (RuntimeError, ValueError) as recovery_error:
            self.logger.error("Recovery failed: %s", recovery_error)

        return {"prediction": "error", "confidence": 0.0, "error": str(e)}

    def run_inference(self, data):
        """Run inference using NVIDIA GPU acceleration with health checks.

        Important: This function is designed to be *fail-open*.
        If models fail to load on GPU, it falls back to CPU instead of raising,
        preventing the API/service from getting "lockdowned"/blocked.
        """
        if not self.models_loaded:
            self.logger.warning("Models not loaded, attempting to reload...")
            self.load_models()

            if not self.models_loaded:
                self.logger.error("Models not loaded after reload. Forcing CPU fallback.")
                # Force CPU and try loading again
                try:
                    self.device = torch.device("cpu")
                    self.cuda_available = False
                    self.gpu_count = 0
                    self.models_loaded = False
                    self.load_models()
                except Exception as cpu_fallback_error:
                    self.logger.error("CPU fallback model load failed: %s", cpu_fallback_error)
                    return {
                        "prediction": "error",
                        "confidence": 0.0,
                        "device_used": str(self.device),
                        "error": "Models not loaded and CPU fallback failed",
                    }

                if not self.models_loaded:
                    return {
                        "prediction": "error",
                        "confidence": 0.0,
                        "device_used": str(self.device),
                        "error": "Models not loaded after reload and CPU fallback",
                    }

        try:
            input_data = self._prepare_input_data(data)
            input_data = self._check_gpu_health(input_data)
            return self._perform_inference(input_data)
        except (RuntimeError, ValueError, TypeError) as e:
            return self._handle_inference_error(e, data)

    def get_latest_prediction(self):
        """Get latest prediction for sync (placeholder)"""
        return [0.95, 0.85, 0.75]

    def get_model_status(self):
        """Get model loading and GPU status"""
        return {
            "models_loaded": self.models_loaded,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "models_count": len(self.models),
        }

    def optimize_models_for_inference(self):
        """Optimize models using NVIDIA TensorRT with real implementation"""
        if not self.models_loaded or not TENSORRT_AVAILABLE:
            return

        self.logger.info("Optimizing models for NVIDIA TensorRT inference...")

        try:
            for model_name, model in self.models.items():
                dummy_input = torch.randn(1, 10).to(self.device)
                onnx_path = f"{model_name}_optimized.onnx"
                torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

                with trt.Builder(trt.Logger(trt.Logger.WARNING)) as builder:
                    flag = trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
                    network = builder.create_network(1 << int(flag))
                    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

                    with open(onnx_path, "rb") as f:
                        parser.parse(f.read())

                    config = builder.create_builder_config()
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
                    engine = builder.build_serialized_network(network, config)

                    if engine:
                        self.tensorrt_engines[model_name] = engine
                        self.logger.info("TensorRT engine created for %s", model_name)

            self.logger.info("Models optimized for TensorRT inference.")
        except (RuntimeError, ValueError, OSError) as e:
            self.logger.error("TensorRT optimization failed: %s", e)

    def _setup_multi_gpu(self):
        """Setup multi-GPU support using NVIDIA NCCL"""
        try:
            if not dist.is_initialized():
                dist.init_process_group("nccl", rank=0, world_size=self.gpu_count)

            for model_name in list(self.models.keys()):
                self.models[model_name] = DDP(self.models[model_name])

            self.logger.info("Multi-GPU setup completed with %d GPUs", self.gpu_count)
        except (RuntimeError, ValueError, OSError) as e:
            self.logger.error("Multi-GPU setup failed: %s", e)

    def run_parallel_inference(self, data_batch: List[Dict]) -> List[Dict]:
        """Run parallel inference across multiple GPUs"""
        if not self.models_loaded or self.gpu_count <= 1:
            return [self.run_inference(data) for data in data_batch]

        try:
            batch_size = len(data_batch)
            gpu_batch_size = batch_size // self.gpu_count

            results = []
            for gpu_id in range(self.gpu_count):
                start_idx = gpu_id * gpu_batch_size
                end_idx = start_idx + gpu_batch_size if gpu_id < self.gpu_count - 1 else batch_size
                gpu_data = data_batch[start_idx:end_idx]

                with torch.cuda.device(gpu_id):
                    gpu_results = [self.run_inference(data) for data in gpu_data]
                    results.extend(gpu_results)

            return results
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.error("Parallel inference failed: %s", e)
            return [self.run_inference(data) for data in data_batch]

    def get_gpu_memory_usage(self) -> Dict[str, Dict[str, float]]:
        """Get detailed GPU memory usage across all GPUs"""
        memory_usage: Dict[str, Dict[str, float]] = {}

        if self.cuda_available:
            for i in range(self.gpu_count):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                    memory_usage[f"gpu_{i}"] = {
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                        "total_gb": total,
                        "utilization_percent": (allocated / total) * 100,
                    }
                except RuntimeError as e:
                    self.logger.error("Error getting GPU %d memory: %s", i, e)

        return memory_usage

