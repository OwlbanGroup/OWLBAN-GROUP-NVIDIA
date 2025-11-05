import logging
import time
from performance_optimization.advanced_anomaly_detection import AdvancedAnomalyDetection

class AnomalyDetection:
    def __init__(self, nim_manager, owlban_ai):
        self.nim_manager = nim_manager
        self.owlban_ai = owlban_ai
        self.advanced_detector = AdvancedAnomalyDetection(use_gpu=True)  # NVIDIA GPU acceleration
        self.logger = logging.getLogger("AnomalyDetection")
        self.anomaly_history = []

    def detect_anomalies(self):
        """Detect anomalies using NVIDIA GPU-accelerated AI"""
        self.logger.info("Detecting anomalies in infrastructure metrics with NVIDIA GPU-accelerated AI...")

        # Get real-time NVIDIA resource status
        resource_status = self.nim_manager.get_resource_status()
        self.logger.debug(f"Resource status for anomaly detection: {resource_status}")

        # Ensure OWLBAN AI models are loaded with GPU support
        if not self.owlban_ai.models_loaded:
            self.owlban_ai.load_models()

        # Use NVIDIA GPU-accelerated anomaly detection
        is_anomaly, score = self.advanced_detector.detect(resource_status)

        # Store anomaly history for trend analysis
        self.anomaly_history.append({
            'timestamp': time.time(),
            'is_anomaly': is_anomaly,
            'score': score,
            'resource_status': resource_status
        })

        # Keep only recent history
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]

        self.logger.info(f"NVIDIA GPU anomaly detection result: anomaly={is_anomaly}, score={score:.4f}")

        # Trigger alerts if anomaly detected
        if is_anomaly:
            self._trigger_anomaly_alert(score, resource_status)

        return is_anomaly, score

    def _trigger_anomaly_alert(self, score, resource_status):
        """Trigger alerts for detected anomalies using NVIDIA monitoring"""
        self.logger.warning(f"🚨 NVIDIA GPU Anomaly Alert: Score {score:.4f}")
        self.logger.warning(f"Affected resources: {resource_status}")

        # In practice, this would integrate with NVIDIA monitoring systems
        # and trigger automated remediation actions

    def get_anomaly_history(self):
        """Get historical anomaly data for analysis"""
        return self.anomaly_history

    def get_latest_anomaly(self):
        """Get latest anomaly for sync (placeholder)"""
        if self.anomaly_history:
            latest = self.anomaly_history[-1]
            return [latest['is_anomaly'], latest['score']]
        return [False, 0.0]

    def get_nvidia_anomaly_status(self):
        """Get NVIDIA anomaly detection status"""
        return {
            "gpu_status": self.advanced_detector.get_gpu_status(),
            "anomaly_history_size": len(self.anomaly_history),
            "owlban_ai_status": self.owlban_ai.get_model_status() if hasattr(self.owlban_ai, 'get_model_status') else {}
        }
