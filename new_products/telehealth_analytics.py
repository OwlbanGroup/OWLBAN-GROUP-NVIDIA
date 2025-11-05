import torch
import logging
import time

class TelehealthAnalytics:
    def __init__(self, nim_manager, owlban_ai):
        self.nim_manager = nim_manager
        self.owlban_ai = owlban_ai
        self.logger = logging.getLogger("TelehealthAnalytics")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patient_history = []

    def analyze_patient_data(self, patient_data):
        """Analyze patient data using NVIDIA GPU-accelerated AI"""
        self.logger.info("Analyzing patient data with NVIDIA GPU-accelerated AI...")

        # Ensure OWLBAN AI models are loaded with GPU support
        if not self.owlban_ai.models_loaded:
            self.owlban_ai.load_models()

        # Run inference using NVIDIA GPU acceleration
        result = self.owlban_ai.run_inference(patient_data)
        self.logger.info(f"NVIDIA GPU patient data analysis result: {result}")

        # Store analysis in patient history
        self.patient_history.append({
            'timestamp': time.time(),
            'patient_data': patient_data,
            'analysis_result': result
        })

        # Keep only recent history
        if len(self.patient_history) > 1000:
            self.patient_history = self.patient_history[-1000:]

        return result

    def monitor_infrastructure(self):
        """Monitor telehealth infrastructure using NVIDIA resource monitoring"""
        resource_status = self.nim_manager.get_resource_status()
        self.logger.info(f"Telehealth NVIDIA infrastructure status: {resource_status}")

        # Check for infrastructure issues that could affect telehealth
        infrastructure_health = self._assess_infrastructure_health(resource_status)

        if infrastructure_health['status'] != 'healthy':
            self.logger.warning(f"Telehealth infrastructure issues detected: {infrastructure_health}")
            # In practice, this would trigger automated remediation

        return resource_status

    def _assess_infrastructure_health(self, resource_status):
        """Assess infrastructure health for telehealth operations"""
        health_status = {'status': 'healthy', 'issues': []}

        # Check GPU memory availability (critical for telehealth AI)
        for key, value in resource_status.items():
            if 'Memory' in key and 'GB' in str(value):
                try:
                    memory_gb = float(str(value).replace('GB', '').strip())
                    if memory_gb < 10:  # Less than 10GB available
                        health_status['issues'].append(f"Low GPU memory: {memory_gb}GB")
                        health_status['status'] = 'warning'
                except:
                    pass

            elif 'Usage' in key and '%' in str(value):
                try:
                    usage = float(str(value).strip('%'))
                    if usage > 90:  # Over 90% usage
                        health_status['issues'].append(f"High resource usage: {usage}%")
                        health_status['status'] = 'critical'
                except:
                    pass

        return health_status

    def get_patient_analytics_summary(self):
        """Get summary of patient analytics using NVIDIA GPU processing"""
        if not self.patient_history:
            return {"total_analyses": 0, "positive_cases": 0, "avg_confidence": 0.0}

        total_analyses = len(self.patient_history)
        positive_cases = sum(1 for h in self.patient_history if h['analysis_result'].get('prediction') == 'positive')
        avg_confidence = sum(h['analysis_result'].get('confidence', 0) for h in self.patient_history) / total_analyses

        return {
            "total_analyses": total_analyses,
            "positive_cases": positive_cases,
            "avg_confidence": avg_confidence,
            "device": str(self.device)
        }

    def get_nvidia_telehealth_status(self):
        """Get NVIDIA telehealth analytics status"""
        return {
            "patient_history_size": len(self.patient_history),
            "device": str(self.device),
            "owlban_ai_status": self.owlban_ai.get_model_status() if hasattr(self.owlban_ai, 'get_model_status') else {},
            "nim_capabilities": self.nim_manager.get_nvidia_capabilities() if hasattr(self.nim_manager, 'get_nvidia_capabilities') else {}
        }
