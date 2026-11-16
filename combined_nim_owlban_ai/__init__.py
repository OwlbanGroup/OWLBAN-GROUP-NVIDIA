"""Public interface for the OWLBAN NIM + AI integration package."""

from __future__ import annotations

from typing import Any, Dict

from .nim import NimManager
from .owlban_ai import OwlbanAI
from .dcgm_monitor import DCGMMonitor
from .energy_optimizer import EnergyOptimizer
from .quantum_financial_omniscient_system import QuantumFinancialOmniscientSystem

# Keep optional high-end AI modules importable even when heavyweight dependencies
# are not installed in a CPU-only environment.
try:
    from .multi_modal_ai import MultiModalFusion, MultiModalInput, MultiModalEmbedding
except Exception:  # pragma: no cover - optional dependency safety
    MultiModalFusion = None  # type: ignore[assignment]
    MultiModalInput = None  # type: ignore[assignment]
    MultiModalEmbedding = None  # type: ignore[assignment]

try:
    from .predictive_medicine_oracle import PredictiveMedicineOracle
except Exception:  # pragma: no cover - optional dependency safety
    PredictiveMedicineOracle = None  # type: ignore[assignment]

try:
    from .quantum_monitor import QuantumMonitor
except Exception:  # pragma: no cover - optional dependency safety
    QuantumMonitor = None  # type: ignore[assignment]


class CombinedSystem:
    """Compatibility wrapper exposing the expected public API for the app."""

    def __init__(self, *args, **kwargs):
        from .dcgm_monitor import DCGMMonitor
        from .energy_optimizer import EnergyOptimizer
        from .owlban_ai import OwlbanAI
        from .rapids_integration import RAPIDSDataProcessor
        from .triton_inference_server import TritonInferenceServer

        self.nim_manager = NimManager()
        self.owlban_ai = OwlbanAI()
        self.dcgm_monitor = DCGMMonitor()
        self.energy_optimizer = EnergyOptimizer(self.dcgm_monitor)
        self.rapids_processor = RAPIDSDataProcessor()
        self.triton_inference_server = TritonInferenceServer()
        self.quantum_system = QuantumFinancialOmniscientSystem(
            self.rapids_processor,
            self.triton_inference_server,
            self.energy_optimizer,
        )
        self.initialized = False
        self.initialize()

    def initialize(self):
        self.nim_manager.initialize()
        self.owlban_ai.load_models()
        self.initialized = True
        return True

    def run_inference(self, data):
        if not self.initialized:
            self.initialize()

        if isinstance(data, dict):
            payload = data
        else:
            payload = {"input": data}

        result = self.owlban_ai.run_inference(payload)
        if isinstance(result, dict):
            return {
                "prediction": result.get("prediction", result),
                "confidence": result.get("confidence", 0.85),
                "source": "combined_system",
            }
        return {"prediction": result, "confidence": 0.85, "source": "combined_system"}

    def get_ai_summary(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "nim_manager": self.nim_manager.get_resource_status() if hasattr(self.nim_manager, "get_resource_status") else {},
            "owlban_ai_ready": self.owlban_ai._loaded if hasattr(self.owlban_ai, "_loaded") else True,
            "energy_optimizer": self.energy_optimizer.get_efficiency_metrics(),
            "dcgm_metrics": self.dcgm_monitor.collect_metrics(),
        }


__all__ = [
    "CombinedSystem",
    "NimManager",
    "OwlbanAI",
    "DCGMMonitor",
    "EnergyOptimizer",
    "QuantumFinancialOmniscientSystem",
    "QuantumMonitor",
    "PredictiveMedicineOracle",
    "MultiModalFusion",
    "MultiModalInput",
    "MultiModalEmbedding",
]
