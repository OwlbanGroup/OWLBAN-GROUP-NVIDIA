"""
OWLBAN GROUP - NVIDIA NIM Integration Package

E2E-safe: keep imports lightweight and optional dependencies guarded.
"""
__version__ = "1.0.0"
__author__ = "Owlban Group"

from .integration import QuantumIntegratedSystem, CombinedSystem

# Optional exports (guarded)
__all__ = [
    "QuantumIntegratedSystem",
    "CombinedSystem",
]

# Guard optional modules to prevent import-time failures in minimal envs.
try:
    from .nim import NimManager
    __all__.append("NimManager")
except Exception:
    pass

try:
    from .owlban_ai import OwlbanAI
    __all__.append("OwlbanAI")
except Exception:
    pass

try:
    from .quantum_financial_omniscient_system import QuantumFinancialOmniscientSystem
    __all__.append("QuantumFinancialOmniscientSystem")
except Exception:
    pass

try:
    from .azure_integration_manager import AzureQuantumIntegrationManager
    __all__.append("AzureQuantumIntegrationManager")
except Exception:
    pass

try:
    from .triton_inference_server import TritonInferenceServer, TritonModelManager
    __all__.extend(["TritonInferenceServer", "TritonModelManager"])
except Exception:
    pass

try:
    from .rapids_integration import RAPIDSDataProcessor
    __all__.append("RAPIDSDataProcessor")
except Exception:
    pass

try:
    from .dcgm_monitor import DCGMMonitor
    __all__.append("DCGMMonitor")
except Exception:
    pass

try:
    from .energy_optimizer import EnergyOptimizer
    __all__.append("EnergyOptimizer")
except Exception:
    pass

try:
    from .multi_modal_ai import MultiModalAI, MultiModalInput, MultiModalEmbedding
    __all__.extend(["MultiModalAI", "MultiModalInput", "MultiModalEmbedding"])
except Exception:
    pass

try:
    from .quantum_ai_perfection import (
        QuantumAIPerfection,
        QuantumCircuitOptimizer,
        QuantumErrorCorrectionSystem,
    )
    __all__.extend(["QuantumAIPerfection", "QuantumCircuitOptimizer", "QuantumErrorCorrectionSystem"])
except Exception:
    pass
