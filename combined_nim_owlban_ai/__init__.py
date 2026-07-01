"""
OWLBAN GROUP - NVIDIA NIM Integration Package.
Quantum-accelerated AI systems with enterprise monitoring.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

__version__ = "1.0.0"
__author__ = "Owlban Group"
__description__ = "NVIDIA NIM and OWLBAN AI integration with quantum acceleration"

_REQUIRED_EXPORTS = {
    "QuantumIntegratedSystem": (".integration", "QuantumIntegratedSystem"),
    "CombinedSystem": (".integration", "QuantumIntegratedSystem"),
    "NimManager": (".nim", "NimManager"),
    "OwlbanAI": (".owlban_ai", "OwlbanAI"),
    "NGCatalogManager": (".ngc_catalog", "NGCatalogManager"),
    "QuantumFinancialOmniscientSystem": (
        ".quantum_financial_omniscient_system",
        "QuantumFinancialOmniscientSystem",
    ),
    "AzureQuantumIntegrationManager": (
        ".azure_integration_manager",
        "AzureQuantumIntegrationManager",
    ),
    "TritonInferenceServer": (".triton_inference_server", "TritonInferenceServer"),
    "TritonModelManager": (".triton_inference_server", "TritonModelManager"),
    "RAPIDSDataProcessor": (".rapids_integration", "RAPIDSDataProcessor"),
    "DCGMMonitor": (".dcgm_monitor", "DCGMMonitor"),
    "EnergyOptimizer": (".energy_optimizer", "EnergyOptimizer"),
}

_OPTIONAL_EXPORTS = {
    "MultiModalAI": (".multi_modal_ai", "MultiModalAI"),
    "MultiModalInput": (".multi_modal_ai", "MultiModalInput"),
    "MultiModalEmbedding": (".multi_modal_ai", "MultiModalEmbedding"),
    "QuantumAIPerfection": (".quantum_ai_perfection", "QuantumAIPerfection"),
    "QuantumCircuitOptimizer": (
        ".quantum_ai_perfection",
        "QuantumCircuitOptimizer",
    ),
    "QuantumErrorCorrectionSystem": (
        ".quantum_ai_perfection",
        "QuantumErrorCorrectionSystem",
    ),
}


def _module_available(module_name: str) -> bool:
    """Return True when a submodule can be located without importing it."""
    full_name = f"{__name__}{module_name}" if module_name.startswith(".") else module_name
    return importlib.util.find_spec(full_name) is not None


__all__ = list(_REQUIRED_EXPORTS)
for name, (module_name, _) in _OPTIONAL_EXPORTS.items():
    if _module_available(module_name):
        __all__.append(name)


def __getattr__(name: str) -> Any:
    """Lazily load exports to avoid import-time failures during analysis."""
    if name in _REQUIRED_EXPORTS:
        module_name, attribute_name = _REQUIRED_EXPORTS[name]
    elif name in _OPTIONAL_EXPORTS:
        module_name, attribute_name = _OPTIONAL_EXPORTS[name]
        if not _module_available(module_name):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the available public names for introspection."""
    return sorted(set(globals()) | set(__all__))
