"""
OWLBAN GROUP - NVIDIA NIM Integration Package
Quantum-accelerated AI systems with enterprise monitoring
"""

__version__ = "1.0.0"
__author__ = "Owlban Group"
__description__ = "NVIDIA NIM and OWLBAN AI integration with quantum acceleration"

from .integration import QuantumIntegratedSystem as CombinedSystem, QuantumIntegratedSystem
from .nim import NimManager
from .owlban_ai import OwlbanAI
from .quantum_financial_omniscient_system import QuantumFinancialOmniscientSystem
from .azure_integration_manager import AzureQuantumIntegrationManager
from .triton_inference_server import TritonInferenceServer, TritonModelManager
from .rapids_integration import RAPIDSDataProcessor
from .dcgm_monitor import DCGMMonitor
from .energy_optimizer import EnergyOptimizer

__all__ = [
    "QuantumIntegratedSystem",
    "CombinedSystem",
    "NimManager",
    "OwlbanAI",
    "QuantumFinancialOmniscientSystem",
    "AzureQuantumIntegrationManager",
    "TritonInferenceServer",
    "TritonModelManager",
    "RAPIDSDataProcessor",
    "DCGMMonitor",
    "EnergyOptimizer"
]
