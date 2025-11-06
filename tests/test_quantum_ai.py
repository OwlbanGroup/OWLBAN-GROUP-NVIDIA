import pytest
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combined_nim_owlban_ai import NimManager as NIM
from combined_nim_owlban_ai import OwlbanAI as OWLBANAI
from combined_nim_owlban_ai import QuantumFinancialOmniscientSystem
from combined_nim_owlban_ai import EnergyOptimizer
from combined_nim_owlban_ai import DCGMMonitor as QuantumMonitor
from quantum_financial_ai.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
from quantum_financial_ai.quantum_risk_analyzer import QuantumRiskAnalyzer
from quantum_financial_ai.quantum_market_predictor import QuantumMarketPredictor


class TestQuantumAISystems:
    """Comprehensive tests for all quantum AI systems."""

    def test_nim_gpu_acceleration(self):
        """Test NIM GPU processing capabilities."""
        nim = NIM()
        nim.initialize()
        # Test GPU availability
        capabilities = nim.get_nvidia_capabilities()
        assert isinstance(capabilities, dict)
        assert 'cuda_available' in capabilities

        # Test resource status
        status = nim.get_resource_status()
        assert isinstance(status, dict)
        assert 'CPU_Usage' in status

    def test_owlban_ai_inference(self):
        """Test OWLBAN AI inference pipeline."""
        owlban = OWLBANAI()
        owlban.load_models()
        sample_input = np.random.rand(50, 20)
        result = owlban.run_inference(sample_input)
        assert isinstance(result, dict)
        assert 'prediction' in result

    def test_quantum_financial_system(self):
        """Test quantum financial omniscient system."""
        # Create mock dependencies
        class MockRAPIDS:
            def load_data_gpu(self, data): return data
            def preprocess_financial_data(self, data): return data
            def optimize_portfolio_gpu(self, data): return {'weights': {'stocks': 0.6, 'bonds': 0.4}}
            def cluster_market_data(self, data): return data, []

        class MockTriton:
            pass

        class MockEnergy:
            pass

        qfs = QuantumFinancialOmniscientSystem(MockRAPIDS(), MockTriton(), MockEnergy())
        market_data = {"stocks": np.random.rand(100, 5)}
        result = qfs.process_global_markets(market_data)
        assert isinstance(result, dict)
        assert 'predictions' in result

    def test_energy_optimizer(self):
        """Test energy optimization algorithms."""
        eo = EnergyOptimizer()
        system_load = np.random.rand(24, 10)  # 24 hours, 10 metrics
        result = eo.optimize_energy(system_load)
        assert isinstance(result, dict)
        assert 'optimized_schedule' in result

    def test_quantum_monitor(self):
        """Test quantum system monitoring."""
        qm = QuantumMonitor()
        metrics = qm.collect_metrics()
        assert isinstance(metrics, dict)
        assert 'gpu_utilization' in metrics

    def test_portfolio_optimizer(self):
        """Test quantum portfolio optimization."""
        qpo = QuantumPortfolioOptimizer()
        portfolio = np.random.rand(50, 10)  # 50 assets, 10 features
        result = qpo.optimize_portfolio(portfolio)
        assert isinstance(result, dict)
        assert 'optimal_weights' in result

    def test_risk_analyzer(self):
        """Test quantum risk analysis."""
        qra = QuantumRiskAnalyzer()
        positions = np.random.rand(100, 5)
        result = qra.analyze_risk(positions)
        assert isinstance(result, dict)
        assert 'risk_metrics' in result

    def test_market_predictor(self):
        """Test quantum market prediction."""
        qmp = QuantumMarketPredictor()
        historical_data = np.random.rand(1000, 20)
        predictions = qmp.predict_market(historical_data)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 1000


if __name__ == "__main__":
    pytest.main([__file__])
