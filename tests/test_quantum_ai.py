IT"""
Comprehensive tests for quantum AI systems integration.
Tests all quantum AI components including NIM, OWLBAN AI, financial systems,
energy optimization, monitoring, portfolio optimization, risk analysis, and market prediction.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from combined_nim_owlban_ai import NimManager as NIM
from combined_nim_owlban_ai import OwlbanAI as OWLBANAI
from combined_nim_owlban_ai import QuantumFinancialOmniscientSystem
from combined_nim_owlban_ai import EnergyOptimizer
from combined_nim_owlban_ai import DCGMMonitor as QuantumMonitor
from quantum_financial_ai.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
from quantum_financial_ai.quantum_risk_analyzer import QuantumRiskAnalyzer
from quantum_financial_ai.quantum_market_predictor import QuantumMarketPredictor, MarketData


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
        # Check for either gpu_utilization or error message
        assert 'gpu_utilization' in metrics or 'error' in metrics

    def test_portfolio_optimizer(self):
        """Test quantum portfolio optimization."""
        qpo = QuantumPortfolioOptimizer()
        # Add some sample assets first
        from quantum_financial_ai.quantum_portfolio_optimizer import PortfolioAsset
        qpo.add_asset(PortfolioAsset("AAPL", 0.08, 0.15, 150.0, 100))
        qpo.add_asset(PortfolioAsset("GOOGL", 0.10, 0.18, 2500.0, 50))
        qpo.add_asset(PortfolioAsset("MSFT", 0.09, 0.16, 300.0, 75))
        result = qpo.optimize_portfolio()
        assert hasattr(result, 'optimal_weights')
        assert hasattr(result, 'expected_return')
        assert hasattr(result, 'sharpe_ratio')

    def test_risk_analyzer(self):
        """Test quantum risk analysis."""
        qra = QuantumRiskAnalyzer()
        positions = np.random.rand(100, 5)
        result = qra.analyze_risk(positions)
        assert hasattr(result, 'value_at_risk')
        assert hasattr(result, 'conditional_var')
        assert hasattr(result, 'expected_shortfall')

    def test_market_predictor(self):
        """Test quantum market prediction."""
        qmp = QuantumMarketPredictor()

        # Create sample market data
        prices = np.random.rand(1000) * 100 + 100  # Random prices around 100
        volumes = np.random.rand(1000) * 1000000  # Random volumes
        timestamps = np.arange(1000)

        market_data = MarketData(
            symbol="AAPL",
            prices=prices,
            volumes=volumes,
            timestamps=timestamps
        )

        # Add market data and train
        qmp.add_market_data(market_data)
        qmp.train_quantum_model("AAPL", epochs=1)  # Quick training for test

        # Make prediction
        prediction = qmp.predict_market_movement("AAPL")
        assert hasattr(prediction, 'predicted_price')
        assert hasattr(prediction, 'confidence')
        assert hasattr(prediction, 'direction')
        assert prediction.direction in ['up', 'down', 'neutral']


if __name__ == "__main__":
    pytest.main([__file__])
