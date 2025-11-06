"""
Test Quantum Machine Learning Pipeline
OWLBAN GROUP - Testing quantum ML for financial applications
"""

import unittest
import numpy as np
import logging
from quantum_financial_ai.quantum_machine_learning_pipeline import (
    QuantumMLPipeline,
    FinancialQuantumMLApplication,
    create_synthetic_financial_data,
    FinancialFeatureData,
    QuantumFeatureEncoder,
    QuantumSVMClassifier,
    VariationalQuantumClassifier,
    QuantumNeuralNetwork
)

class TestQuantumFeatureEncoder(unittest.TestCase):
    """Test quantum feature encoding"""

    def setUp(self):
        self.encoder = QuantumFeatureEncoder(n_qubits=4)

    def test_encoding_creation(self):
        """Test feature encoding circuit creation"""
        features = np.random.randn(10, 4)
        circuit = self.encoder.encode_features(features)
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.num_qubits, 4)

    def test_feature_map(self):
        """Test feature map circuit"""
        feature_map = self.encoder.get_feature_map_circuit()
        self.assertIsNotNone(feature_map)

class TestQuantumSVMClassifier(unittest.TestCase):
    """Test Quantum SVM Classifier"""

    def setUp(self):
        self.classifier = QuantumSVMClassifier(C=1.0)

    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertIsNone(self.classifier.qsvc)

    def test_training(self):
        """Test QSVM training"""
        # Create synthetic data
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50)

        # Train model
        self.classifier.fit(X, y)

        # Check if trained
        self.assertIsNotNone(self.classifier.qsvc)
        self.assertGreater(self.classifier.training_time, 0)

    def test_prediction(self):
        """Test QSVM prediction"""
        # Create and train model
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50)
        self.classifier.fit(X, y)

        # Make predictions
        X_test = np.random.randn(10, 4)
        predictions = self.classifier.predict(X_test)

        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

class TestVariationalQuantumClassifier(unittest.TestCase):
    """Test Variational Quantum Classifier"""

    def setUp(self):
        self.classifier = VariationalQuantumClassifier(n_qubits=4, n_layers=2)

    def test_initialization(self):
        """Test VQC initialization"""
        self.assertIsNotNone(self.classifier)
        self.assertIsNone(self.classifier.vqc)

    def test_circuit_creation(self):
        """Test VQC circuit creation"""
        circuit = self.classifier._create_vqc_circuit()
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.num_qubits, 4)

class TestQuantumNeuralNetwork(unittest.TestCase):
    """Test Quantum Neural Network"""

    def setUp(self):
        self.qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)

    def test_initialization(self):
        """Test QNN initialization"""
        self.assertIsNotNone(self.qnn)
        self.assertIsNone(self.qnn.qnn)

    def test_circuit_creation(self):
        """Test QNN circuit creation"""
        qnn_circuit = self.qnn._create_qnn_circuit()
        self.assertIsNotNone(qnn_circuit)

class TestQuantumMLPipeline(unittest.TestCase):
    """Test complete quantum ML pipeline"""

    def setUp(self):
        self.pipeline = QuantumMLPipeline(model_type="qsvm", n_qubits=4)

    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.model_type, "qsvm")
        self.assertEqual(self.pipeline.n_qubits, 4)

    def test_data_preprocessing(self):
        """Test data preprocessing"""
        data = create_synthetic_financial_data(n_samples=100, n_features=6)
        X_processed, y_processed = self.pipeline.preprocess_data(data)

        # Check dimensions (should be reduced to n_qubits)
        self.assertEqual(X_processed.shape[1], 4)  # n_qubits
        self.assertEqual(len(y_processed), 100)

    def test_training(self):
        """Test pipeline training"""
        data = create_synthetic_financial_data(n_samples=100, n_features=4)
        result = self.pipeline.train(data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result.accuracy, float)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)

    def test_prediction(self):
        """Test pipeline prediction"""
        # Train first
        data = create_synthetic_financial_data(n_samples=100, n_features=4)
        self.pipeline.train(data)

        # Make predictions
        test_features = np.random.randn(10, 4)
        result = self.pipeline.predict(test_features)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.predictions), 10)

class TestFinancialQuantumMLApplication(unittest.TestCase):
    """Test financial quantum ML application"""

    def setUp(self):
        self.app = FinancialQuantumMLApplication()

    def test_pipeline_creation(self):
        """Test pipeline creation methods"""
        buy_sell_pipeline = self.app.create_buy_sell_classifier()
        self.assertIsNotNone(buy_sell_pipeline)
        self.assertEqual(buy_sell_pipeline.model_type, "qsvm")

        market_pipeline = self.app.create_market_predictor()
        self.assertIsNotNone(market_pipeline)
        self.assertEqual(market_pipeline.model_type, "vqc")

        risk_pipeline = self.app.create_risk_assessor()
        self.assertIsNotNone(risk_pipeline)
        self.assertEqual(risk_pipeline.model_type, "qnn")

    def test_performance_report(self):
        """Test performance report generation"""
        report = self.app.get_pipeline_performance_report()
        self.assertIsNotNone(report)
        self.assertIn("pipelines", report)
        self.assertIn("overall_quantum_advantage", report)

class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic financial data generation"""

    def test_synthetic_data_creation(self):
        """Test synthetic data creation"""
        data = create_synthetic_financial_data(n_samples=100, n_features=4)

        self.assertIsNotNone(data)
        self.assertEqual(data.features.shape, (100, 4))
        self.assertEqual(len(data.labels), 100)
        self.assertEqual(len(data.feature_names), 4)

    def test_data_labels(self):
        """Test that labels are binary"""
        data = create_synthetic_financial_data(n_samples=50, n_features=3)
        unique_labels = np.unique(data.labels)
        self.assertTrue(all(label in [0, 1] for label in unique_labels))

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main()
