#!/usr/bin/env python3
"""
Test script for ML model improvements
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.ml_model import AnomalyDetector
from src.logger import telemetry_logger

def test_ml_model():
    """Test the improved ML model"""
    print("Testing improved ML model...")

    try:
        detector = AnomalyDetector()

        # Check GPU stats
        gpu_stats = detector.get_gpu_stats()
        print(f"GPU Available: {gpu_stats['gpu_available']}")
        print(f"GPU Count: {gpu_stats['gpu_count']}")

        # Test with sample data
        X = np.random.randn(100, 5)  # 100 samples, 5 features

        print("Training model...")
        detector.train(X)

        print("Making predictions...")
        predictions = detector.predict(X[:20])
        scores = detector.get_anomaly_score(X[:20])

        print(f"Predictions shape: {predictions.shape}")
        print(f"Anomaly scores shape: {scores.shape}")
        print(f"Unique predictions: {np.unique(predictions)}")

        print("✅ ML model test passed!")
        return True

    except Exception as e:
        print(f"❌ ML model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_model()
    sys.exit(0 if success else 1)
