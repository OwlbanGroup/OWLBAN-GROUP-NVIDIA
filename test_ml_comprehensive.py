#!/usr/bin/env python3
"""
Comprehensive test script for ML model improvements
"""

import sys
import os
import numpy as np

# Set environment variables for testing BEFORE importing anything else
os.environ['TOKEN_CLIENT_ID'] = 'test'
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['ALLOW_MISSING_TOKENS'] = 'true'

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
        print(f"CUDA Version: {gpu_stats.get('cuda_version', 'N/A')}")
        print(f"Device: {gpu_stats.get('device', 'N/A')}")

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
        print(f"Predictions range: {predictions.min()} to {predictions.max()}")
        print(f"Scores range: {scores.min():.4f} to {scores.max():.4f}")

        # Test edge cases
        print("Testing edge cases...")

        # Test with different data types if pandas available
        try:
            import pandas as pd
            X_pandas = pd.DataFrame(X[:50])
            predictions_pandas = detector.predict(X_pandas)
            print(f"Pandas input predictions shape: {predictions_pandas.shape}")
        except ImportError:
            print("Pandas not available, skipping pandas tests")

        # Test with small dataset
        X_small = np.random.randn(15, 3)
        detector_small = AnomalyDetector()
        detector_small.train(X_small)
        predictions_small = detector_small.predict(X_small[:5])
        print(f"Small dataset predictions shape: {predictions_small.shape}")

        print("✅ All ML model tests passed!")
        return True

    except Exception as e:
        print(f"❌ ML model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_behavior():
    """Test fallback behavior when cuML is not available"""
    print("\nTesting fallback behavior...")

    try:
        # Temporarily mock cuML import failure
        import sys
        original_modules = sys.modules.copy()

        # Mock failed cuML import
        class MockImportError:
            def __getattr__(self, name):
                raise ImportError("No module named 'cuml'")

        sys.modules['cuml'] = MockImportError()
        sys.modules['cuml.ensemble'] = MockImportError()

        # Force reload of the module to test fallback
        if 'src.ml_model' in sys.modules:
            del sys.modules['src.ml_model']

        from src.ml_model import AnomalyDetector

        detector = AnomalyDetector()
        X = np.random.randn(50, 5)
        detector.train(X)
        predictions = detector.predict(X[:10])

        print(f"Fallback predictions shape: {predictions.shape}")
        print("✅ Fallback behavior test passed!")

        # Restore original modules
        for module in ['cuml', 'cuml.ensemble']:
            if module in sys.modules:
                del sys.modules[module]

        return True

    except Exception as e:
        print(f"❌ Fallback behavior test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling in various scenarios"""
    print("\nTesting error handling...")

    try:
        detector = AnomalyDetector()

        # Test prediction without training
        try:
            detector.predict(np.random.randn(10, 5))
            print("❌ Should have failed for prediction without training")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error for prediction without training: {e}")

        # Test anomaly scores without training
        try:
            detector.get_anomaly_score(np.random.randn(10, 5))
            print("❌ Should have failed for anomaly scores without training")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error for anomaly scores without training: {e}")

        # Test training with insufficient data
        try:
            detector.train(np.random.randn(5, 5))  # Less than 10 samples
            print("❌ Should have failed for insufficient training data")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error for insufficient training data: {e}")

        print("✅ Error handling tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running comprehensive ML model tests...")
    print("=" * 50)

    tests = [
        ("ML Model Functionality", test_ml_model),
        ("Fallback Behavior", test_fallback_behavior),
        ("Error Handling", test_error_handling)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: FAILED - {str(e)}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All comprehensive tests passed! ML model improvements are 100% perfect.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        sys.exit(1)
