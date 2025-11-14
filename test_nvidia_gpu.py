#!/usr/bin/env python3
"""
Test script for NVIDIA GPU integration
"""

import sys
import os
import subprocess
from datetime import datetime, timezone

# Set environment variables for testing
os.environ['ALLOW_MISSING_TOKENS'] = 'true'
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing'

sys.path.append(os.path.dirname(__file__))

from src.ml_model import AnomalyDetector
from src.nvidia_telemetry_parser import nvidia_telemetry_parser, GPUTelemetryEvent
from src.logger import telemetry_logger

def test_gpu_ml_model():
    """Test GPU-accelerated ML model"""
    print("Testing GPU-accelerated ML model...")

    try:
        detector = AnomalyDetector()

        # Check GPU availability
        gpu_stats = detector.get_gpu_stats()
        print(f"GPU Available: {gpu_stats['gpu_available']}")
        print(f"GPU Count: {gpu_stats['gpu_count']}")
        print(f"CUDA Version: {gpu_stats['cuda_version']}")
        print(f"Device: {gpu_stats['device']}")

        # Test with sample data
        import numpy as np
        X = np.random.randn(50, 5)  # 50 samples, 5 features

        detector.train(X)
        predictions = detector.predict(X[:10])
        scores = detector.get_anomaly_score(X[:10])

        print(f"Model trained successfully")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Anomaly scores shape: {scores.shape}")

        return True

    except Exception as e:
        print(f"GPU ML model test failed: {str(e)}")
        return False

def test_nvidia_smi_parsing():
    """Test NVIDIA SMI output parsing"""
    print("\nTesting NVIDIA SMI output parsing...")

    try:
        # Try to get real nvidia-smi output
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            nvidia_smi_output = result.stdout
            print("Using real nvidia-smi output")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            # Use mock output for testing
            nvidia_smi_output = """
NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P8    15W / 250W |    512MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1234      C   python                           512MiB |
+-----------------------------------------------------------------------------+
"""
            print("Using mock nvidia-smi output for testing")

        # Test parsing
        if nvidia_telemetry_parser.validate_gpu_telemetry(nvidia_smi_output):
            event = nvidia_telemetry_parser.parse_nvidia_smi_output(nvidia_smi_output)
            if event:
                print(f"Successfully parsed GPU telemetry for GPU {event.gpu_id}")
                print(f"GPU Name: {event.gpu_name}")
                print(f"GPU Utilization: {event.gpu_utilization_percent}%")
                print(f"Memory Used: {event.memory_used_mb}MB / {event.memory_total_mb}MB")
                print(f"Temperature: {event.temperature_celsius}°C")
                print(f"Power: {event.power_draw_watts}W / {event.power_limit_watts}W")
                print(f"Processes: {len(event.processes)}")

                # Test metrics extraction
                metrics = nvidia_telemetry_parser.extract_gpu_metrics(event)
                print(f"Extracted {len(metrics)} GPU metrics")

                return True
            else:
                print("Failed to parse nvidia-smi output")
                return False
        else:
            print("NVIDIA SMI output validation failed")
            return False

    except Exception as e:
        print(f"NVIDIA SMI parsing test failed: {str(e)}")
        return False

def test_dcgm_parsing():
    """Test DCGM metrics parsing"""
    print("\nTesting DCGM metrics parsing...")

    try:
        # Mock DCGM metrics data
        dcgm_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'gpu_id': 0,
            'gpu_name': 'NVIDIA A100',
            'driver_version': '525.105.17',
            'cuda_version': '12.0',
            'gpu_utilization': 45.5,
            'memory_used_mb': 2048.0,
            'memory_total_mb': 40960.0,
            'memory_utilization': 5.0,
            'temperature_celsius': 65.0,
            'power_draw_watts': 180.0,
            'power_limit_watts': 300.0,
            'fan_speed_percent': 40.0,
            'clock_graphics_mhz': 1410,
            'clock_sm_mhz': 1410,
            'clock_memory_mhz': 1215,
            'clock_video_mhz': 555,
            'pcie_link_gen': 4,
            'pcie_link_width': 16,
            'processes': [
                {'pid': 1234, 'name': 'python', 'memory_mb': 512.0}
            ],
            'hostname': 'test-host',
            'nvidia_smi_version': '525.105.17'
        }

        event = nvidia_telemetry_parser.parse_dcgm_metrics(dcgm_data)
        if event:
            print(f"Successfully parsed DCGM metrics for GPU {event.gpu_id}")
            print(f"GPU Name: {event.gpu_name}")
            print(f"GPU Utilization: {event.gpu_utilization_percent}%")
            print(f"Memory Used: {event.memory_used_mb}MB / {event.memory_total_mb}MB")

            # Test metrics extraction
            metrics = nvidia_telemetry_parser.extract_gpu_metrics(event)
            print(f"Extracted {len(metrics)} GPU metrics")

            return True
        else:
            print("Failed to parse DCGM metrics")
            return False

    except Exception as e:
        print(f"DCGM parsing test failed: {str(e)}")
        return False

def test_ngc_configuration():
    """Test NGC configuration"""
    print("\nTesting NGC configuration...")

    try:
        from config import config

        ngc_config = {
            'ngc_api_key': config.NGC_API_KEY,
            'ngc_cli_path': config.NGC_CLI_PATH,
            'ngc_registry_url': config.NGC_REGISTRY_URL,
            'ngc_org': config.NGC_ORG,
            'nvidia_visible_devices': config.NVIDIA_VISIBLE_DEVICES,
            'cuda_visible_devices': config.CUDA_VISIBLE_DEVICES,
            'gpu_memory_fraction': config.GPU_MEMORY_FRACTION
        }

        print("NGC Configuration:")
        for key, value in ngc_config.items():
            if 'key' in key.lower() and value:
                print(f"  {key}: [REDACTED]")
            else:
                print(f"  {key}: {value}")

        # Check if NGC CLI is available
        try:
            result = subprocess.run([config.NGC_CLI_PATH, '--version'],
                                  capture_output=True, text=True, timeout=5)
            print(f"NGC CLI available: {result.returncode == 0}")
            if result.returncode == 0:
                print(f"NGC CLI version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("NGC CLI not available in PATH")

        return True

    except Exception as e:
        print(f"NGC configuration test failed: {str(e)}")
        return False

def main():
    """Run all NVIDIA GPU integration tests"""
    print("Running NVIDIA GPU Integration Tests")
    print("=" * 50)

    tests = [
        ("GPU ML Model", test_gpu_ml_model),
        ("NVIDIA SMI Parsing", test_nvidia_smi_parsing),
        ("DCGM Parsing", test_dcgm_parsing),
        ("NGC Configuration", test_ngc_configuration)
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
        print("🎉 All NVIDIA GPU integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
