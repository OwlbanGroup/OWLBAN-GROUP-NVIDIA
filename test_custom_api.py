#!/usr/bin/env python3
"""
Custom API Call Testing Script
Demonstrates how to make custom API calls to JPMorgan Financial APIs
"""
import requests
import json
from datetime import datetime, timezone

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("=== Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_custom_telemetry():
    """Test custom telemetry processing"""
    print("=== Custom Telemetry Processing ===")

    # Custom telemetry data with your own fields
    telemetry_data = {
        "ver": "4.0",
        "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.CustomOperation",
        "time": datetime.now(timezone.utc).isoformat(),
        "data": {
            "Op": "CustomStoreOperation::ProcessUserData",
            "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
            "shell_id": 99999,
            "event_flags": 1,
            "pg_name": "CustomStoreServer",
            "dvc_sample": 1.0,
            "flags": 0,
            "edition": 1,
            "epoch": datetime.now(timezone.utc).isoformat(),
            "seq": 1,
            "data_type": 2,
            "is_required": 1,
            "data_category": 2,
            "product": 1,
            "priv_tags": 0,
            "policies": 0,
            "cv": "2.0.0",
            "boot_id": 987654321,
            "os_name": "Windows",
            "os_version": "11.0.22621",
            "exp_id": "custom_experiment_2025",
            "app_id": "Microsoft.WindowsStore.Custom",
            "app_version": "22507.1401.8.0",
            "is_1p": 1,
            "as_id": 456,
            "local_id": "custom_local_456",
            "device_class": "Laptop",
            "dev_make": "Dell",
            "dev_model": "XPS 13",
            "ticket_keys": json.dumps({"custom_key": "custom_value", "session_id": "abc123"}),
            "user_local_id": "custom_user_456",
            "tz": "EST",
            "pn1": "custom_param",
            "p1": "custom_value",
            # Add your own custom fields
            "custom_field_1": "custom_value_1",
            "custom_field_2": 42,
            "custom_metadata": {
                "source": "custom_api_test",
                "environment": "development",
                "tags": ["test", "custom", "api"]
            }
        },
        "ext": {
            "utc_seq": datetime.now(timezone.utc).isoformat(),
            "custom_ext_field": "additional_data"
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/telemetry", json=telemetry_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_batch_custom_telemetry():
    """Test batch processing with custom data"""
    print("=== Batch Custom Telemetry Processing ===")

    batch_data = {
        "telemetry_data": [
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.CustomOperation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": "CustomBatchOperation::ProcessMultipleItems",
                    "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                    "shell_id": 11111,
                    "batch_size": 3,
                    "custom_field": "batch_item_1"
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.CustomOperation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": "CustomBatchOperation::ProcessMultipleItems",
                    "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                    "shell_id": 22222,
                    "batch_size": 3,
                    "custom_field": "batch_item_2"
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.CustomOperation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": "CustomBatchOperation::ProcessMultipleItems",
                    "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                    "shell_id": 33333,
                    "batch_size": 3,
                    "custom_field": "batch_item_3"
                }
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/telemetry/batch", json=batch_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_data_conversion_custom():
    """Test data conversion with custom data"""
    print("=== Custom Data Conversion ===")

    custom_data = [
        {
            "event_type": "custom_telemetry",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": "user_12345",
            "action": "login",
            "metadata": {
                "device": "mobile",
                "browser": "chrome",
                "location": "US"
            }
        },
        {
            "event_type": "custom_telemetry",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": "user_67890",
            "action": "purchase",
            "metadata": {
                "device": "desktop",
                "browser": "firefox",
                "location": "UK"
            }
        }
    ]

    conversion_request = {
        "data": custom_data,
        "from_format": "json",
        "to_format": "csv"
    }

    try:
        response = requests.post(f"{BASE_URL}/data/convert", json=conversion_request)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("CSV Output:")
            print(response.text)
        else:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_anomaly_detection_custom():
    """Test anomaly detection with custom data"""
    print("=== Custom Anomaly Detection ===")

    anomaly_test_data = {
        "telemetry_data": [
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "duration": 100,
                    "success": True,
                    "custom_metric": 50
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "duration": 5000,  # Anomalous duration
                    "success": False,  # Anomalous failure
                    "custom_metric": 500  # Anomalous metric
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "duration": 150,
                    "success": True,
                    "custom_metric": 75
                }
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/ml/anomalies", json=anomaly_test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def main():
    """Run all custom API tests"""
    print("🚀 JPMorgan Financial APIs - Custom API Call Testing")
    print("=" * 60)

    # Test basic endpoints (no auth required)
    test_health_check()
    test_data_conversion_custom()

    # Test authenticated endpoints (may fail without proper token)
    test_custom_telemetry()
    test_batch_custom_telemetry()
    test_anomaly_detection_custom()

    print("✅ Custom API testing completed!")
    print("\n📝 Note: Some endpoints require authentication tokens.")
    print("Set API_TOKEN environment variable for full testing.")

if __name__ == "__main__":
    main()
