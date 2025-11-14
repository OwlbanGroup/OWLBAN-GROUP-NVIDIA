"""
API Examples and Testing Scripts
Run this file to test the API endpoints with sample data
"""
import requests
import json
import time
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:5000"
AUTH_TOKEN = os.getenv("API_TOKEN", "your_token_here")  # Replace with actual token

def get_auth_headers():
    """Get authorization headers"""
    return {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

def example_health_check():
    """Example: Health check"""
    print("=== Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_single_telemetry():
    """Example: Process single telemetry event"""
    print("=== Single Telemetry Processing ===")

    telemetry_data = {
        "ver": "4.0",
        "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
        "time": datetime.now(timezone.utc).isoformat(),
        "data": {
            "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
            "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
            "shell_id": 12345,
            "event_flags": 0,
            "pg_name": "StoreConfigurationServer",
            "dvc_sample": 1.0,
            "flags": 0,
            "edition": 1,
            "epoch": "2025-01-15T10:30:00.000Z",
            "seq": 1,
            "data_type": 1,
            "is_required": 1,
            "data_category": 1,
            "product": 1,
            "priv_tags": 0,
            "policies": 0,
            "cv": "1.0.0",
            "boot_id": 123456789,
            "os_name": "Windows",
            "os_version": "10.0.19045",
            "exp_id": "experiment_1",
            "app_id": "Microsoft.WindowsStore",
            "app_version": "22507.1401.7.0",
            "is_1p": 1,
            "as_id": 123,
            "local_id": "local_123",
            "device_class": "Desktop",
            "dev_make": "Microsoft",
            "dev_model": "Surface",
            "ticket_keys": json.dumps({"key1": "value1"}),
            "user_local_id": "user_123",
            "tz": "UTC",
            "pn1": "param1",
            "p1": "value1"
        },
        "ext": {
            "utc_seq": "2025-01-15T10:30:00.0000000Z"
        }
    }

    try:
        response = requests.post(
            f"{BASE_URL}/telemetry",
            json=telemetry_data,
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_batch_telemetry():
    """Example: Process batch telemetry events"""
    print("=== Batch Telemetry Processing ===")

    telemetry_batch = {
        "telemetry_data": [
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                    "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                    "shell_id": 12345,
                    "event_flags": 0
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                    "duration": 100,
                    "success": True
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                    "result": "Success"
                }
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/telemetry/batch",
            json=telemetry_batch,
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_telemetry_metrics():
    """Example: Get telemetry metrics"""
    print("=== Telemetry Metrics ===")
    try:
        response = requests.get(
            f"{BASE_URL}/telemetry/metrics?hours=24",
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_anomaly_detection():
    """Example: Anomaly detection"""
    print("=== Anomaly Detection ===")

    anomaly_data = {
        "telemetry_data": [
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "duration": 100,
                    "success": True
                }
            },
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "duration": 5000,  # Anomalous duration
                    "success": False   # Anomalous failure
                }
            }
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/ml/anomalies",
            json=anomaly_data,
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_train_ml_model():
    """Example: Train ML model"""
    print("=== Train ML Model ===")

    training_data = {
        "telemetry_data": [
            {
                "ver": "4.0",
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "time": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "duration": 100,
                    "success": True
                }
            },
            # Add more training examples...
        ]
    }

    try:
        response = requests.post(
            f"{BASE_URL}/ml/train",
            json=training_data,
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_export_telemetry():
    """Example: Export telemetry data"""
    print("=== Export Telemetry ===")
    try:
        response = requests.get(
            f"{BASE_URL}/telemetry/export?limit=100&format=json",
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Export successful - data received")
            # For large exports, you might want to save to file
            # with open('export.json', 'w') as f:
            #     json.dump(response.json(), f, indent=2)
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_websocket_status():
    """Example: WebSocket status"""
    print("=== WebSocket Status ===")
    try:
        response = requests.get(f"{BASE_URL}/ws/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_cloud_export():
    """Example: Export to cloud storage"""
    print("=== Cloud Storage Export ===")

    export_config = {
        "operation": "StoreConfigurationServer",
        "limit": 100,
        "format": "json",
        "providers": ["aws"],
        "filename_prefix": "telemetry_export"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/storage/export",
            json=export_config,
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_data_conversion():
    """Example: Data format conversion"""
    print("=== Data Format Conversion ===")

    conversion_data = {
        "data": [
            {
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "duration": 100,
                "success": True
            },
            {
                "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Operation",
                "duration": 200,
                "success": False
            }
        ],
        "from_format": "json",
        "to_format": "csv"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/data/convert",
            json=conversion_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Conversion successful")
            print("CSV Output:")
            print(response.text)
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_github_repos():
    """Example: Search GitHub repositories"""
    print("=== GitHub Repository Search ===")
    try:
        response = requests.get(
            f"{BASE_URL}/mcp/repos?query=telemetry&per_page=5",
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_github_issues():
    """Example: List GitHub issues"""
    print("=== GitHub Issues List ===")
    try:
        response = requests.get(
            f"{BASE_URL}/mcp/issues/microsoft/vscode?state=open&per_page=5",
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_create_github_issue():
    """Example: Create GitHub issue"""
    print("=== Create GitHub Issue ===")

    issue_data = {
        "title": "Test Issue from API",
        "body": "This is a test issue created via the JPMorgan Financial APIs",
        "assignees": []
    }

    try:
        response = requests.post(
            f"{BASE_URL}/mcp/issues/microsoft/vscode",
            json=issue_data,
            headers=get_auth_headers()
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def example_prometheus_metrics():
    """Example: Get Prometheus metrics"""
    print("=== Prometheus Metrics ===")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status: {response.status_code}")
        print("Metrics received (first 500 chars):")
        print(response.text[:500] + "...")
    except Exception as e:
        print(f"Error: {e}")
    print()

def run_all_examples():
    """Run all API examples"""
    print("🚀 JPMorgan Financial APIs - Testing Examples")
    print("=" * 50)

    # Basic endpoints (no auth required)
    example_health_check()
    example_websocket_status()
    example_data_conversion()
    example_prometheus_metrics()

    # Authenticated endpoints
    if AUTH_TOKEN != "your_token_here":
        example_single_telemetry()
        example_batch_telemetry()
        example_telemetry_metrics()
        example_anomaly_detection()
        example_train_ml_model()
        example_export_telemetry()
        example_cloud_export()
        example_github_repos()
        example_github_issues()
        # example_create_github_issue()  # Uncomment to test issue creation
    else:
        print("⚠️  Skipping authenticated endpoints - set API_TOKEN environment variable")
        print()

    print("✅ All examples completed!")

if __name__ == "__main__":
    run_all_examples()
