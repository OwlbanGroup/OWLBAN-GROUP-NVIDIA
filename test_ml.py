"""
Test script for ML integration in telemetry API
"""
import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_ml_anomalies():
    """Test the /ml/anomalies endpoint"""
    print("Testing /ml/anomalies endpoint...")

    # Sample telemetry data
    telemetry_data = [
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
            "time": "2025-09-22T19:42:10.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "dvc_sample": 0.5,
                "flags": 1,
                "edition": 1,
                "seq": 1,
                "data_type": 1,
                "is_required": True,
                "data_category": 1,
                "product": 1,
                "priv_tags": 1,
                "policies": 1,
                "cv": "1.0",
                "boot_id": 1,
                "os_name": "Windows",
                "os_version": "10.0",
                "exp_id": "1",
                "app_id": "1",
                "app_version": "1.0",
                "is_1p": 1,
                "as_id": 1,
                "local_id": "1",
                "device_class": "Desktop",
                "dev_make": "Microsoft",
                "dev_model": "Surface",
                "ticket_keys": [],
                "user_local_id": "1",
                "tz": "UTC",
                "pn1": "1",
                "p1": "1",
                "pn2": "2",
                "p2": "2",
                "pn3": "3",
                "p3": "3",
                "pn4": "4",
                "p4": "4"
            },
            "ext": {}
        }
    ] * 20  # Create 20 events for testing

    payload = {
        "telemetry_data": telemetry_data
    }

    try:
        response = requests.post(f"{BASE_URL}/ml/anomalies", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Anomalies detected: {result['anomaly_results']['anomalies_count']}")
            print("Test passed!")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_ml_train():
    """Test the /ml/train endpoint"""
    print("Testing /ml/train endpoint...")

    # Sample telemetry data for training
    telemetry_data = [
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
            "time": "2025-09-22T19:42:10.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "dvc_sample": 0.5,
                "flags": 1,
                "edition": 1,
                "seq": 1,
                "data_type": 1,
                "is_required": True,
                "data_category": 1,
                "product": 1,
                "priv_tags": 1,
                "policies": 1,
                "cv": "1.0",
                "boot_id": 1,
                "os_name": "Windows",
                "os_version": "10.0",
                "exp_id": "1",
                "app_id": "1",
                "app_version": "1.0",
                "is_1p": 1,
                "as_id": 1,
                "local_id": "1",
                "device_class": "Desktop",
                "dev_make": "Microsoft",
                "dev_model": "Surface",
                "ticket_keys": [],
                "user_local_id": "1",
                "tz": "UTC",
                "pn1": "1",
                "p1": "1",
                "pn2": "2",
                "p2": "2",
                "pn3": "3",
                "p3": "3",
                "pn4": "4",
                "p4": "4"
            },
            "ext": {}
        }
    ] * 50  # Create 50 events for training

    payload = {
        "telemetry_data": telemetry_data
    }

    try:
        response = requests.post(f"{BASE_URL}/ml/train", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Message: {result['message']}")
            print("Test passed!")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    """Run all ML tests"""
    print("Starting ML integration tests...")

    # Start the server in a separate process (assuming it's not running)
    print("Please ensure the server is running on http://localhost:5000")
    print("Run: python app.py")

    time.sleep(2)  # Wait for server to start

    # Test anomalies endpoint
    anomalies_success = test_ml_anomalies()

    # Test train endpoint
    train_success = test_ml_train()

    if anomalies_success and train_success:
        print("All ML tests passed!")
    else:
        print("Some ML tests failed!")

if __name__ == "__main__":
    main()
