import pytest
import requests
import json
import time
import os
import tempfile
from threading import Thread
from app_final import app as flask_app, limiter

# Sample telemetry data for testing
SAMPLE_TELEMETRY_DATA = {
    "ver": "4.0",
    "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
    "time": "2025-09-22T19:42:10.2549325Z",
    "data": {
        "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
        "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
        "OS": "Windows 11",
        "DeviceModel": "Surface Pro 9",
        "UserId": "test_user_123"
    },
    "ext": {
        "flags": 1,
        "privacy": "public"
    }
}

SAMPLE_BATCH_DATA = {
    "telemetry_data": [
        SAMPLE_TELEMETRY_DATA,
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
            "time": "2025-09-22T19:42:11.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_456"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        }
    ]
}

# Large batch data for ML training
LARGE_BATCH_DATA = {
    "telemetry_data": [
        SAMPLE_TELEMETRY_DATA,
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
            "time": "2025-09-22T19:42:11.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_456"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Error",
            "time": "2025-09-22T19:42:12.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "Surface Pro 9",
                "UserId": "test_user_789",
                "ErrorCode": "0x80070005"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
            "time": "2025-09-22T19:42:13.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::GetStoreConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 10",
                "DeviceModel": "Dell XPS 13",
                "UserId": "test_user_101"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
            "time": "2025-09-22T19:42:14.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::GetStoreConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 10",
                "DeviceModel": "Dell XPS 13",
                "UserId": "test_user_101"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
            "time": "2025-09-22T19:42:15.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::UpdateStoreConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "HP Spectre",
                "UserId": "test_user_202"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
            "time": "2025-09-22T19:42:16.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::UpdateStoreConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "HP Spectre",
                "UserId": "test_user_202"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.Error",
            "time": "2025-09-22T19:42:17.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::UpdateStoreConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "HP Spectre",
                "UserId": "test_user_202",
                "ErrorCode": "0x80070005"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
            "time": "2025-09-22T19:42:18.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::ValidateConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 10",
                "DeviceModel": "Lenovo ThinkPad",
                "UserId": "test_user_303"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.EndOperation",
            "time": "2025-09-22T19:42:19.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::ValidateConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 10",
                "DeviceModel": "Lenovo ThinkPad",
                "UserId": "test_user_303"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        },
        {
            "ver": "4.0",
            "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
            "time": "2025-09-22T19:42:20.2549325Z",
            "data": {
                "Op": "StoreConfigurationServer::SyncConfigurationAsync",
                "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
                "OS": "Windows 11",
                "DeviceModel": "ASUS ROG",
                "UserId": "test_user_404"
            },
            "ext": {
                "flags": 1,
                "privacy": "public"
            }
        }
    ]
}

@pytest.fixture(scope='module')
def test_client():
    """Fixture to start the Flask app in test mode"""
    flask_app.config['TESTING'] = True
    limiter.enabled = False
    with flask_app.test_client() as client:
        yield client

def test_health_check(test_client):
    """Test the health check endpoint"""
    response = test_client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data
    assert 'version' in data

def test_single_telemetry_processing(test_client):
    """Test processing a single telemetry event"""
    response = test_client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'message' in data
    assert 'timestamp' in data

def test_batch_telemetry_processing(test_client):
    """Test processing a batch of telemetry events"""
    response = test_client.post('/telemetry/batch',
                                data=json.dumps(SAMPLE_BATCH_DATA),
                                content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'message' in data
    assert 'statistics' in data
    assert 'timestamp' in data

def test_telemetry_metrics(test_client):
    """Test retrieving telemetry metrics"""
    response = test_client.get('/telemetry/metrics?hours=24')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'metrics' in data
    assert 'timestamp' in data

def test_ml_anomaly_detection(test_client):
    """Test ML anomaly detection"""
    response = test_client.post('/ml/anomalies',
                                data=json.dumps(SAMPLE_BATCH_DATA),
                                content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'anomaly_results' in data
    assert 'timestamp' in data

def test_ml_model_training(test_client):
    """Test ML model training"""
    response = test_client.post('/ml/train', json=LARGE_BATCH_DATA)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'message' in data
    assert 'timestamp' in data

def test_telemetry_export(test_client):
    """Test telemetry data export"""
    response = test_client.get('/telemetry/export?limit=10&format=json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'events' in data
    assert 'count' in data
    assert 'timestamp' in data

def test_error_handling_invalid_json(test_client):
    """Test error handling for invalid JSON"""
    response = test_client.post('/telemetry',
                                data='invalid json',
                                content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'error' in data

def test_error_handling_missing_data(test_client):
    """Test error handling for missing telemetry data"""
    response = test_client.post('/telemetry',
                                data=json.dumps({}),
                                content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'error' in data

def test_full_e2e_flow(test_client):
    """Test the full end-to-end flow"""
    # Step 1: Health check
    response = test_client.get('/health')
    assert response.status_code == 200

    # Step 2: Process single telemetry
    response = test_client.post('/telemetry',
                                data=json.dumps(SAMPLE_TELEMETRY_DATA),
                                content_type='application/json')
    assert response.status_code == 200

    # Step 3: Process batch
    response = test_client.post('/telemetry/batch',
                                data=json.dumps(SAMPLE_BATCH_DATA),
                                content_type='application/json')
    assert response.status_code == 200

    # Step 4: Get metrics
    response = test_client.get('/telemetry/metrics?hours=24')
    assert response.status_code == 200

    # Step 5: Detect anomalies
    response = test_client.post('/ml/anomalies',
                                data=json.dumps(SAMPLE_BATCH_DATA),
                                content_type='application/json')
    assert response.status_code == 200

    # Step 6: Train model
    response = test_client.post('/ml/train',
                                data=json.dumps(LARGE_BATCH_DATA),
                                content_type='application/json')
    assert response.status_code == 200

    # Step 7: Export data
    response = test_client.get('/telemetry/export?limit=10&format=json')
    assert response.status_code == 200

    print("Full E2E flow completed successfully!")
