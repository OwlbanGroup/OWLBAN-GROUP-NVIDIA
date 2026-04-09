#!/usr/bin/env python3
"""
Simple test script to verify the fixes made to the JPMorgan Financial APIs
"""
import requests
import json
import time
import threading
from app_final import app

def test_basic_functionality():
    """Test basic functionality of the fixed application"""
    print("Testing basic functionality...")

    # Test data
    sample_data = {
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

    with app.test_client() as client:
        # Test health check
        print("1. Testing health check...")
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        print("   ✓ Health check passed")

        # Test single telemetry processing
        print("2. Testing single telemetry processing...")
        response = client.post('/telemetry', json=sample_data)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        print("   ✓ Single telemetry processing passed")

        # Test metrics endpoint
        print("3. Testing metrics endpoint...")
        response = client.get('/telemetry/metrics?hours=24')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        print("   ✓ Metrics endpoint passed")

        # Test ML train endpoint
        print("4. Testing ML train endpoint...")
        response = client.post('/ml/train', json={"telemetry_data": [sample_data]})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        print("   ✓ ML train endpoint passed")

        # Test telemetry export endpoint
        print("5. Testing telemetry export endpoint...")
        response = client.get('/telemetry/export?limit=10&format=json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        print("   ✓ Telemetry export endpoint passed")

        # Test error handling for invalid JSON
        print("6. Testing error handling for invalid JSON...")
        response = client.post('/telemetry', data='invalid json', content_type='application/json')
        assert response.status_code == 400  # Should be 400, not 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        print("   ✓ Error handling for invalid JSON passed (returns 400)")

        # Test error handling for missing data
        print("7. Testing error handling for missing data...")
        response = client.post('/telemetry', data=json.dumps({}), content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        print("   ✓ Error handling for missing data passed")

        print("\n🎉 All tests passed! The fixes are working correctly.")

if __name__ == "__main__":
    test_basic_functionality()

