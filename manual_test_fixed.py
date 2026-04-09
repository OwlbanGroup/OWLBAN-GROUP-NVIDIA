#!/usr/bin/env python3
"""
Manual test script for the new endpoints
"""
import json
import requests
from app_final import app

def test_endpoints():
    """Test the new endpoints manually"""
    print("Testing new endpoints...")

    with app.test_client() as client:
        # Test /telemetry/export endpoint
        print("\n1. Testing /telemetry/export endpoint:")

        # Test basic export
        response = client.get('/telemetry/export')
        print(f"   Basic export: Status {response.status_code}")
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"   Response: {data['status']}, count: {data.get('count', 'N/A')}")

        # Test CSV export
        response = client.get('/telemetry/export?format=csv')
        print(f"   CSV export: Status {response.status_code}")
        if response.status_code == 200:
            print(f"   Content-Type: {response.content_type}")

        # Test with parameters
        response = client.get('/telemetry/export?limit=100&operation=test')
        print(f"   With params: Status {response.status_code}")

        # Test invalid limit
        response = client.get('/telemetry/export?limit=15000')
        print(f"   Invalid limit: Status {response.status_code}")

        # Test /ml/train endpoint
        print("\n2. Testing /ml/train endpoint:")

        # Test without auth (should fail)
        payload = {'training_data': [[1, 2, 3], [4, 5, 6]]}
        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json')
        print(f"   Without auth: Status {response.status_code}")

        # Test with invalid data
        response = client.post('/ml/train',
                                data=json.dumps({}),
                                content_type='application/json')
        print(f"   Invalid data: Status {response.status_code}")

        # Test with insufficient data
        payload = {'training_data': [[1, 2, 3]]}
        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json')
        print(f"   Insufficient data: Status {response.status_code}")

        print("\n3. Testing root endpoint documentation:")
        response = client.get('/')
        if response.status_code == 200:
            data = json.loads(response.data)
            endpoints = data.get('endpoints', [])
            print(f"   Listed endpoints: {len(endpoints)}")
            for endpoint in endpoints:
                if 'export' in endpoint or 'train' in endpoint:
                    print(f"   Found: {endpoint}")

    print("\nManual testing completed!")

if __name__ == '__main__':
    test_endpoints()

