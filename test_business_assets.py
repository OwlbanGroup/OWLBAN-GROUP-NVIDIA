#!/usr/bin/env python3
"""
Test script for business and asset management endpoints
"""
import os
import sys
import json
import requests
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_business_endpoints():
    """Test business CRUD endpoints"""
    print("Testing Business Endpoints...")

    # Set testing environment
    os.environ['TESTING'] = '1'

    # Import after setting environment
    from app_final import app
    from src.database_fixed import db_manager

    # Initialize database
    db_manager.init_db()

    with app.test_client() as client:
        # Test 1: Create business
        print("1. Creating business...")
        business_data = {
            'name': 'Test Corporation',
            'type': 'corporation',
            'registration_number': '123456789',
            'address': '123 Test St, Test City, TC 12345',
            'contact_info': {
                'email': 'contact@testcorp.com',
                'phone': '+1-555-0123'
            }
        }

        response = client.post('/businesses',
                             json=business_data,
                             headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            business_response = response.get_json()
            business_id = business_response['business']['id']
            print(f"   Created business with ID: {business_id}")
        else:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 2: Get business
        print("2. Getting business...")
        response = client.get(f'/businesses/{business_id}',
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 3: Update business
        print("3. Updating business...")
        update_data = {
            'name': 'Updated Test Corporation',
            'contact_info': {
                'email': 'updated@testcorp.com',
                'phone': '+1-555-0124'
            }
        }
        response = client.put(f'/businesses/{business_id}',
                            json=update_data,
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 4: List businesses
        print("4. Listing businesses...")
        response = client.get('/businesses',
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 5: Delete business
        print("5. Deleting business...")
        response = client.delete(f'/businesses/{business_id}',
                               headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

    print("Business endpoints test completed successfully!")
    return True

def test_asset_endpoints():
    """Test asset CRUD endpoints"""
    print("\nTesting Asset Endpoints...")

    # Set testing environment
    os.environ['TESTING'] = '1'

    # Import after setting environment
    from app_final import app
    from src.database_fixed import db_manager

    # Initialize database
    db_manager.init_db()

    with app.test_client() as client:
        # First create a business for the asset
        business_data = {
            'name': 'Asset Test Corp',
            'type': 'corporation',
            'registration_number': '987654321',
            'address': '456 Asset St, Asset City, AC 67890',
            'contact_info': {
                'email': 'assets@testcorp.com',
                'phone': '+1-555-0567'
            }
        }

        response = client.post('/businesses',
                             json=business_data,
                             headers={'Authorization': 'Bearer test_token'})
        business_id = response.get_json()['business']['id']

        # Test 1: Create asset
        print("1. Creating asset...")
        asset_data = {
            'business_id': business_id,
            'name': 'Office Building',
            'type': 'real_estate',
            'value': 1000000.00,
            'acquisition_date': '2023-01-15',
            'current_value': 1100000.00,
            'ownership_percentage': 100.0,
            'description': 'Main office building'
        }

        response = client.post('/assets',
                             json=asset_data,
                             headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            asset_response = response.get_json()
            asset_id = asset_response['asset']['id']
            print(f"   Created asset with ID: {asset_id}")
        else:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 2: Get asset
        print("2. Getting asset...")
        response = client.get(f'/assets/{asset_id}',
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 3: Update asset
        print("3. Updating asset...")
        update_data = {
            'current_value': 1200000.00,
            'description': 'Updated main office building'
        }
        response = client.put(f'/assets/{asset_id}',
                            json=update_data,
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 4: List assets
        print("4. Listing assets...")
        response = client.get('/assets',
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 5: Get business assets
        print("5. Getting business assets...")
        response = client.get(f'/businesses/{business_id}/assets',
                            headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 6: Add asset to business
        print("6. Adding asset to business...")
        new_asset_data = {
            'business_id': business_id,
            'name': 'Company Car',
            'type': 'equipment',
            'value': 50000.00,
            'acquisition_date': '2023-06-01',
            'current_value': 45000.00,
            'ownership_percentage': 100.0,
            'description': 'Executive company car'
        }

        response = client.post(f'/businesses/{business_id}/assets',
                             json=new_asset_data,
                             headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 201:
            print(f"   Error: {response.get_json()}")
            return False

        # Test 7: Delete asset
        print("7. Deleting asset...")
        response = client.delete(f'/assets/{asset_id}',
                               headers={'Authorization': 'Bearer test_token'})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.get_json()}")
            return False

    print("Asset endpoints test completed successfully!")
    return True

if __name__ == '__main__':
    print("Starting Business and Asset Management Tests...")

    business_success = test_business_endpoints()
    asset_success = test_asset_endpoints()

    if business_success and asset_success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
