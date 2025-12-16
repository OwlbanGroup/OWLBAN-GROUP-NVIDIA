#!/usr/bin/env python3
"""
Test script for business creation functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.database_fixed import db_manager
from src.schemas import BusinessCreate, AssetCreate

def test_business_creation():
    """Test creating a business"""
    print("Testing business creation...")

    # Test data
    business_data = {
        'name': 'Test Business Inc.',
        'type': 'corporation',
        'registration_number': '123456789',
        'address': '123 Test St, Test City, TC 12345',
        'contact_info': {
            'email': 'contact@testbusiness.com',
            'phone': '+1-555-0123',
            'website': 'https://testbusiness.com'
        }
    }

    try:
        # Validate with Pydantic schema
        business_schema = BusinessCreate(**business_data)
        print(f"✅ Schema validation passed: {business_schema}")

        # Create business
        business = db_manager.create_business(business_schema.dict())
        print(f"✅ Business created successfully!")
        print(f"   ID: {business.id}")
        print(f"   Name: {business.name}")
        print(f"   Type: {business.type}")

        # Test asset creation
        print("\nTesting asset creation...")
        asset_data = {
            'business_id': business.id,
            'name': 'Test Office Building',
            'type': 'real_estate',
            'value': 1000000.00,
            'acquisition_date': '2023-01-15T00:00:00Z',
            'description': 'Test asset for business'
        }

        # Validate with Pydantic schema
        asset_schema = AssetCreate(**asset_data)
        print(f"✅ Asset schema validation passed: {asset_schema}")

        # Create asset
        asset = db_manager.create_asset(asset_schema.dict())
        print(f"✅ Asset created successfully!")
        print(f"   ID: {asset.id}")
        print(f"   Name: {asset.name}")
        print(f"   Value: ${asset.value:,.2f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_business_creation()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
TE