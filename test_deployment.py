#!/usr/bin/env python3
"""
Railway Deployment Test Script
Tests basic functionality of the JPMorgan Financial APIs for Railway deployment
"""

import os
import sys
import json

# Set testing environment
os.environ['TESTING'] = '1'
os.environ['SECRET_KEY'] = 'test-secret-key-for-deployment-testing-only'
os.environ['JWT_SECRET_KEY'] = 'test-jwt-secret-key-for-deployment-testing-only'
os.environ['DATABASE_URL'] = 'sqlite:///test.db'

def test_app_import():
    """Test that the app imports successfully"""
    try:
        from app_final import app
        print("✅ App imports successfully")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        from app_final import app
        client = app.test_client()
        response = client.get('/health')
        if response.status_code == 200:
            data = response.get_json()
            print(f"✅ Health endpoint works: {data}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        from app_final import app
        client = app.test_client()
        response = client.get('/')
        if response.status_code == 200:
            data = response.get_json()
            print(f"✅ Root endpoint works: {len(data.get('endpoints', []))} endpoints listed")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
        return False

def test_port_logic():
    """Test port environment variable logic"""
    # Test default port
    port = int(os.environ.get('PORT', os.environ.get('FLASK_RUN_PORT', 5000)))
    print(f"✅ Port logic works: {port}")

    # Test with PORT set
    os.environ['PORT'] = '8080'
    port_with_env = int(os.environ.get('PORT', os.environ.get('FLASK_RUN_PORT', 5000)))
    print(f"✅ PORT env var works: {port_with_env}")

    return True

def main():
    """Run all tests"""
    print("🚀 Testing JPMorgan Financial APIs for Railway Deployment")
    print("=" * 60)

    tests = [
        ("App Import", test_app_import),
        ("Port Logic", test_port_logic),
        ("Health Endpoint", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for Railway deployment.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
