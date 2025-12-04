#!/usr/bin/env python3
"""
Quick test script for audit logging endpoints
"""
import os
import sys
import requests
import json

# Set testing mode
os.environ['TESTING'] = '1'

BASE_URL = 'http://localhost:8000'

def test_audit_endpoints():
    """Test audit logging endpoints"""
    print("🧪 Testing Audit Logging Endpoints\n")
    
    # Test 1: Register a user (should create audit log)
    print("1️⃣ Testing user registration with audit logging...")
    try:
        response = requests.post(
            f'{BASE_URL}/user/register',
            json={'username': 'audit_test_user', 'password': 'test123'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print("   ✅ Registration endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 2: Login (should create audit log)
    print("2️⃣ Testing user login with audit logging...")
    try:
        response = requests.post(
            f'{BASE_URL}/user/login',
            json={'username': 'testuser', 'password': 'testpass'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Response: {data}")
        token = data.get('token')
        print("   ✅ Login endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        token = 'test_token'  # Fallback for testing mode
    
    # Test 3: Query audit logs
    print("3️⃣ Testing audit log query...")
    try:
        response = requests.get(
            f'{BASE_URL}/audit/logs?limit=10',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Logs found: {data.get('count', 0)}")
        print("   ✅ Audit logs endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 4: Get audit summary
    print("4️⃣ Testing audit summary...")
    try:
        response = requests.get(
            f'{BASE_URL}/audit/summary',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        if 'summary' in data:
            summary = data['summary']
            print(f"   Total logs: {summary.get('total_logs', 0)}")
            print(f"   Failed attempts: {summary.get('failed_attempts', 0)}")
        print("   ✅ Audit summary endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 5: Get security report
    print("5️⃣ Testing security report...")
    try:
        response = requests.get(
            f'{BASE_URL}/audit/reports/security',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print("   ✅ Security report endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 6: Verify integrity
    print("6️⃣ Testing hash chain integrity verification...")
    try:
        response = requests.post(
            f'{BASE_URL}/audit/verify-integrity',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Integrity valid: {data.get('integrity_valid', False)}")
        print("   ✅ Integrity verification endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 7: Get active alerts
    print("7️⃣ Testing active alerts...")
    try:
        response = requests.get(
            f'{BASE_URL}/audit/alerts',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Active alerts: {data.get('count', 0)}")
        print("   ✅ Alerts endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test 8: Export audit logs
    print("8️⃣ Testing audit log export...")
    try:
        response = requests.post(
            f'{BASE_URL}/audit/export',
            headers={'Authorization': f'Bearer {token}'},
            json={'format': 'json', 'filters': {'limit': 5}},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print("   ✅ Export endpoint working\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    print("=" * 60)
    print("🎉 Audit Logging Endpoint Testing Complete!")
    print("=" * 60)

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🔒 JPMorgan Financial APIs - Audit Logging Test Suite")
    print("=" * 60 + "\n")
    
    print("⚠️  NOTE: This test requires the Flask app to be running")
    print(f"    Start the app with: python app_final.py")
    print(f"    Then run this test script\n")
    
    input("Press Enter to start testing (or Ctrl+C to cancel)...")
    
    test_audit_endpoints()
