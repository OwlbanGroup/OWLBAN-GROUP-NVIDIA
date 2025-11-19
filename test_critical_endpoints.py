#!/usr/bin/env python3
"""
Critical Path Testing Script
Tests key API endpoints to verify production readiness
"""
import json
from datetime import datetime

import requests

def test_endpoint(name, url, method='GET', data=None, headers=None):
    """Test a single endpoint"""
    try:
        if method == 'GET':
            response = requests.get(url, timeout=5)
        elif method == 'POST':
            response = requests.post(url, json=data, headers=headers, timeout=5)
        else:
            print(f"✗ ERROR | {name}")
            print(f"  Error: Invalid HTTP method: {method}")
            print()
            return False

        status = "✓ PASS" if response.status_code < 400 else "✗ FAIL"
        print(f"{status} | {name}")
        print(f"  Status: {response.status_code}")
        if response.status_code < 400:
            try:
                print(f"  Response: {json.dumps(response.json(), indent=2)[:200]}...")
            except (json.JSONDecodeError, ValueError):
                print(f"  Response: {response.text[:100]}...")
        else:
            print(f"  Error: {response.text[:200]}")
        print()
        return response.status_code < 400
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"✗ ERROR | {name}")
        print(f"  Error: {str(e)}")
        print()
        return False

def main():
    """Main test execution function"""
    print("=" * 60)
    print("CRITICAL PATH TESTING - JPMorgan Financial APIs")
    print("=" * 60)
    print(f"Test Time: {datetime.now().isoformat()}")
    print()

    base_url = "http://localhost:8000"
    results = []
    
    # Test 1: Health Check
    print("TEST 1: Health Check Endpoint")
    print("-" * 60)
    results.append(test_endpoint(
        "Health Check",
        f"{base_url}/health"
    ))
    
    # Test 2: Root Endpoint
    print("TEST 2: Root API Information")
    print("-" * 60)
    results.append(test_endpoint(
        "Root Endpoint",
        f"{base_url}/"
    ))
    
    # Test 3: Metrics Endpoint
    print("TEST 3: Prometheus Metrics")
    print("-" * 60)
    results.append(test_endpoint(
        "Metrics Endpoint",
        f"{base_url}/metrics"
    ))
    
    # Test 4: User Registration
    print("TEST 4: User Registration")
    print("-" * 60)
    results.append(test_endpoint(
        "User Registration",
        f"{base_url}/user/register",
        method='POST',
        data={
            "username": f"testuser_{datetime.now().timestamp()}",
            "password": "TestPass123!"
        }
    ))
    
    # Test 5: Swagger/API Docs
    print("TEST 5: Swagger Documentation")
    print("-" * 60)
    swagger_result = test_endpoint(
        "Swagger UI",
        f"{base_url}/swagger/"
    )
    if not swagger_result:
        # Try alternative endpoint
        swagger_result = test_endpoint(
            "Swagger UI (alt)",
            f"{base_url}/api/docs/"
        )
    results.append(swagger_result)
    
    # Test 6: Data Formats Info
    print("TEST 6: Data Formats Endpoint")
    print("-" * 60)
    results.append(test_endpoint(
        "Data Formats",
        f"{base_url}/data/formats"
    ))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print()
    
    if pass_rate >= 80:
        print("✓ CRITICAL PATH TESTING: PASSED")
        print("  The API is responding correctly to key endpoints.")
    else:
        print("✗ CRITICAL PATH TESTING: FAILED")
        print("  Some critical endpoints are not responding correctly.")
    
    print()
    print("=" * 60)
    return pass_rate >= 80

if __name__ == "__main__":
    SUCCESS = main()
    exit(0 if SUCCESS else 1)
