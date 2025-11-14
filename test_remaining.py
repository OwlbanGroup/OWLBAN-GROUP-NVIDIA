#!/usr/bin/env python3
"""
Test remaining areas: Performance, Integration, Security, and Load Testing
"""
import os
os.environ['TESTING'] = '1'  # Set testing mode before importing app

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from app_final import app

# Sample data for testing
SAMPLE_DATA = {
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

def test_performance():
    """Test API performance"""
    results = []
    with app.test_client() as client:
        # Test response times for key endpoints
        endpoints = [
            ('/health', 'GET', None),
            ('/telemetry', 'POST', SAMPLE_DATA),
            ('/telemetry/metrics?hours=24', 'GET', None),
            ('/ml/train', 'POST', {"telemetry_data": [SAMPLE_DATA]}),
            ('/telemetry/export?limit=10', 'GET', None)
        ]

        for endpoint, method, data in endpoints:
            start_time = time.time()
            if method == 'GET':
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=data)
            end_time = time.time()

            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            results.append({
                'endpoint': endpoint,
                'method': method,
                'status_code': response.status_code,
                'response_time_ms': round(response_time, 2),
                'performance': 'GOOD' if response_time < 500 else 'SLOW'
            })

    return results

def test_load_concurrent_requests():
    """Test concurrent load handling"""
    def make_request():
        with app.test_client() as client:
            response = client.post('/telemetry', json=SAMPLE_DATA)
            return response.status_code

    # Test with 10 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [future.result() for future in futures]

    success_count = sum(1 for status in results if status == 200)
    return {
        'total_requests': len(results),
        'successful_requests': success_count,
        'success_rate': f"{(success_count/len(results))*100:.1f}%"
    }

def test_security_rate_limiting():
    """Test security features like rate limiting"""
    # Temporarily disable testing mode to enable rate limiting
    original_testing = app.config.get('TESTING', False)
    app.config['TESTING'] = False

    results = []
    try:
        with app.test_client() as client:
            # Test rate limiting by making multiple rapid requests
            for i in range(15):  # Exceed typical rate limits
                response = client.post('/telemetry', json=SAMPLE_DATA, headers={'Authorization': 'Bearer dummy_token'})
                results.append({
                    'request': i+1,
                    'status_code': response.status_code,
                    'limited': response.status_code == 429
                })
                time.sleep(0.1)  # Small delay between requests
    finally:
        # Restore testing mode
        app.config['TESTING'] = original_testing

    rate_limited_requests = sum(1 for r in results if r['limited'])
    return {
        'total_requests': len(results),
        'rate_limited': rate_limited_requests,
        'rate_limiting_working': rate_limited_requests > 0
    }

def test_integration_websocket():
    """Test WebSocket integration (basic connectivity)"""
    try:
        from src.websocket_manager import WebSocketManager
        ws_manager = WebSocketManager()
        # Basic instantiation test
        return {
            'websocket_manager': 'instantiated',
            'status': 'success'
        }
    except Exception as e:
        return {
            'websocket_manager': 'failed',
            'error': str(e),
            'status': 'error'
        }

def test_integration_cloud_storage():
    """Test cloud storage integration"""
    try:
        from src.cloud_storage import CloudStorageManager
        storage_manager = CloudStorageManager()
        # Basic instantiation test
        return {
            'cloud_storage': 'instantiated',
            'status': 'success'
        }
    except Exception as e:
        return {
            'cloud_storage': 'failed',
            'error': str(e),
            'status': 'error'
        }

def test_error_handling_comprehensive():
    """Comprehensive error handling test"""
    test_cases = [
        ('Invalid JSON', 'invalid json', 400),
        ('Empty payload', {}, 400),
        ('Missing required fields', {'ver': '4.0'}, 400),
        ('Invalid endpoint', '/nonexistent', 404),
    ]

    results = []
    with app.test_client() as client:
        for test_name, payload, expected_status in test_cases:
            if test_name == 'Invalid endpoint':
                response = client.get(payload)
            else:
                response = client.post('/telemetry', json=payload)
            results.append({
                'test': test_name,
                'expected_status': expected_status,
                'actual_status': response.status_code,
                'passed': response.status_code == expected_status
            })

    return results

def run_all_remaining_tests():
    """Run all remaining test categories"""
    print("🚀 Running Remaining Test Areas for JPMorgan Financial APIs")
    print("=" * 70)

    # Enable testing mode to skip auth and rate limiting
    app.config['TESTING'] = True

    test_results = {}

    # Performance Testing
    print("\n📊 Performance Testing...")
    perf_results = test_performance()
    test_results['performance'] = perf_results
    print(f"✅ Tested {len(perf_results)} endpoints for response times")

    # Load Testing
    print("\n⚡ Load Testing (Concurrent Requests)...")
    load_results = test_load_concurrent_requests()
    test_results['load'] = load_results
    print(f"✅ Tested concurrent requests: {load_results['success_rate']} success rate")

    # Security Testing
    print("\n🔒 Security Testing (Rate Limiting)...")
    security_results = test_security_rate_limiting()
    test_results['security'] = security_results
    print(f"✅ Rate limiting test: {'Working' if security_results['rate_limiting_working'] else 'Not detected'}")

    # Integration Testing
    print("\n🔗 Integration Testing...")
    ws_results = test_integration_websocket()
    cloud_results = test_integration_cloud_storage()
    test_results['integration'] = {
        'websocket': ws_results,
        'cloud_storage': cloud_results
    }
    print(f"✅ WebSocket: {ws_results['status']}")
    print(f"✅ Cloud Storage: {cloud_results['status']}")

    # Comprehensive Error Handling
    print("\n❌ Comprehensive Error Handling Testing...")
    error_results = test_error_handling_comprehensive()
    test_results['error_handling'] = error_results
    passed_errors = sum(1 for r in error_results if r['passed'])
    print(f"✅ Error handling: {passed_errors}/{len(error_results)} tests passed")

    # Save results to file
    with open('remaining_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("📋 REMAINING AREAS TEST SUMMARY")
    print("=" * 70)

    all_passed = True

    # Performance check
    slow_endpoints = [r for r in perf_results if r['performance'] == 'SLOW']
    if slow_endpoints:
        print(f"⚠️  Performance: {len(slow_endpoints)} endpoints with slow response times")
        all_passed = False
    else:
        print("✅ Performance: All endpoints within acceptable response times")

    # Load check
    if load_results['success_rate'] == '100.0%':
        print("✅ Load: All concurrent requests handled successfully")
    else:
        print(f"⚠️  Load: {load_results['success_rate']} success rate for concurrent requests")
        all_passed = False

    # Security check
    if security_results['rate_limiting_working']:
        print("✅ Security: Rate limiting is working")
    else:
        print("⚠️  Security: Rate limiting not detected")
        all_passed = False

    # Integration check
    if ws_results['status'] == 'success' and cloud_results['status'] == 'success':
        print("✅ Integration: All services instantiated successfully")
    else:
        print("⚠️  Integration: Some services failed to initialize")
        all_passed = False

    # Error handling check
    if passed_errors == len(error_results):
        print("✅ Error Handling: All error scenarios handled correctly")
    else:
        print(f"⚠️  Error Handling: {len(error_results) - passed_errors} error scenarios failed")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL REMAINING TESTS PASSED! Project is 100% PERFECT!")
        return True
    else:
        print("⚠️  Some tests had issues. Review results above for details.")
        return False

if __name__ == "__main__":
    success = run_all_remaining_tests()
    exit(0 if success else 1)
