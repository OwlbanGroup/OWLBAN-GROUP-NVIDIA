#!/usr/bin/env python3
"""
Test script for JPMorgan data filtering functionality
"""
import os
import sys
import requests
import json
from threading import Thread
import time

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set testing environment
os.environ['TESTING'] = '1'

def start_test_server():
    """Start the Flask app in testing mode"""
    from app_final import app
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

def test_filtering():
    """Test the filtering functionality"""
    time.sleep(2)  # Wait for server to start

    base_url = 'http://127.0.0.1:5001'

    print("Testing JPMorgan data filtering...")

    # Test 1: No filters (should return all data)
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Test 1 PASSED: No filters - returned all data")
            print(f"   Financial metrics: {len(data.get('financial_metrics', []))}")
            print(f"   Assets: {len(data.get('assets', []))}")
            print(f"   Stock tickers: {len(data.get('stock_tickers', []))}")
        else:
            print(f"❌ Test 1 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")

    # Test 2: Filter by env=prod
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data?env=prod', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            all_prod = all(item.get('env') == 'prod' for item in data.get('financial_metrics', []) +
                          data.get('assets', []) + data.get('stock_tickers', []))
            if all_prod:
                print("✅ Test 2 PASSED: env=prod filter working")
            else:
                print("❌ Test 2 FAILED: env=prod filter not working correctly")
        else:
            print(f"❌ Test 2 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")

    # Test 3: Filter by region=US
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data?region=US', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            all_us = all(item.get('region') == 'US' for item in data.get('financial_metrics', []) +
                        data.get('assets', []) + data.get('stock_tickers', []))
            if all_us:
                print("✅ Test 3 PASSED: region=US filter working")
            else:
                print("❌ Test 3 FAILED: region=US filter not working correctly")
        else:
            print(f"❌ Test 3 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")

    # Test 4: Filter by payment_type=Card
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data?payment_type=Card', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            all_card = all(item.get('payment_type') == 'Card' for item in data.get('assets', []) +
                          data.get('stock_tickers', []))
            if all_card:
                print("✅ Test 4 PASSED: payment_type=Card filter working")
            else:
                print("❌ Test 4 FAILED: payment_type=Card filter not working correctly")
        else:
            print(f"❌ Test 4 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")

    # Test 5: Filter by status=success
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data?status=success', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            all_success = all(item.get('status') == 'success' for item in data.get('assets', []) +
                             data.get('stock_tickers', []))
            if all_success:
                print("✅ Test 5 PASSED: status=success filter working")
            else:
                print("❌ Test 5 FAILED: status=success filter not working correctly")
        else:
            print(f"❌ Test 5 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")

    # Test 6: Multiple filters (env=prod&region=US)
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data?env=prod&region=US', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            all_prod_us = all(item.get('env') == 'prod' and item.get('region') == 'US'
                             for item in data.get('financial_metrics', []) +
                             data.get('assets', []) + data.get('stock_tickers', []))
            if all_prod_us:
                print("✅ Test 6 PASSED: Multiple filters (env=prod&region=US) working")
            else:
                print("❌ Test 6 FAILED: Multiple filters not working correctly")
        else:
            print(f"❌ Test 6 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")

    # Test 7: Invalid filter (should return empty or error)
    try:
        headers = {'Authorization': 'Bearer test_token'}
        response = requests.get(f'{base_url}/api/jpmorgan-data?env=invalid', headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            total_items = len(data.get('financial_metrics', [])) + len(data.get('assets', [])) + len(data.get('stock_tickers', []))
            if total_items == 0:
                print("✅ Test 7 PASSED: Invalid filter returns empty results")
            else:
                print("❌ Test 7 FAILED: Invalid filter should return empty results")
        else:
            print(f"❌ Test 7 FAILED: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 7 FAILED: {e}")

    print("\nTesting complete!")

if __name__ == '__main__':
    # Start server in background thread
    server_thread = Thread(target=start_test_server, daemon=True)
    server_thread.start()

    # Run tests
    test_filtering()
