#!/usr/bin/env python3
"""
Reinforcement Learning API Test Script
Tests the RL endpoints: /rl/action, /rl/learn, /gpu/status

Run instructions:
1. Start the API server: python main.py
2. Set credentials: set API_PASSWORD=testpassword123
3. Run tests: python test_rl.py
"""

import urllib.request
import json

BASE_URL = "http://owlban_admin:testpassword123@127.0.0.1:8000"
"""Base URL with auth credentials"""


def test_gpu_status():
    """Test GPU status endpoint"""
    print("=" * 60)
    print("TEST 1: GPU Status")
    print("=" * 60)
    try:
        url = f"{BASE_URL}/gpu/status"
        r = urllib.request.urlopen(url, timeout=5)
        result = json.loads(r.read().decode())
        print(f"✅ Device: {result.get('device')}")
        print(f"✅ CUDA Available: {result.get('cuda_available')}")
        print(f"✅ Memory Allocated: {result.get('memory_allocated', 'N/A')} GB")
        print(f"✅ Memory Reserved: {result.get('memory_reserved', 'N/A')} GB")
        print(f"✅ Buffer Size: {result.get('experience_buffer_size')} entries")
        print()
        return True
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code}")
        print(e.read().decode()[:200])
        print()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return False


def test_rl_action(state="0.5,-1.2,0.8,0.1"):
    """Test RL action selection endpoint"""
    print("=" * 60)
    print("TEST 2: Get RL Action")
    print("=" * 60)
    try:
        url = f"{BASE_URL}/rl/action?state={state}"
        r = urllib.request.urlopen(url, timeout=10)
        result = json.loads(r.read().decode())
        action = result.get("action")
        print(f"✅ State: {state}")
        print(f"✅ Action: {action}")
        print()
        return True
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code}")
        print(e.read().decode()[:200])
        print()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return False


def test_rl_learn(state="0.5,-1.2,0.8,0.1", action="optimize", reward=1.5, next_state="0.3,-0.8,0.6,0.2"):
    """Test RL learning endpoint"""
    print("=" * 60)
    print("TEST 3: Learn Transition")
    print("=" * 60)
    try:
        url = f"{BASE_URL}/rl/learn?state={state}&action={action}&reward={reward}&next_state={next_state}"
        r = urllib.request.urlopen(url, timeout=10)
        result = r.read().decode()
        print(f"✅ Input: state={state}, action={action}, reward={reward}, next_state={next_state}")
        print(f"✅ Result: {result}")
        print()
        return True
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code}")
        print(e.read().decode()[:200])
        print()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return False


def main():
    """Run all RL tests"""
    print("\n" + "=" * 60)
    print("REINFORCEMENT LEARNING API TEST SUITE")
    print("=" * 60)
    print(f"Server: {BASE_URL}")
    print("Credentials: owlban_admin / testpassword123")
    print()
    
    results = []
    results.append(("GPU Status", test_gpu_status()))
    results.append(("Get RL Action", test_rl_action()))
    results.append(("Learn Transition", test_rl_learn()))
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print()
    if passed == total:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check the server is running and credentials are correct.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)