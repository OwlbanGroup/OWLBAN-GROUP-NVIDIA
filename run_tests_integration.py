import subprocess
import time
import sys
import os
import urllib.request
import json

# Start the API server in the background
env = {}
env["API_PASSWORD"] = "testpassword123"

process = subprocess.Popen(
    [sys.executable, "main.py"],
    env={**os.environ, "API_PASSWORD": "testpassword123"},
    cwd=r"C:\Users\bsean\OneDrive\BLACKBOX-AI\.github\.github\.github\OSCAR-BROOME-REVENUE\.blackbox\OWLBAN-GROUP-NVIDIA",
    stdout=open("server_stdout.log", "w"),
    stderr=open("server_stderr.log", "w"),
)

# Wait for server to start
time.sleep(3)

# Test if server is running
try:
    url = "http://127.0.0.1:8000/gpu/status"
    r = urllib.request.urlopen(url, timeout=5)
    result = json.loads(r.read().decode())
    print(f"Server is running: {result}")
except Exception as e:
    print(f"Server not responding: {e}")
    process.terminate()
    sys.exit(1)

# Run the RL tests
print("\n" + "=" * 60)
print("REINFORCEMENT LEARNING API TEST SUITE")
print("=" * 60)
print("Server: http://127.0.0.1:8000")
print("Credentials: owlban_admin / testpassword123")
print()

results = []

# TEST 1: GPU Status
print("=" * 60)
print("TEST 1: GPU Status")
print("=" * 60)
try:
    url = "http://127.0.0.1:8000/gpu/status"
    r = urllib.request.urlopen(url, timeout=5)
    result = json.loads(r.read().decode())
    print(f"✅ Device: {result.get('device')}")
    print(f"✅ CUDA Available: {result.get('cuda_available')}")
    print(f"✅ Memory Allocated: {result.get('memory_allocated', 'N/A')} GB")
    print(f"✅ Memory Reserved: {result.get('memory_reserved', 'N/A')} GB")
    print(f"✅ Buffer Size: {result.get('experience_buffer_size')} entries")
    print()
    results.append(("GPU Status", True))
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    results.append(("GPU Status", False))

# TEST 2: Get RL Action
print("=" * 60)
print("TEST 2: Get RL Action")
print("=" * 60)
try:
    url = "http://127.0.0.1:8000/rl/action?state=0.5,-1.2,0.8,0.1"
    r = urllib.request.urlopen(url, timeout=10)
    result = json.loads(r.read().decode())
    action = result.get("action")
    print("✅ State: 0.5,-1.2,0.8,0.1")
    print(f"✅ Action: {action}")
    print()
    results.append(("Get RL Action", True))
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    results.append(("Get RL Action", False))

# TEST 3: Learn Transition
print("=" * 60)
print("TEST 3: Learn Transition")
print("=" * 60)
try:
    url = "http://127.0.0.1:8000/rl/learn?state=0.5,-1.2,0.8,0.1&action=optimize&reward=1.5&next_state=0.3,-0.8,0.6,0.2"
    r = urllib.request.urlopen(url, timeout=10)
    result = r.read().decode()
    print("✅ Input: state=0.5,-1.2,0.8,0.1, action=optimize, reward=1.5, next_state=0.3,-0.8,0.6,0.2")
    print(f"✅ Result: {result}")
    print()
    results.append(("Learn Transition", True))
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    results.append(("Learn Transition", False))

# Print summary
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

# Keep server running for a bit longer then terminate
time.sleep(5)
process.terminate()
print("\nServer stopped.")