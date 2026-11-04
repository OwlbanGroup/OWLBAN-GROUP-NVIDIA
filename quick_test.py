import subprocess
import sys
import os
import time
import urllib.request
import json

# Start the API server
print("Starting API server...")
server_proc = subprocess.Popen(
    [sys.executable, "main.py"],
    cwd=r"C:\Users\bsean\OneDrive\BLACKBOX-AI\.github\.github\.github\OSCAR-BROOME-REVENUE\.blackbox\OWLBAN-GROUP-NVIDIA",
    stdout=open("server_stdout.log", "w"),
    stderr=open("server_stderr.log", "w"),
)

# Wait for server to start - wait longer
print("Waiting for server to start...")
for i in range(15):
    time.sleep(1)
    try:
        url = "http://127.0.0.1:8000/gpu/status"
        r = urllib.request.urlopen(url, timeout=2)
        result = json.loads(r.read().decode())
        print(f"✅ Server is running after {i+1} seconds!")
        print(f"   Device: {result.get('device')}")
        print(f"   CUDA Available: {result.get('cuda_available')}")
        print(f"   Buffer Size: {result.get('experience_buffer_size')} entries")
        
        # Test RL action
        print("\nTesting RL action...")
        url = "http://127.0.0.1:8000/rl/action?state=0.5,-1.2,0.8,0.1"
        r = urllib.request.urlopen(url, timeout=10)
        result = json.loads(r.read().decode())
        print(f"✅ RL Action: {result.get('action')}")
        
        # Test RL learn
        print("\nTesting RL learn...")
        url = "http://127.0.0.1:8000/rl/learn?state=0.5,-1.2,0.8,0.1&action=optimize&reward=1.5&next_state=0.3,-0.8,0.6,0.2"
        r = urllib.request.urlopen(url, timeout=10)
        result = r.read().decode()
        print(f"✅ RL Learn Result: {result}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("=" * 60)
        break
    except Exception as e:
        if i == 14:
            print(f"❌ Server not responding after 15 seconds.")
            print("Server may have failed to start properly.")
            print(f"stdout log length: {len(open('server_stdout.log').read()) if os.path.exists('server_stdout.log') else 0}")
            print(f"stderr log length: {len(open('server_stderr.log').read()) if os.path.exists('server_stderr.log') else 0}")
        continue
# But the script will exit, which may cause the server to shut down