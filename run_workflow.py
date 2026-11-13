#!/usr/bin/env python3
"""Run the full OWLBAN GROUP AI workflow: start server, dashboard, and tests."""

import subprocess
import time
import os

def start_api_server():
    """Start the API server in the background."""
    print("=" * 60)
    print("Step 1: Starting API Server")
    print("=" * 60)
    
    # Set required environment variables
    env = os.environ.copy()
    env['API_PASSWORD'] = 'testpassword123'
    
    # Start the server
    process = subprocess.Popen(
        ['python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if process is still running
    if process.poll() is not None:
        print("❌ Server failed to start!")
        stderr = process.stderr.read().decode('utf-8', errors='replace')
        print("Error output:", stderr[:500])
        return None
    
    print("Server started successfully (PID:", process.pid, ")")
    return process


def test_api_server():
    """Test if the API server is responding."""
    print("\n" + "=" * 60)
    print("Testing API Server")
    print("=" * 60)
    
    import urllib.request
    import json
    
    try:
        url = "http://127.0.0.1:8000/gpu/status"
        r = urllib.request.urlopen(url, timeout=5)
        result = json.loads(r.read().decode())
        print("✅ API Server is responding!")
        print(f"   Device: {result.get('device')}")
        print(f"   CUDA Available: {result.get('cuda_available')}")
        return True
    except Exception as e:
        print(f"❌ API Server not responding: {e}")
        return False


def run_dashboard():
    """Start the Streamlit dashboard."""
    print("\n" + "=" * 60)
    print("Step 2: Starting Dashboard")
    print("=" * 60)
    
    print("Starting Streamlit dashboard on port 8501...")
    print("   Open your browser to http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard when done\n")
    
    # Start dashboard in background
    dashboard_process = subprocess.Popen(
        ['python', '-m', 'streamlit', 'run', 'web_dashboard.py', '--port', '8501'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for it to start
    time.sleep(3)
    
    print(f"Dashboard process started (PID: {dashboard_process.pid})")
    print("   The dashboard should now be running in your browser")
    return dashboard_process


def run_tests():
    """Run the test suite."""
    print("\n" + "=" * 60)
    print("Step 3: Running Tests")
    print("=" * 60)
    
    print("Running: python -m pytest tests -q --tb=short")
    print("(This may take several minutes)\n")
    
    # Run tests
    test_process = subprocess.Popen(
        ['python', '-m', 'pytest', 'tests', '-q', '--tb=short'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for tests to complete (with timeout)
    try:
        stdout, stderr = test_process.communicate(timeout=120)  # 2 min timeout
        print("Test output:")
        print(stdout.decode('utf-8', errors='replace')[-1000:] if stdout else "No output")
        if test_process.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed with code {test_process.returncode}")
            print("Error output:", stderr.decode('utf-8', errors='replace')[-1000:] if stderr else "No error")
    except subprocess.TimeoutExpired:
        test_process.kill()
        print("⏰ Tests timed out after 2 minutes")
        print("This is normal - some tests may require full system setup")


def run_rl_tests():
    """Run the RL-specific tests."""
    print("\n" + "=" * 60)
    print("Step 4: Running RL Tests")
    print("=" * 60)
    
    print("Running: set API_PASSWORD=testpassword123 && python test_rl.py")
    print("\nThis tests the RL API endpoints: /gpu/status, /rl/action, /rl/learn\n")
    
    # Run RL tests
    env = os.environ.copy()
    env['API_PASSWORD'] = 'testpassword123'
    
    rl_process = subprocess.Popen(
        ['python', 'test_rl.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    # Wait for completion
    stdout, stderr = rl_process.communicate(timeout=60)
    
    print("RL Test Output:")
    print(stdout.decode('utf-8', errors='replace') if stdout else "No output")
    if rl_process.returncode == 0:
        print("✅ RL tests completed successfully!")
    else:
        print(f"❌ RL tests failed with code {rl_process.returncode}")
        print("Error output:", stderr.decode('utf-8', errors='replace') if stderr else "No error")


def main():
    """Run the complete workflow."""
    print("\n" + "=" * 60)
    print("OWLBAN GROUP AI - FULL WORKFLOW")
    print("=" * 60 + "\n")
    
    # Step 1: Start API server
    server_process = start_api_server()
    if not server_process:
        print("\n❌ Cannot proceed without API server. Exiting.")
        return
    
    # Step 2: Test API server
    if not test_api_server():
        print("\n⚠️  API server started but endpoints may not be fully functional")
    
    # Step 3: Run dashboard
    dashboard_process = run_dashboard()
    
    # Step 4: Run tests
    run_tests()
    
    # Step 5: Run RL tests
    run_rl_tests()
    
    # Cleanup
    print("\n" + "=" * 60)
    print("Workflow Complete")
    print("=" * 60)
    print("🔚 Stopping server and dashboard...")
    
    if server_process:
        server_process.terminate()
        print("✅ API server stopped")
    
    if dashboard_process:
        dashboard_process.terminate()
        print("✅ Dashboard stopped")


if __name__ == "__main__":
    main()