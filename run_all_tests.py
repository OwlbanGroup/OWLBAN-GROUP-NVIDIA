#!/usr/bin/env python3
"""
Seamless test script to run all tests for the JPMorgan Financial APIs.
This script starts the Flask server in a separate thread, runs all tests,
and then stops the server.
"""

import subprocess
import time
import sys
import os
from threading import Thread

# Add the current directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def start_server():
    """Start the Flask server in a separate thread"""
    print("Starting Flask server...")
    # Run the Flask app in the background
    subprocess.run([sys.executable, 'app.py'], cwd=os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Run the pytest tests"""
    print("Running tests...")
    result = subprocess.run([sys.executable, '-m', 'pytest', 'e2e_test.py', '-v'], cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode

if __name__ == '__main__':
    # Start the server in a separate thread
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait a bit for the server to start
    time.sleep(5)

    # Run the tests
    test_result = run_tests()

    # Exit with the test result code
    sys.exit(test_result)
