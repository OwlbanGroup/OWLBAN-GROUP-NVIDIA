#!/usr/bin/env python3
"""
Simple test runner for Phase 8 endpoints
Starts a Flask server with the PFM blueprint and runs tests
"""
import sys
import os
import time
import threading

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Set testing mode
os.environ['TESTING'] = '1'

from flask import Flask
from flask_cors import CORS

# Import PFM blueprint
try:
    from blueprints.pfm import pfm_bp
    print("Successfully imported PFM blueprint")
except ImportError as e:
    print(f"Warning: Failed to import PFM blueprint: {e}")
    pfm_bp = None


# Create Flask app
app = Flask(__name__)
app.config['TESTING'] = True

# Enable CORS
CORS(app)

# Register PFM blueprint
app.register_blueprint(pfm_bp, url_prefix='/pfm')

# Register Banking blueprint for testing
try:
    from blueprints import banking_bp
    app.register_blueprint(banking_bp, url_prefix='/banking')
    print("Successfully imported and registered banking blueprint for testing")
except ImportError as e:
    print(f"Warning: Banking blueprint not available for testing: {e}")


print("Flask app created with PFM blueprint")
print("Available routes:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.methods} {rule.rule}")

def run_server():
    """Run the Flask server in a background thread"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("\nServer started on http://127.0.0.1:5000")
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    # Run the tests
    print("\n" + "="*50)
    print("Running Phase 8 tests...")
    print("="*50 + "\n")
    
    # Import and run tests
    try:
        # Change to test file directory to import it
        os.chdir(_project_root)
        
        # Import the test module
        import test_phase8_endpoints as test_module
        
        # Run the main function from the test module
        test_module.main()
        
        print("\n" + "="*50)
        print("Phase 8 testing completed!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Keep server running for a bit to let any async operations complete
    time.sleep(1)
    print("\nTest runner finished.")
