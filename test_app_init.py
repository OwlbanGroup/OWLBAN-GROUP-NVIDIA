#!/usr/bin/env python3
"""
Simple test script to check Flask app initialization
"""
import os
import sys

# Set testing environment
os.environ['TESTING'] = '1'

try:
    from app_final import app
    print('✅ Flask app imported and initialized successfully')
    print(f'✅ App testing mode: {app.config.get("TESTING", False)}')
    print(f'✅ App debug mode: {app.config.get("DEBUG", False)}')
    print('✅ All imports successful - app is ready for testing')
    sys.exit(0)
except Exception as e:
    print(f'❌ App initialization failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
