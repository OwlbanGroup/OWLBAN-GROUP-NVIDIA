"""
Test script for additional endpoints - simplified version without pytest
"""
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Set testing environment
os.environ['TESTING'] = '1'

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import test config first to avoid config issues
from test_config import test_config

# Create a mock config module
import types
mock_config = types.ModuleType('config')
mock_config.config = test_config

def test_imports():
    """Test that we can import the app with mocked config"""
    print("Testing app import with mocked config...")

    try:
        # Patch the config module before importing app
        with patch.dict('sys.modules', {'config': mock_config}):
            from app import app
            print("✓ App imported successfully")
            return True
    except Exception as e:
        print(f"✗ Failed to import app: {e}")
        return False

def test_basic_functionality():
    """Test basic app functionality"""
    print("Testing basic app functionality...")

    try:
        with patch.dict('sys.modules', {'config': mock_config}):
            from app import app

            # Create test client
            app.config['TESTING'] = True
            client = app.test_client()

            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health endpoint works")
                return True
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run simplified tests"""
    print("🚀 Simplified Additional Endpoints Test")
    print("=" * 50)

    import_success = test_imports()
    if not import_success:
        print("❌ Cannot proceed with tests due to import failure")
        return

    basic_success = test_basic_functionality()

    if basic_success:
        print("\n✅ All basic tests passed!")
    else:
        print("\n❌ Some tests failed")

if __name__ == "__main__":
    main()
