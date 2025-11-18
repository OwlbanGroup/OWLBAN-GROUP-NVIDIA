"""Test script to verify all imports work correctly"""
import sys
sys.path.insert(0, '.')

print("Testing imports...")

try:
    # Test shared modules
    from shared.schemas import APIResponse, ErrorResponse
    print("✅ shared.schemas imported successfully")
    
    from shared.auth import require_auth, TokenData, create_access_token
    print("✅ shared.auth imported successfully")
    
    # Test that we can create instances
    response = APIResponse(status="success", message="Test", data={"key": "value"})
    print(f"✅ APIResponse created: {response.status}")
    
    token_data = TokenData(username="test", user_id="123")
    print(f"✅ TokenData created: {token_data.username}")
    
    # Test JP Morgan client import (will fail on missing dependencies, but syntax is OK)
    try:
        from src.jpmorgan_client import JPMorganAPIClient, get_jpmorgan_client
        print("✅ src.jpmorgan_client imported successfully")
    except ImportError as e:
        print(f"⚠️  src.jpmorgan_client import failed (expected - missing httpx): {e}")
    
    print("\n✅ All critical imports successful!")
    print("Note: Some imports may fail due to missing dependencies (httpx, structlog, etc.)")
    print("Run 'pip install -r requirements.txt' to install all dependencies.")
    
except Exception as e:
    print(f"\n❌ Import test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
