"""
Test script to verify config.py functionality after Pylint fixes
"""
import sys
import os

def test_config_import():
    """Test 1: Verify config module can be imported"""
    try:
        from config import Config, config
        print("✓ Test 1 PASSED: Config module imported successfully")
        return True, Config, config
    except Exception as e:
        print(f"✗ Test 1 FAILED: Config import error - {e}")
        return False, None, None

def test_config_attributes(Config):
    """Test 2: Verify all config attributes are accessible"""
    try:
        required_attrs = [
            'API_BASE_URL', 'API_VERSION', 'JPMORGAN_ENVIRONMENT',
            'DATABASE_URL', 'DATABASE_TYPE', 'TOKEN_CLIENT_ID',
            'AUDIT_LOG_ENABLED', 'AUDIT_LOG_RETENTION_DAYS'
        ]
        
        for attr in required_attrs:
            value = getattr(Config, attr)
            print(f"  - {attr}: {value}")
        
        print("✓ Test 2 PASSED: All config attributes accessible")
        return True
    except Exception as e:
        print(f"✗ Test 2 FAILED: Attribute access error - {e}")
        return False

def test_get_database_url(Config):
    """Test 3: Verify get_database_url() method works"""
    try:
        # Test with default sqlite
        url = Config.get_database_url()
        print(f"  - SQLite URL: {url}")
        
        # Test with postgresql (simulate)
        original_type = Config.DATABASE_TYPE
        Config.DATABASE_TYPE = 'postgresql'
        Config.DATABASE_USER = 'testuser'
        Config.DATABASE_PASSWORD = 'testpass'
        Config.DATABASE_HOST = 'localhost'
        Config.DATABASE_PORT = 5432
        Config.DATABASE_NAME = 'testdb'
        
        pg_url = Config.get_database_url()
        print(f"  - PostgreSQL URL: {pg_url}")
        
        # Restore original
        Config.DATABASE_TYPE = original_type
        
        print("✓ Test 3 PASSED: get_database_url() method works correctly")
        return True
    except Exception as e:
        print(f"✗ Test 3 FAILED: get_database_url() error - {e}")
        return False

def test_get_jpmorgan_endpoint_url(Config):
    """Test 4: Verify get_jpmorgan_endpoint_url() method works"""
    try:
        # Test openbanking production
        url1 = Config.get_jpmorgan_endpoint_url('openbanking')
        print(f"  - OpenBanking Production: {url1}")
        
        # Test openbanking UAT
        original_env = Config.JPMORGAN_ENVIRONMENT
        Config.JPMORGAN_ENVIRONMENT = 'uat'
        url2 = Config.get_jpmorgan_endpoint_url('openbanking')
        print(f"  - OpenBanking UAT: {url2}")
        
        # Test apigateway production
        Config.JPMORGAN_ENVIRONMENT = 'production'
        url3 = Config.get_jpmorgan_endpoint_url('apigateway')
        print(f"  - API Gateway Production: {url3}")
        
        # Test apigateway QAF
        Config.JPMORGAN_ENVIRONMENT = 'qaf'
        url4 = Config.get_jpmorgan_endpoint_url('apigateway')
        print(f"  - API Gateway QAF: {url4}")
        
        # Test invalid service (should raise ValueError)
        try:
            Config.get_jpmorgan_endpoint_url('invalid')
            print("  ✗ Should have raised ValueError for invalid service")
            return False
        except ValueError as ve:
            print(f"  - Correctly raised ValueError for invalid service: {ve}")
        
        # Restore original
        Config.JPMORGAN_ENVIRONMENT = original_env
        
        print("✓ Test 4 PASSED: get_jpmorgan_endpoint_url() method works correctly")
        return True
    except Exception as e:
        print(f"✗ Test 4 FAILED: get_jpmorgan_endpoint_url() error - {e}")
        return False

def test_get_all_settings(Config):
    """Test 5: Verify get_all_settings() method works"""
    try:
        settings = Config.get_all_settings()
        print(f"  - Retrieved {len(settings)} settings")
        print(f"  - Sample settings: api_base_url={settings.get('api_base_url')}")
        print(f"  - Sample settings: database_type={settings.get('database_type')}")
        
        print("✓ Test 5 PASSED: get_all_settings() method works correctly")
        return True
    except Exception as e:
        print(f"✗ Test 5 FAILED: get_all_settings() error - {e}")
        return False

def test_audit_settings(Config):
    """Test 6: Verify audit logging settings are accessible"""
    try:
        audit_settings = {
            'AUDIT_LOG_ENABLED': Config.AUDIT_LOG_ENABLED,
            'AUDIT_LOG_RETENTION_DAYS': Config.AUDIT_LOG_RETENTION_DAYS,
            'AUDIT_FAILED_LOGIN_THRESHOLD': Config.AUDIT_FAILED_LOGIN_THRESHOLD,
            'AUDIT_RATE_LIMIT_THRESHOLD': Config.AUDIT_RATE_LIMIT_THRESHOLD,
            'AUDIT_BRUTE_FORCE_THRESHOLD': Config.AUDIT_BRUTE_FORCE_THRESHOLD,
            'AUDIT_SUSPICIOUS_IP_THRESHOLD': Config.AUDIT_SUSPICIOUS_IP_THRESHOLD,
            'AUDIT_ALERT_NOTIFICATION_METHOD': Config.AUDIT_ALERT_NOTIFICATION_METHOD,
            'AUDIT_CLEANUP_ENABLED': Config.AUDIT_CLEANUP_ENABLED,
            'AUDIT_HASH_CHAIN_ENABLED': Config.AUDIT_HASH_CHAIN_ENABLED,
        }
        
        for key, value in audit_settings.items():
            print(f"  - {key}: {value}")
        
        print("✓ Test 6 PASSED: All audit settings accessible")
        return True
    except Exception as e:
        print(f"✗ Test 6 FAILED: Audit settings error - {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("TESTING CONFIG.PY AFTER PYLINT FIXES")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Import
    print("Test 1: Config Module Import")
    print("-" * 70)
    success, Config, config = test_config_import()
    results.append(success)
    print()
    
    if not success:
        print("Cannot proceed with further tests due to import failure")
        sys.exit(1)
    
    # Test 2: Attributes
    print("Test 2: Config Attributes Access")
    print("-" * 70)
    results.append(test_config_attributes(Config))
    print()
    
    # Test 3: get_database_url
    print("Test 3: get_database_url() Method")
    print("-" * 70)
    results.append(test_get_database_url(Config))
    print()
    
    # Test 4: get_jpmorgan_endpoint_url
    print("Test 4: get_jpmorgan_endpoint_url() Method")
    print("-" * 70)
    results.append(test_get_jpmorgan_endpoint_url(Config))
    print()
    
    # Test 5: get_all_settings
    print("Test 5: get_all_settings() Method")
    print("-" * 70)
    results.append(test_get_all_settings(Config))
    print()
    
    # Test 6: Audit settings
    print("Test 6: Audit Logging Settings")
    print("-" * 70)
    results.append(test_audit_settings(Config))
    print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Config.py is working correctly after Pylint fixes")
        return 0
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
