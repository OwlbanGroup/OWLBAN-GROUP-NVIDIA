#!/usr/bin/env python3
"""
Test script to verify environment configuration for JPMorgan Financial APIs
"""
import os
import sys

# Add the current directory to sys.path to import config
sys.path.insert(0, os.path.dirname(__file__))

from config import Config

def test_environment_config():
    """Test the environment configuration logic"""

    print("Testing JPMorgan Environment Configuration")
    print("=" * 50)

    # Test environments
    environments = ['dev', 'staging', 'prod']

    for env in environments:
        print(f"\nTesting environment: {env}")
        print("-" * 30)

        # Temporarily set the environment variable
        original_env = os.environ.get('JPMORGAN_ENVIRONMENT')
        os.environ['JPMORGAN_ENVIRONMENT'] = env

        # Reload the config class to pick up the new environment
        # Since it's a class with class variables, we need to manually update
        Config.JPMORGAN_ENVIRONMENT = env

        # Test merchant service
        merchant_url = Config.get_jpmorgan_endpoint_url('merchant', use_mtls=False)
        merchant_mtls_url = Config.get_jpmorgan_endpoint_url('merchant', use_mtls=True)

        # Test openbanking service
        openbanking_url = Config.get_jpmorgan_endpoint_url('openbanking')

        # Test apigateway service
        apigateway_url = Config.get_jpmorgan_endpoint_url('apigateway')

        print(f"Merchant URL: {merchant_url}")
        print(f"Merchant mTLS URL: {merchant_mtls_url}")
        print(f"OpenBanking URL: {openbanking_url}")
        print(f"API Gateway URL: {apigateway_url}")

        # Verify expectations
        if env in ['dev', 'staging']:
            expected_merchant = Config.JPMORGAN_MERCHANT_UAT_URL
            expected_merchant_mtls = Config.JPMORGAN_MERCHANT_MTLS_UAT_URL
            expected_openbanking = Config.JPMORGAN_OPENBANKING_UAT_URL
            expected_apigateway = Config.JPMORGAN_APIGATEWAY_PRODUCTION_URL  # qaf only for apigateway
        else:  # prod
            expected_merchant = Config.JPMORGAN_MERCHANT_PRODUCTION_URL
            expected_merchant_mtls = Config.JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL
            expected_openbanking = Config.JPMORGAN_OPENBANKING_PRODUCTION_URL
            expected_apigateway = Config.JPMORGAN_APIGATEWAY_PRODUCTION_URL

        # Check results
        checks = [
            ('Merchant URL', merchant_url, expected_merchant),
            ('Merchant mTLS URL', merchant_mtls_url, expected_merchant_mtls),
            ('OpenBanking URL', openbanking_url, expected_openbanking),
            ('API Gateway URL', apigateway_url, expected_apigateway)
        ]

        all_passed = True
        for check_name, actual, expected in checks:
            if actual == expected:
                print(f"✅ {check_name}: PASS")
            else:
                print(f"❌ {check_name}: FAIL")
                print(f"   Expected: {expected}")
                print(f"   Actual: {actual}")
                all_passed = False

        if all_passed:
            print(f"🎉 Environment '{env}' configuration: ALL TESTS PASSED")
        else:
            print(f"💥 Environment '{env}' configuration: TESTS FAILED")

        # Restore original environment
        if original_env is not None:
            os.environ['JPMORGAN_ENVIRONMENT'] = original_env
        elif 'JPMORGAN_ENVIRONMENT' in os.environ:
            del os.environ['JPMORGAN_ENVIRONMENT']

    print("\n" + "=" * 50)
    print("Configuration testing complete!")

if __name__ == '__main__':
    test_environment_config()
