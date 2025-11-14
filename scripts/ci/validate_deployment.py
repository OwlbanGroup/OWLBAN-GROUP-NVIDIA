#!/usr/bin/env python3
"""
Deployment validation script for CI/CD pipeline
"""
import requests
import time
import sys
import os
from typing import Dict, List, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(level)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Validates deployment health and functionality"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout

    def check_health_endpoint(self) -> bool:
        """Check if health endpoint is responding"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info("✅ Health check passed")
                    return True
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False

    def check_api_endpoints(self) -> bool:
        """Check critical API endpoints"""
        endpoints = [
            ('GET', '/'),
            ('GET', '/metrics'),
            ('GET', '/telemetry/metrics'),
            ('GET', '/data/formats'),
            ('GET', '/ws/status')
        ]

        failed_endpoints = []

        for method, endpoint in endpoints:
            try:
                if method == 'GET':
                    response = self.session.get(f"{self.base_url}{endpoint}")
                    if response.status_code not in [200, 401]:  # 401 is expected for protected endpoints
                        failed_endpoints.append(f"{method} {endpoint}: {response.status_code}")
                        logger.error(f"❌ {method} {endpoint}: {response.status_code}")
                    else:
                        logger.info(f"✅ {method} {endpoint}: {response.status_code}")
            except Exception as e:
                failed_endpoints.append(f"{method} {endpoint}: {str(e)}")
                logger.error(f"❌ {method} {endpoint}: {e}")

        if failed_endpoints:
            logger.error(f"Failed endpoints: {failed_endpoints}")
            return False

        logger.info("✅ All API endpoints responding correctly")
        return True

    def check_security_headers(self) -> bool:
        """Check if security headers are properly set"""
        try:
            response = self.session.get(f"{self.base_url}/health")

            required_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options',
                'Strict-Transport-Security',
                'Content-Security-Policy'
            ]

            missing_headers = []
            for header in required_headers:
                if header not in response.headers:
                    missing_headers.append(header)

            if missing_headers:
                logger.error(f"❌ Missing security headers: {missing_headers}")
                return False

            logger.info("✅ Security headers are properly configured")
            return True

        except Exception as e:
            logger.error(f"❌ Security headers check error: {e}")
            return False

    def check_ssl_certificate(self) -> bool:
        """Check if SSL certificate is valid (for HTTPS URLs)"""
        if not self.base_url.startswith('https://'):
            logger.info("ℹ️  Skipping SSL check for non-HTTPS URL")
            return True

        try:
            # Force HTTPS verification
            response = requests.get(self.base_url, verify=True, timeout=self.timeout)
            if response.status_code == 200:
                logger.info("✅ SSL certificate is valid")
                return True
            else:
                logger.error(f"❌ SSL certificate check failed: {response.status_code}")
                return False
        except requests.exceptions.SSLError as e:
            logger.error(f"❌ SSL certificate error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ SSL check error: {e}")
            return False

    def check_performance(self) -> bool:
        """Check response times for performance regression"""
        try:
            times = []
            for _ in range(5):
                start = time.time()
                response = self.session.get(f"{self.base_url}/health")
                end = time.time()

                if response.status_code == 200:
                    times.append(end - start)
                time.sleep(0.1)  # Small delay between requests

            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)

                # Check if average response time is acceptable (< 500ms)
                if avg_time > 0.5:
                    logger.error(f"Average response time too high: {avg_time:.3f}s")
                    return False

                # Check if max response time is acceptable (< 1s)
                if max_time > 1.0:
                    logger.error(f"Max response time too high: {max_time:.3f}s")
                    return False

                logger.info(f"Average response time: {avg_time:.3f}s, Max response time: {max_time:.3f}s")
            else:
                logger.error("❌ Could not measure response times")
                return False

        except Exception as e:
            logger.error(f"❌ Performance check error: {e}")
            return False

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        logger.info("🚀 Starting deployment validation...")

        checks = [
            ("Health Endpoint", self.check_health_endpoint),
            ("API Endpoints", self.check_api_endpoints),
            ("Security Headers", self.check_security_headers),
            ("SSL Certificate", self.check_ssl_certificate),
            ("Performance", self.check_performance)
        ]

        results = []
        for check_name, check_func in checks:
            logger.info(f"Running {check_name} check...")
            result = check_func()
            results.append((check_name, result))

        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)

        logger.info(f"\n📊 Validation Summary: {passed}/{total} checks passed")

        if passed == total:
            logger.info("🎉 All validation checks passed!")
            return True
        else:
            logger.error("❌ Some validation checks failed!")
            for check_name, result in results:
                status = "✅" if result else "❌"
                logger.info(f"  {status} {check_name}")
            return False


def main():
    """Main validation function"""
    # Get environment variables
    environment = os.getenv('DEPLOY_ENV', 'staging')
    base_url = os.getenv('API_BASE_URL')

    if not base_url:
        # Default URLs based on environment
        if environment == 'production':
            base_url = 'https://api.jpmorgan.com'
        else:
            base_url = 'http://staging-api.jpmorgan.com'

    logger.info(f"Validating deployment for environment: {environment}")
    logger.info(f"API Base URL: {base_url}")

    validator = DeploymentValidator(base_url)

    if validator.run_all_checks():
        logger.info("✅ Deployment validation successful!")
        sys.exit(0)
    else:
        logger.error("❌ Deployment validation failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
