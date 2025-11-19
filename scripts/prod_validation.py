#!/usr/bin/env python3
"""
Production Validation Script for JPMorgan Financial APIs

This script performs automated checks to validate production readiness
including health checks, configuration validation, and performance metrics.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import psycopg2  # type: ignore
import redis
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Production validation class for checking system readiness."""

    def __init__(self, base_url="http://localhost:5000", db_url=None,
                redis_url=None):
        self.base_url = base_url.rstrip('/')
        default_db = (
            'postgresql://jpmorgan_user:password@postgresql:5432/'
            'jpmorgan_financial_apis'
        )
        self.db_url = db_url or os.getenv('DATABASE_URL', default_db)
        self.redis_url = redis_url or os.getenv(
            'REDIS_URL',
            'redis://redis:6379/0'
        )
        self.results = []

    def log_result(self, check_name, status, message, details=None):
        """Log a validation result"""
        result = {
            'check': check_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.results.append(result)
        logger.info("%s: %s - %s", check_name, status.upper(), message)

    def check_application_health(self):
        """Check application health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_result(
                        'application_health',
                        'pass',
                        'Application is healthy'
                    )
                else:
                    self.log_result(
                        'application_health',
                        'fail',
                        f'Application status: {data.get("status")}'
                    )
            else:
                self.log_result(
                    'application_health',
                    'fail',
                    f'Health check failed with status {response.status_code}'
                )
        except (requests.RequestException, ValueError) as e:
            self.log_result(
                'application_health',
                'fail',
                f'Health check error: {str(e)}'
            )

    def check_database_connection(self):
        """Check database connectivity"""
        try:
            conn = psycopg2.connect(self.db_url)
            conn.close()
            self.log_result('database_connection', 'pass', 'Database connection successful')
        except psycopg2.Error as e:
            self.log_result(
                'database_connection',
                'fail',
                f'Database connection failed: {str(e)}'
            )

    def check_redis_connection(self):
        """Check Redis connectivity"""
        try:
            r = redis.from_url(self.redis_url)
            r.ping()
            self.log_result('redis_connection', 'pass', 'Redis connection successful')
        except redis.ConnectionError as e:
            self.log_result(
                'redis_connection',
                'fail',
                f'Redis connection failed: {str(e)}'
            )

    def check_api_endpoints(self):
        """Check critical API endpoints"""
        endpoints = [
            '/api/v1/accounts',
            '/api/v1/market/quotes',
            '/api/v1/telemetry'
        ]

        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                # 401 is expected for unauthenticated requests
                endpoint_name = (
                    f'api_endpoint_{endpoint.replace("/", "_")}'
                )
                if response.status_code in [200, 401]:
                    self.log_result(
                        endpoint_name,
                        'pass',
                        f'Endpoint {endpoint} accessible'
                    )
                else:
                    self.log_result(
                        endpoint_name,
                        'fail',
                        f'Endpoint {endpoint} returned '
                        f'{response.status_code}'
                    )
            except requests.RequestException as e:
                endpoint_name = (
                    f'api_endpoint_{endpoint.replace("/", "_")}'
                )
                self.log_result(
                    endpoint_name,
                    'fail',
                    f'Endpoint {endpoint} error: {str(e)}'
                )

    def check_configuration(self):
        """Check critical configuration"""
        required_env_vars = [
            'SECRET_KEY',
            'TOKEN_CLIENT_ID',
            'TOKEN_CLIENT_SECRET'
        ]

        for var in required_env_vars:
            config_name = f'config_{var.lower()}'
            if os.getenv(var):
                self.log_result(
                    config_name,
                    'pass',
                    f'Environment variable {var} is set'
                )
            else:
                self.log_result(
                    config_name,
                    'fail',
                    f'Environment variable {var} is not set'
                )

    def check_performance_metrics(self):
        """Check basic performance metrics"""
        try:
            # Simple response time check
            start_time = time.time()
            requests.get(f"{self.base_url}/health", timeout=10)
            response_time = time.time() - start_time

            if response_time < 1.0:  # Less than 1 second
                self.log_result(
                    'performance_response_time',
                    'pass',
                    f'Response time: {response_time:.2f}s'
                )
            else:
                self.log_result(
                    'performance_response_time',
                    'warn',
                    f'Response time: {response_time:.2f}s (slow)'
                )
        except requests.RequestException as e:
            self.log_result(
                'performance_response_time',
                'fail',
                f'Performance check failed: {str(e)}'
            )

    def run_all_checks(self):
        """Run all validation checks"""
        logger.info("Starting production validation checks...")

        checks = [
            self.check_application_health,
            self.check_database_connection,
            self.check_redis_connection,
            self.check_api_endpoints,
            self.check_configuration,
            self.check_performance_metrics
        ]

        for check in checks:
            check()

        return self.results

    def generate_report(self):
        """Generate validation report"""
        total_checks = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'pass'])
        failed = len([r for r in self.results if r['status'] == 'fail'])
        warnings = len([r for r in self.results if r['status'] == 'warn'])

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'success_rate': (passed / total_checks * 100) if total_checks > 0 else 0
            },
            'results': self.results
        }

        return report


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description='Production Validation Script'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='Base URL of the application'
    )
    parser.add_argument(
        '--db-url',
        help='Database connection URL'
    )
    parser.add_argument(
        '--redis-url',
        help='Redis connection URL'
    )
    parser.add_argument(
        '--output',
        choices=['json', 'text'],
        default='text',
        help='Output format'
    )

    args = parser.parse_args()

    validator = ProductionValidator(args.url, args.db_url, args.redis_url)
    validator.run_all_checks()

    if args.output == 'json':
        report = validator.generate_report()
        print(json.dumps(report, indent=2))
    else:
        report = validator.generate_report()
        print("Production Validation Report")
        print("=" * 50)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print("\nDetailed Results:")
        for result in report['results']:
            status_icons = {
                'pass': '✓',
                'fail': '✗',
                'warn': '⚠'
            }
            status_icon = status_icons[result['status']]
            print(f"{status_icon} {result['check']}: {result['message']}")

        # Exit with appropriate code
        if report['summary']['failed'] > 0:
            sys.exit(1)


if __name__ == '__main__':
    main()
