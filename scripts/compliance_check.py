#!/usr/bin/env python3
"""
Compliance Check Script for JPMorgan Financial APIs

This script performs automated checks for GDPR and SOC2 compliance
including data handling, encryption, access controls, and audit logging.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import psycopg2
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s'
)
logger = logging.getLogger(__name__)


class ComplianceChecker:
    """Class for performing compliance checks on JPMorgan Financial APIs."""

    def __init__(self, base_url="http://localhost:8000", db_url=None):
        self.base_url = base_url.rstrip('/')
        default_db = 'postgresql://user:pass@localhost:5432/jpmorgan_financial_apis'
        self.db_url = db_url or os.getenv('DATABASE_URL', default_db)
        self.results = []

    def log_result(self, check_name, status, message, details=None):
        """Log a compliance check result"""
        result = {
            'check': check_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.results.append(result)
        logger.info("%s: %s - %s", check_name, status.upper(), message)

    def check_data_encryption(self):
        """Check data encryption at rest and in transit"""
        try:
            # Check database encryption
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            cursor.execute("SHOW ssl;")
            ssl_status = cursor.fetchone()[0]

            if ssl_status == 'on':
                self.log_result(
                    'data_encryption_ssl',
                    'pass',
                    'SSL/TLS encryption enabled for database connections'
                )
            else:
                self.log_result(
                    'data_encryption_ssl',
                    'fail',
                    'SSL/TLS encryption not enabled for database '
                    'connections'
                )

            conn.close()

            # Check HTTPS for API
            response = requests.get(
                f"{self.base_url}/health",
                verify=True,
                timeout=10
            )
            if response.url.startswith('https://'):
                self.log_result(
                    'data_encryption_https',
                    'pass',
                    'HTTPS encryption enabled for API endpoints'
                )
            else:
                self.log_result(
                    'data_encryption_https',
                    'fail',
                    'HTTPS encryption not enabled for API endpoints'
                )

        except (psycopg2.Error, requests.RequestException) as e:
            self.log_result(
                'data_encryption',
                'fail',
                f'Data encryption check failed: {str(e)}'
            )

    def check_access_controls(self):
        """Check access control mechanisms"""
        try:
            # Check for authentication requirements
            endpoints = ['/api/v1/accounts', '/api/v1/telemetry']
            for endpoint in endpoints:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    timeout=10
                )
                check_name = (
                    f'access_control_{endpoint.replace("/", "_")}'
                )
                if response.status_code in [401, 403]:
                    self.log_result(
                        check_name,
                        'pass',
                        f'Access control enforced for {endpoint}'
                    )
                else:
                    self.log_result(
                        check_name,
                        'warn',
                        f'Access control may not be enforced for '
                        f'{endpoint}'
                    )

            # Check for role-based access
            # This would require valid tokens - simplified check
            self.log_result(
                'access_control_roles',
                'info',
                'Role-based access control requires manual verification'
            )

        except (requests.RequestException, OSError) as e:
            self.log_result(
                'access_controls',
                'fail',
                f'Access control check failed: {str(e)}'
            )

    def check_audit_logging(self):
        """Check audit logging implementation"""
        try:
            # Check if audit logs exist
            log_files = ['logs/telemetry.log', 'logs/production.log']
            for log_file in log_files:
                log_exists = (
                    os.path.exists(log_file) and
                    os.path.getsize(log_file) > 0
                )
                check_name = (
                    f'audit_logging_{log_file.replace("/", "_")}'
                )
                if log_exists:
                    self.log_result(
                        check_name,
                        'pass',
                        f'Audit log exists: {log_file}'
                    )
                else:
                    self.log_result(
                        check_name,
                        'fail',
                        f'Audit log missing or empty: {log_file}'
                    )

            # Check log aggregation setup
            if os.path.exists('scripts/log_aggregation.py'):
                self.log_result(
                    'audit_logging_aggregation',
                    'pass',
                    'Log aggregation script exists'
                )
            else:
                self.log_result(
                    'audit_logging_aggregation',
                    'fail',
                    'Log aggregation script missing'
                )

        except OSError as e:
            self.log_result(
                'audit_logging',
                'fail',
                f'Audit logging check failed: {str(e)}'
            )

    def check_data_retention(self):
        """Check data retention policies"""
        try:
            # Check backup retention
            backup_dir = 'backups'
            if os.path.exists(backup_dir):
                backup_files = os.listdir(backup_dir)
                if len(backup_files) > 0:
                    self.log_result(
                        'data_retention_backups',
                        'pass',
                        f'Backup files found: {len(backup_files)} files'
                    )
                else:
                    self.log_result(
                        'data_retention_backups',
                        'warn',
                        'No backup files found'
                    )
            else:
                self.log_result(
                    'data_retention_backups',
                    'fail',
                    'Backup directory does not exist'
                )

            # Check data cleanup procedures (simplified)
            self.log_result(
                'data_retention_cleanup',
                'info',
                'Data retention cleanup requires manual verification'
            )

        except OSError as e:
            self.log_result(
                'data_retention',
                'fail',
                f'Data retention check failed: {str(e)}'
            )

    def check_privacy_compliance(self):
        """Check GDPR privacy compliance"""
        try:
            # Check for privacy policy
            if os.path.exists('docs/privacy-policy.md'):
                self.log_result(
                    'privacy_compliance_policy',
                    'pass',
                    'Privacy policy document exists'
                )
            else:
                self.log_result(
                    'privacy_compliance_policy',
                    'fail',
                    'Privacy policy document missing'
                )

            # Check data processing consent (simplified)
            self.log_result(
                'privacy_compliance_consent',
                'info',
                'Data processing consent requires manual verification'
            )

            # Check data minimization
            # This would require analyzing data collection
            self.log_result(
                'privacy_compliance_minimization',
                'info',
                'Data minimization requires manual verification'
            )

        except OSError as e:
            self.log_result(
                'privacy_compliance',
                'fail',
                f'Privacy compliance check failed: {str(e)}'
            )

    def check_security_headers(self):
        """Check security headers in HTTP responses"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            headers = response.headers

            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000',
            }

            for header, expected in security_headers.items():
                check_name = (
                    f'security_headers_'
                    f'{header.lower().replace("-", "_")}'
                )
                if header in headers:
                    if headers[header] == expected:
                        self.log_result(
                            check_name,
                            'pass',
                            f'Security header {header} properly set'
                        )
                    else:
                        self.log_result(
                            check_name,
                            'warn',
                            f'Security header {header} has unexpected '
                            f'value'
                        )
                else:
                    self.log_result(
                        check_name,
                        'fail',
                        f'Security header {header} missing'
                    )

        except (requests.RequestException, OSError) as e:
            self.log_result(
                'security_headers',
                'fail',
                f'Security headers check failed: {str(e)}'
            )

    def run_all_checks(self):
        """Run all compliance checks"""
        logger.info("Starting compliance checks...")

        checks = [
            self.check_data_encryption,
            self.check_access_controls,
            self.check_audit_logging,
            self.check_data_retention,
            self.check_privacy_compliance,
            self.check_security_headers
        ]

        for check in checks:
            check()

        return self.results

    def generate_report(self):
        """Generate compliance report"""
        total_checks = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'pass'])
        failed = len([r for r in self.results if r['status'] == 'fail'])
        warnings = len([r for r in self.results if r['status'] == 'warn'])
        info = len([r for r in self.results if r['status'] == 'info'])

        compliance_score = (
            (passed / total_checks * 100) if total_checks > 0 else 0
        )
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'info': info,
                'compliance_score': compliance_score
            },
            'results': self.results
        }

        return report


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Compliance Check Script'
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
        '--output',
        choices=['json', 'text'],
        default='text',
        help='Output format'
    )

    args = parser.parse_args()

    checker = ComplianceChecker(args.url, args.db_url)
    checker.run_all_checks()

    if args.output == 'json':
        report = checker.generate_report()
        print(json.dumps(report, indent=2))
    else:
        report = checker.generate_report()
        print("Compliance Check Report")
        print("=" * 30)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Info: {report['summary']['info']}")
        compliance_score = report['summary']['compliance_score']
        print(f"Compliance Score: {compliance_score:.1f}%")
        print("\nDetailed Results:")
        for result in report['results']:
            status_icons = {
                'pass': '✓',
                'fail': '✗',
                'warn': '⚠',
                'info': 'ℹ'
            }
            status_icon = status_icons[result['status']]
            print(f"{status_icon} {result['check']}: {result['message']}")

        # Exit with appropriate code
        if report['summary']['failed'] > 0:
            sys.exit(1)


if __name__ == '__main__':
    main()
