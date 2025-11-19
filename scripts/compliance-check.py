#!/usr/bin/env python3
"""
Compliance Check Script for JPMorgan Financial APIs
Validates GDPR, SOC 2, and other compliance requirements
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplianceChecker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[Dict] = []
        self.passed_checks: List[str] = []

    def log_issue(self, check: str, severity: str, message: str, file: Optional[str] = None):
        """Log a compliance issue"""
        issue = {
            'check': check,
            'severity': severity,
            'message': message,
            'file': str(file) if file else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.issues.append(issue)
        logger.error(f"[{severity}] {check}: {message}")

    def log_pass(self, check: str):
        """Log a passed check"""
        self.passed_checks.append(check)
        logger.info(f"[PASS] {check}")

    def check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance requirements"""
        logger.info("Checking GDPR compliance...")

        # Check for data processing agreements
        dpa_files = list(self.project_root.glob("**/dpa*.md")) + \
                    list(self.project_root.glob("**/data-processing*.md"))
        if not dpa_files:
            self.log_issue("GDPR", "HIGH", "No data processing agreement found")

        # Check for data retention policies
        retention_files = list(self.project_root.glob("**/retention*.md")) + \
                        list(self.project_root.glob("**/data-retention*.md"))
        if not retention_files:
            self.log_issue("GDPR", "HIGH", "No data retention policy found")

        # Check for privacy policy
        privacy_files = list(self.project_root.glob("**/privacy*.md")) + \
                        list(self.project_root.glob("**/gdpr*.md"))
        if not privacy_files:
            self.log_issue("GDPR", "HIGH", "No privacy policy found")

        # Check for data subject access request procedures
        dsar_files = list(self.project_root.glob("**/dsar*.md")) + \
                    list(self.project_root.glob("**/data-subject*.md"))
        if not dsar_files:
            self.log_issue("GDPR", "MEDIUM", "No data subject access request procedures found")

        # Check code for data handling
        self._check_data_handling_code()

        return len([i for i in self.issues if i['check'] == 'GDPR']) == 0

    def _check_data_handling_code(self):
        """Check code for proper data handling practices"""
        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for proper data encryption
                if 'password' in content.lower() or 'ssn' in content.lower():
                    if 'encrypt' not in content.lower() and 'hash' not in content.lower():
                        self.log_issue("GDPR", "HIGH",
                                    "Sensitive data handling without encryption",
                                    file_path)

                # Check for data logging without masking
                if 'log.' in content.lower() and ('email' in content.lower() or 'phone' in content.lower()):
                    if 'mask' not in content.lower():
                        self.log_issue("GDPR", "MEDIUM",
                                    "Personal data logged without masking",
                                    file_path)

            except Exception as e:
                logger.warning(f"Could not check file {file_path}: {e}")

    def check_soc2_compliance(self) -> bool:
        """Check SOC 2 compliance requirements"""
        logger.info("Checking SOC 2 compliance...")

        # Check for access control implementation
        auth_files = list(self.project_root.glob("**/auth*.py")) + \
                    list(self.project_root.glob("**/security*.py"))
        if not auth_files:
            self.log_issue("SOC2", "HIGH", "No authentication/authorization implementation found")

        # Check for audit logging
        audit_files = list(self.project_root.glob("**/audit*.py")) + \
                    list(self.project_root.glob("**/logging*.py"))
        if not audit_files:
            self.log_issue("SOC2", "HIGH", "No audit logging implementation found")

        # Check for change management
        change_files = list(self.project_root.glob("**/change*.md")) + \
                        list(self.project_root.glob("**/deployment*.md"))
        if not change_files:
            self.log_issue("SOC2", "MEDIUM", "No change management procedures found")

        # Check for incident response
        incident_files = list(self.project_root.glob("**/incident*.md")) + \
                        list(self.project_root.glob("**/response*.md"))
        if not incident_files:
            self.log_issue("SOC2", "HIGH", "No incident response procedures found")

        # Check for access reviews
        access_files = list(self.project_root.glob("**/access*.md")) + \
                        list(self.project_root.glob("**/rbac*.md"))
        if not access_files:
            self.log_issue("SOC2", "MEDIUM", "No access control documentation found")

        return len([i for i in self.issues if i['check'] == 'SOC2']) == 0

    def check_security_compliance(self) -> bool:
        """Check general security compliance"""
        logger.info("Checking security compliance...")

        # Check for secrets in code
        self._check_for_secrets()

        # Check for secure coding practices
        self._check_secure_coding()

        # Check for dependency vulnerabilities
        self._check_dependencies()

        # Check for security headers
        self._check_security_headers()

        return len([i for i in self.issues if i['check'] == 'SECURITY']) == 0

    def _check_for_secrets(self):
        """Check for hardcoded secrets in code"""
        python_files = list(self.project_root.glob("**/*.py"))
        config_files = list(self.project_root.glob("**/*.json")) + \
                        list(self.project_root.glob("**/*.yaml")) + \
                        list(self.project_root.glob("**/*.yml"))

        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']'
        ]

        for file_path in python_files + config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Filter out test files and config examples
                        if not any(skip in str(file_path) for skip in ['test', 'example', 'sample']):
                            self.log_issue("SECURITY", "CRITICAL",
                                        f"Potential hardcoded secret found: {pattern}",
                                        file_path)
            except Exception as e:
                logger.warning(f"Could not check file {file_path}: {e}")

    def _check_secure_coding(self):
        """Check for secure coding practices"""
        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for SQL injection vulnerabilities
                if 'execute(' in content and ('%' in content or '+' in content):
                    if 'text(' not in content and 'bindparams' not in content:
                        self.log_issue("SECURITY", "HIGH",
                                    "Potential SQL injection vulnerability",
                                    file_path)

                # Check for XSS vulnerabilities
                if 'render_template' in content and '<' in content:
                    if 'escape' not in content and 'Markup' not in content:
                        self.log_issue("SECURITY", "MEDIUM",
                                    "Potential XSS vulnerability in template rendering",
                                    file_path)

                # Check for insecure deserialization
                if 'pickle.loads' in content or 'yaml.load' in content:
                    self.log_issue("SECURITY", "HIGH",
                                "Insecure deserialization detected",
                                file_path)

            except Exception as e:
                logger.warning(f"Could not check file {file_path}: {e}")

    def _check_dependencies(self):
        """Check for dependency vulnerabilities"""
        try:
            # Check if safety is available
            result = subprocess.run([sys.executable, '-m', 'safety', 'check'],
                                    capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                vulnerabilities = result.stdout.strip()
                if vulnerabilities:
                    self.log_issue("SECURITY", "HIGH",
                                f"Dependency vulnerabilities found: {vulnerabilities}")
                else:
                    self.log_pass("Dependency security check")
            else:
                self.log_pass("Dependency security check")

        except FileNotFoundError:
            self.log_issue("SECURITY", "MEDIUM", "Safety tool not installed for dependency checking")
        except Exception as e:
            logger.warning(f"Could not check dependencies: {e}")

    def _check_security_headers(self):
        """Check for security headers implementation"""
        python_files = list(self.project_root.glob("**/*.py"))

        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Content-Security-Policy',
            'Strict-Transport-Security'
        ]

        headers_found = set()
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for header in security_headers:
                    if header.lower() in content.lower():
                        headers_found.add(header)

            except Exception as e:
                logger.warning(f"Could not check file {file_path}: {e}")

        missing_headers = set(security_headers) - headers_found
        if missing_headers:
            self.log_issue("SECURITY", "MEDIUM",
                        f"Missing security headers: {', '.join(missing_headers)}")

    def check_performance_compliance(self) -> bool:
        """Check performance compliance requirements"""
        logger.info("Checking performance compliance...")

        # Check for performance monitoring
        perf_files = list(self.project_root.glob("**/performance*.py")) + \
                    list(self.project_root.glob("**/monitoring*.py"))
        if not perf_files:
            self.log_issue("PERFORMANCE", "MEDIUM", "No performance monitoring implementation found")

        # Check for caching implementation
        cache_files = list(self.project_root.glob("**/cache*.py"))
        if not cache_files:
            self.log_issue("PERFORMANCE", "LOW", "No caching implementation found")

        # Check for database optimization
        db_files = list(self.project_root.glob("**/database*.py")) + \
                    list(self.project_root.glob("**/models*.py"))
        if db_files:
            self._check_database_optimization(db_files)

        return len([i for i in self.issues if i['check'] == 'PERFORMANCE']) == 0

    def _check_database_optimization(self, db_files: List[Path]):
        """Check database optimization practices"""
        for file_path in db_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for N+1 query patterns
                if '.all()' in content and 'select_related' not in content:
                    self.log_issue("PERFORMANCE", "MEDIUM",
                                "Potential N+1 query issue",
                                file_path)

                # Check for missing indexes (basic check)
                if 'where' in content.lower() and 'index' not in content.lower():
                    # This is a very basic check - in practice, you'd need more sophisticated analysis
                    pass

            except Exception as e:
                logger.warning(f"Could not check file {file_path}: {e}")

    def generate_report(self) -> Dict:
        """Generate compliance report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'project': str(self.project_root),
            'summary': {
                'total_checks': len(self.passed_checks) + len(self.issues),
                'passed': len(self.passed_checks),
                'failed': len(self.issues),
                'compliance_rate': 0.0
            },
            'issues': self.issues,
            'passed_checks': self.passed_checks
        }

        if report['summary']['total_checks'] > 0:
            report['summary']['compliance_rate'] = (
                report['summary']['passed'] / report['summary']['total_checks']
            ) * 100

        # Categorize issues by severity
        severity_counts = {}
        for issue in self.issues:
            severity = issue['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report['summary']['issues_by_severity'] = severity_counts

        return report

    def run_all_checks(self) -> bool:
        """Run all compliance checks"""
        logger.info("Starting comprehensive compliance check...")

        checks = [
            ('GDPR Compliance', self.check_gdpr_compliance),
            ('SOC 2 Compliance', self.check_soc2_compliance),
            ('Security Compliance', self.check_security_compliance),
            ('Performance Compliance', self.check_performance_compliance)
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                passed = check_func()
                if not passed:
                    all_passed = False
                logger.info(f"{check_name}: {'PASSED' if passed else 'FAILED'}")
            except Exception as e:
                logger.error(f"Error running {check_name}: {e}")
                all_passed = False

        return all_passed

def main():
    parser = argparse.ArgumentParser(description='Compliance Checker for JPMorgan Financial APIs')
    parser.add_argument('--project-root', default='.',
                        help='Project root directory (default: current directory)')
    parser.add_argument('--output', '-o', help='Output file for JSON report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    checker = ComplianceChecker(args.project_root)

    logger.info("JPMorgan Financial APIs - Compliance Check")
    logger.info("=" * 50)

    success = checker.run_all_checks()

    # Generate report
    report = checker.generate_report()

    logger.info("=" * 50)
    logger.info("COMPLIANCE CHECK RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total Checks: {report['summary']['total_checks']}")
    logger.info(f"Passed: {report['summary']['passed']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    logger.info(f"Compliance Rate: {report['summary']['compliance_rate']:.1f}%")

    if report['summary']['issues_by_severity']:
        logger.info("Issues by Severity:")
        for severity, count in report['summary']['issues_by_severity'].items():
            logger.info(f"  {severity}: {count}")

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
