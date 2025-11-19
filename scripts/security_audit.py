#!/usr/bin/env python3
"""
Security Audit Script
Automated security scanning and vulnerability detection
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any

class SecurityAuditor:
    """Automated security audit tool"""

    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': [],
            'vulnerabilities': [],
            'warnings': [],
            'passed': [],
            'score': 0
        }

    def print_header(self, text):
        """Print section header"""
        print(f"\n{'='*70}")
        print(f"{text}")
        print(f"{'='*70}\n")

    def print_result(self, check_name, status, details=''):
        """Print check result"""
        symbol = 'PASS' if status == 'pass' else 'FAIL' if status == 'fail' else 'WARN'
        color = '\033[92m' if status == 'pass' else '\033[91m' if status == 'fail' else '\033[93m'
        reset = '\033[0m'

        print(f"{color}[{symbol}]{reset} {check_name}")
        if details:
            print(f"      {details}")

        self.results['checks'].append({
            'name': check_name,
            'status': status,
            'details': details
        })

        if status == 'pass':
            self.results['passed'].append(check_name)
        elif status == 'fail':
            self.results['vulnerabilities'].append({
                'check': check_name,
                'details': details
            })
        else:
            self.results['warnings'].append({
                'check': check_name,
                'details': details
            })

    def check_dependencies(self):
        """Check for vulnerable dependencies"""
        self.print_header("Dependency Security Check")

        try:
            # Check if safety is installed
            result = subprocess.run(['safety', '--version'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                self.print_result("Safety tool", "warn",
                                "Safety not installed. Run: pip install safety")
                return

            # Run safety check
            result = subprocess.run(['safety', 'check', '--json'],
                                    capture_output=True, text=True)

            if result.returncode == 0:
                self.print_result("Dependency vulnerabilities", "pass",
                                "No known vulnerabilities found")
            else:
                try:
                    vulns = json.loads(result.stdout)
                    self.print_result("Dependency vulnerabilities", "fail",
                                    f"Found {len(vulns)} vulnerable packages")
                except:
                    self.print_result("Dependency vulnerabilities", "warn",
                                    "Could not parse safety output")
        except FileNotFoundError:
            self.print_result("Safety tool", "warn",
                            "Safety not installed. Run: pip install safety")

    def check_code_security(self):
        """Check code for security issues using bandit"""
        self.print_header("Code Security Analysis")

        try:
            # Check if bandit is installed
            result = subprocess.run(['bandit', '--version'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                self.print_result("Bandit tool", "warn",
                                "Bandit not installed. Run: pip install bandit")
                return

            # Run bandit
            result = subprocess.run(['bandit', '-r', 'src/', '-f', 'json'],
                                    capture_output=True, text=True)

            try:
                report = json.loads(result.stdout)
                high = len([i for i in report.get('results', [])
                            if i.get('issue_severity') == 'HIGH'])
                medium = len([i for i in report.get('results', [])
                            if i.get('issue_severity') == 'MEDIUM'])

                if high > 0:
                    self.print_result("Code security issues", "fail",
                                    f"Found {high} high severity issues")
                elif medium > 0:
                    self.print_result("Code security issues", "warn",
                                    f"Found {medium} medium severity issues")
                else:
                    self.print_result("Code security issues", "pass",
                                    "No high/medium severity issues found")
            except:
                self.print_result("Code security issues", "warn",
                                "Could not parse bandit output")
        except FileNotFoundError:
            self.print_result("Bandit tool", "warn",
                            "Bandit not installed. Run: pip install bandit")

    def check_environment_variables(self):
        """Check for exposed secrets in environment"""
        self.print_header("Environment Variable Security")

        sensitive_vars = [
            'SECRET_KEY', 'TOKEN_CLIENT_SECRET', 'DATABASE_PASSWORD',
            'API_KEY', 'PRIVATE_KEY', 'AWS_SECRET_ACCESS_KEY'
        ]

        exposed = []
        for var in sensitive_vars:
            value = os.environ.get(var, '')
            if value and (value == 'dev_secret' or value == 'dummy' or
                        len(value) < 16):
                exposed.append(var)

        if exposed:
            self.print_result("Environment secrets", "fail",
                            f"Weak/default values: {', '.join(exposed)}")
        else:
            self.print_result("Environment secrets", "pass",
                            "No weak secrets detected")

    def check_file_permissions(self):
        """Check file permissions for sensitive files"""
        self.print_header("File Permission Check")

        sensitive_files = [
            '.env', '.env.production', 'config.py',
            'secrets.json', 'private_key.pem'
        ]

        issues = []
        for filename in sensitive_files:
            if os.path.exists(filename):
                mode = oct(os.stat(filename).st_mode)[-3:]
                if mode != '600' and mode != '400':
                    issues.append(f"{filename} ({mode})")

        if issues:
            self.print_result("File permissions", "warn",
                            f"Insecure permissions: {', '.join(issues)}")
        else:
            self.print_result("File permissions", "pass",
                            "Sensitive files properly protected")

    def check_authentication(self):
        """Check authentication configuration"""
        self.print_header("Authentication Security")

        # Check if testing mode is disabled
        testing_mode = os.environ.get('TESTING', '0')
        flask_env = os.environ.get('FLASK_ENV', 'production')

        if testing_mode == '1' and flask_env == 'production':
            self.print_result("Testing mode", "fail",
                            "Testing mode enabled in production!")
        else:
            self.print_result("Testing mode", "pass",
                            "Testing mode properly configured")

        # Check for hardcoded credentials
        try:
            with open('app_final.py', 'r') as f:
                content = f.read()
                if "users['testuser']" in content or "users['davidleeper']" in content:
                    self.print_result("Hardcoded credentials", "fail",
                                    "Found hardcoded test users in production code")
                else:
                    self.print_result("Hardcoded credentials", "pass",
                                    "No hardcoded credentials found")
        except FileNotFoundError:
            self.print_result("Hardcoded credentials", "warn",
                            "Could not check app_final.py")

    def check_https_configuration(self):
        """Check HTTPS/TLS configuration"""
        self.print_header("HTTPS/TLS Configuration")

        # Check if SSL certificates exist
        cert_files = ['ssl/cert.pem', 'ssl/key.pem', 'nginx/ssl/cert.pem']
        has_certs = any(os.path.exists(f) for f in cert_files)

        if has_certs:
            self.print_result("SSL certificates", "pass",
                            "SSL certificates found")
        else:
            self.print_result("SSL certificates", "warn",
                            "No SSL certificates found")

        # Check nginx configuration
        if os.path.exists('nginx/nginx.conf'):
            with open('nginx/nginx.conf', 'r') as f:
                content = f.read()
                if 'ssl_certificate' in content:
                    self.print_result("HTTPS enabled", "pass",
                                    "HTTPS configured in nginx")
                else:
                    self.print_result("HTTPS enabled", "warn",
                                    "HTTPS not configured in nginx")
        else:
            self.print_result("HTTPS enabled", "warn",
                            "nginx.conf not found")

    def check_rate_limiting(self):
        """Check rate limiting configuration"""
        self.print_header("Rate Limiting Check")

        try:
            with open('app_final.py', 'r') as f:
                content = f.read()

                # Check if rate limiting is bypassed
                if 'if app.config.get(\'TESTING\'):\n        return f' in content:
                    self.print_result("Rate limiting bypass", "fail",
                                    "Rate limiting can be bypassed in testing mode")
                else:
                    self.print_result("Rate limiting bypass", "pass",
                                    "Rate limiting properly enforced")

                # Check if rate limiting is configured
                if 'limiter.limit' in content or '@limiter.limit' in content:
                    self.print_result("Rate limiting configured", "pass",
                                    "Rate limiting is configured")
                else:
                    self.print_result("Rate limiting configured", "warn",
                                    "Rate limiting may not be configured")
        except FileNotFoundError:
            self.print_result("Rate limiting", "warn",
                            "Could not check app_final.py")

    def check_input_validation(self):
        """Check input validation implementation"""
        self.print_header("Input Validation Check")

        # Check if validators exist
        validator_files = [
            'src/validators_comprehensive.py',
            'src/validators_quick.py',
            'src/validation.py'
        ]

        has_validators = any(os.path.exists(f) for f in validator_files)

        if has_validators:
            self.print_result("Input validators", "pass",
                            "Input validation modules found")
        else:
            self.print_result("Input validators", "warn",
                            "No input validation modules found")

    def check_logging(self):
        """Check logging configuration"""
        self.print_header("Logging Security")

        # Check if structured logging exists
        if os.path.exists('src/structured_logger.py'):
            self.print_result("Structured logging", "pass",
                            "Structured logging implemented")
        else:
            self.print_result("Structured logging", "warn",
                            "Structured logging not found")

        # Check for sensitive data in logs
        try:
            with open('app_final.py', 'r') as f:
                content = f.read()
                if 'password' in content.lower() and 'log' in content.lower():
                    self.print_result("Sensitive data logging", "warn",
                                    "Possible password logging detected")
                else:
                    self.print_result("Sensitive data logging", "pass",
                                    "No obvious sensitive data logging")
        except FileNotFoundError:
            pass

    def calculate_score(self):
        """Calculate overall security score"""
        total_checks = len(self.results['checks'])
        passed = len(self.results['passed'])
        failed = len(self.results['vulnerabilities'])

        if total_checks == 0:
            return 0

        # Score: 100 points - (10 points per failure) - (5 points per warning)
        score = 100 - (failed * 10) - (len(self.results['warnings']) * 5)
        score = max(0, min(100, score))  # Clamp between 0-100

        self.results['score'] = score
        return score

    def generate_report(self):
        """Generate security audit report"""
        self.print_header("Security Audit Report")

        score = self.calculate_score()

        print(f"Overall Security Score: {score}/100")
        print(f"Total Checks: {len(self.results['checks'])}")
        print(f"Passed: {len(self.results['passed'])}")
        print(f"Failed: {len(self.results['vulnerabilities'])}")
        print(f"Warnings: {len(self.results['warnings'])}")

        if self.results['vulnerabilities']:
            print(f"\nCritical Vulnerabilities:")
            for vuln in self.results['vulnerabilities']:
                print(f"  - {vuln['check']}: {vuln['details']}")

        if self.results['warnings']:
            print(f"\nWarnings:")
            for warn in self.results['warnings']:
                print(f"  - {warn['check']}: {warn['details']}")

        # Save report to file
        report_file = f"security_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed report saved to: {report_file}")

        # Determine if audit passed
        if score >= 80:
            print(f"\n\033[92mSecurity Audit: PASSED\033[0m")
            return True
        elif score >= 60:
            print(f"\n\033[93mSecurity Audit: NEEDS IMPROVEMENT\033[0m")
            return False
        else:
            print(f"\n\033[91mSecurity Audit: FAILED\033[0m")
            return False

    def run_audit(self):
        """Run complete security audit"""
        print("\n" + "="*70)
        print("JPMorgan Financial APIs - Security Audit")
        print("="*70)

        self.check_dependencies()
        self.check_code_security()
        self.check_environment_variables()
        self.check_file_permissions()
        self.check_authentication()
        self.check_https_configuration()
        self.check_rate_limiting()
        self.check_input_validation()
        self.check_logging()

        return self.generate_report()

def main():
    """Main entry point"""
    auditor = SecurityAuditor()

    try:
        success = auditor.run_audit()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nAudit failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
