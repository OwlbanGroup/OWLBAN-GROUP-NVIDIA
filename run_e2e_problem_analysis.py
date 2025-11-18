#!/usr/bin/env python3
"""
E2E Problem Analysis Test Runner
Validates all issues identified in E2E_PROBLEM_ANALYSIS.md
"""
import os
import sys
import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class E2EProblemAnalyzer:
    def __init__(self, base_url='http://localhost:8000'):
        self.base_url = base_url
        self.results = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }
        
    def print_header(self, text):
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}{text}{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
    def print_test(self, name, status, details=''):
        self.results['total'] += 1
        if status == 'PASS':
            self.results['passed'] += 1
            print(f"{GREEN}✓ PASS{RESET}: {name}")
        else:
            self.results['failed'] += 1
            print(f"{RED}✗ FAIL{RESET}: {name}")
        if details:
            print(f"  {YELLOW}→{RESET} {details}")
            
    def add_issue(self, priority, issue):
        self.results[priority].append(issue)
        
    def test_dashboard_endpoint(self):
        """Test 1.1: Dashboard Endpoint Failure"""
        self.print_header("TEST 1.1: Dashboard Endpoint")
        try:
            response = requests.get(f"{self.base_url}/dashboard", timeout=5)
            if response.status_code == 200:
                self.print_test("Dashboard endpoint accessible", "PASS")
            elif response.status_code == 500:
                self.print_test("Dashboard endpoint returns 500", "FAIL", 
                              "Template file missing - CRITICAL ISSUE")
                self.add_issue('critical', {
                    'issue': 'Dashboard endpoint failure',
                    'status_code': 500,
                    'fix': 'Move dashboard.html to templates/index.html'
                })
            else:
                self.print_test(f"Dashboard returns {response.status_code}", "FAIL")
        except Exception as e:
            self.print_test("Dashboard endpoint test", "FAIL", str(e))
            self.add_issue('critical', {
                'issue': 'Dashboard endpoint unreachable',
                'error': str(e)
            })
            
    def test_authentication_bypass(self):
        """Test 1.2: Authentication Bypass Vulnerability"""
        self.print_header("TEST 1.2: Authentication Bypass")
        
        # Check if TESTING mode can be enabled
        testing_mode = os.environ.get('TESTING', '0')
        if testing_mode == '1':
            self.print_test("Testing mode enabled", "FAIL", 
                          "Authentication bypass possible - CRITICAL SECURITY ISSUE")
            self.add_issue('critical', {
                'issue': 'Authentication bypass in testing mode',
                'severity': 'CRITICAL',
                'fix': 'Add environment validation to prevent testing mode in production'
            })
        else:
            self.print_test("Testing mode disabled", "PASS")
            
        # Test protected endpoint without auth
        try:
            response = requests.post(f"{self.base_url}/telemetry", 
                                   json={'test': 'data'}, timeout=5)
            if response.status_code == 401:
                self.print_test("Protected endpoint requires auth", "PASS")
            else:
                self.print_test("Protected endpoint accessible without auth", "FAIL",
                              f"Status: {response.status_code}")
                self.add_issue('critical', {
                    'issue': 'Authentication not enforced',
                    'endpoint': '/telemetry',
                    'status_code': response.status_code
                })
        except Exception as e:
            self.print_test("Authentication test", "FAIL", str(e))
            
    def test_template_files(self):
        """Test 1.3: Missing Template Files"""
        self.print_header("TEST 1.3: Template Files")
        
        templates_dir = Path('../templates')
        required_templates = ['index.html']
        
        if not templates_dir.exists():
            self.print_test("Templates directory exists", "FAIL", 
                          "Directory not found")
            self.add_issue('critical', {
                'issue': 'Templates directory missing',
                'fix': 'Create templates/ directory'
            })
        else:
            self.print_test("Templates directory exists", "PASS")
            
        for template in required_templates:
            template_path = templates_dir / template
            if template_path.exists():
                self.print_test(f"Template {template} exists", "PASS")
            else:
                self.print_test(f"Template {template} exists", "FAIL",
                              "File not found")
                self.add_issue('critical', {
                    'issue': f'Missing template: {template}',
                    'fix': f'Move dashboard.html to templates/{template}'
                })
                
    def test_hardcoded_credentials(self):
        """Test 1.4: Hardcoded Test Users"""
        self.print_header("TEST 1.4: Hardcoded Credentials")
        
        # Check app_final.py for hardcoded users
        app_file = Path('../app_final.py')
        if app_file.exists():
            content = app_file.read_text()
            
            if "users['testuser']" in content:
                self.print_test("No hardcoded test users", "FAIL",
                              "Found hardcoded 'testuser' in production code")
                self.add_issue('critical', {
                    'issue': 'Hardcoded test credentials',
                    'users': ['testuser', 'davidleeper'],
                    'fix': 'Remove hardcoded users, use test fixtures'
                })
            else:
                self.print_test("No hardcoded test users", "PASS")
                
            if "users['davidleeper']" in content:
                self.print_test("No hardcoded production users", "FAIL",
                              "Found hardcoded 'davidleeper' in production code")
        else:
            self.print_test("Check app_final.py", "FAIL", "File not found")
            
    def test_error_response_consistency(self):
        """Test 1.5: Error Response Consistency"""
        self.print_header("TEST 1.5: Error Response Consistency")
        
        # Test 404 error format
        try:
            response = requests.get(f"{self.base_url}/nonexistent", timeout=5)
            if response.status_code == 404:
                data = response.json()
                if 'status' in data and data['status'] == 'error':
                    self.print_test("404 error format consistent", "PASS")
                else:
                    self.print_test("404 error format consistent", "FAIL",
                                  "Missing 'status' field")
                    self.add_issue('high', {
                        'issue': 'Inconsistent error response format',
                        'endpoint': '/nonexistent',
                        'fix': 'Standardize all error responses'
                    })
        except Exception as e:
            self.print_test("Error response test", "FAIL", str(e))
            
    def test_database_session_management(self):
        """Test 2.1: Database Session Management"""
        self.print_header("TEST 2.1: Database Session Management")
        
        # Check for proper session cleanup in code
        app_file = Path('../app_final.py')
        if app_file.exists():
            content = app_file.read_text()
            
            # Look for session cleanup patterns
            has_context_manager = 'with db_manager' in content
            has_finally_cleanup = 'finally:' in content and 'session.close()' in content
            
            if has_context_manager or has_finally_cleanup:
                self.print_test("Database session cleanup implemented", "PASS")
            else:
                self.print_test("Database session cleanup implemented", "FAIL",
                              "No context managers or finally blocks found")
                self.add_issue('high', {
                    'issue': 'Missing database session cleanup',
                    'risk': 'Memory leaks and connection pool exhaustion',
                    'fix': 'Implement context managers for all DB operations'
                })
                
    def test_ssl_configuration(self):
        """Test 2.3: SSL/TLS Configuration"""
        self.print_header("TEST 2.3: SSL/TLS Configuration")
        
        # Check if HTTPS is enforced
        try:
            # Try HTTPS connection
            response = requests.get(f"https://localhost:8000/health", 
                                  timeout=5, verify=False)
            self.print_test("HTTPS enabled", "PASS")
        except requests.exceptions.SSLError:
            self.print_test("HTTPS enabled", "FAIL",
                          "SSL certificate issues")
            self.add_issue('medium', {
                'issue': 'SSL/TLS not properly configured',
                'fix': 'Complete SSL certificate setup'
            })
        except requests.exceptions.ConnectionError:
            self.print_test("HTTPS enabled", "FAIL",
                          "HTTPS not available")
            self.add_issue('medium', {
                'issue': 'HTTPS not enabled',
                'fix': 'Enable HTTPS in nginx and application'
            })
        except Exception as e:
            self.print_test("HTTPS test", "FAIL", str(e))
            
    def test_deployment_configuration(self):
        """Test 2.4: Deployment Configuration"""
        self.print_header("TEST 2.4: Deployment Configuration")
        
        # Check for multiple docker-compose files
        docker_files = [
            'docker-compose.yml',
            'docker-compose.prod.yml',
            'docker-compose.production.yml'
        ]
        
        found_files = []
        for file in docker_files:
            if Path(f'../{file}').exists():
                found_files.append(file)
                
        if len(found_files) > 1:
            self.print_test("Single deployment configuration", "FAIL",
                          f"Found {len(found_files)} docker-compose files")
            self.add_issue('high', {
                'issue': 'Multiple conflicting deployment configurations',
                'files': found_files,
                'fix': 'Consolidate to single production configuration'
            })
        else:
            self.print_test("Single deployment configuration", "PASS")
            
    def test_environment_files(self):
        """Test 2.5: Environment File Management"""
        self.print_header("TEST 2.5: Environment Files")
        
        # Check for multiple .env files
        env_files = [
            '.env',
            '.env.example',
            '.env.jpmorgan',
            '.env.new',
            '.env.production',
            '.env.production.example'
        ]
        
        found_files = []
        for file in env_files:
            if Path(f'../{file}').exists():
                found_files.append(file)
                
        if len(found_files) > 2:  # Should only have .env and .env.example
            self.print_test("Environment files consolidated", "FAIL",
                          f"Found {len(found_files)} .env files")
            self.add_issue('medium', {
                'issue': 'Too many environment files',
                'files': found_files,
                'fix': 'Keep only .env and .env.example'
            })
        else:
            self.print_test("Environment files consolidated", "PASS")
            
    def test_mock_data_in_production(self):
        """Test 3.1: Mock Data in Production"""
        self.print_header("TEST 3.1: Mock Data in Production")
        
        # Check if endpoints return mock data
        app_file = Path('../app_final.py')
        if app_file.exists():
            content = app_file.read_text()
            
            mock_indicators = [
                'Mock private bank',
                'Mock financial metrics',
                'Mock sync response',
                'account_id": "PB-001"'
            ]
            
            found_mocks = []
            for indicator in mock_indicators:
                if indicator in content:
                    found_mocks.append(indicator)
                    
            if found_mocks:
                self.print_test("No mock data in production endpoints", "FAIL",
                              f"Found {len(found_mocks)} mock data instances")
                self.add_issue('medium', {
                    'issue': 'Mock data in production endpoints',
                    'instances': found_mocks,
                    'fix': 'Replace with real data sources or feature flags'
                })
            else:
                self.print_test("No mock data in production endpoints", "PASS")
                
    def test_input_validation(self):
        """Test 3.2: Input Validation"""
        self.print_header("TEST 3.2: Input Validation")
        
        # Test with invalid data
        try:
            # Test with invalid JSON
            response = requests.post(f"{self.base_url}/telemetry",
                                   data='invalid json',
                                   headers={'Content-Type': 'application/json'},
                                   timeout=5)
            if response.status_code == 400:
                self.print_test("Invalid JSON rejected", "PASS")
            else:
                self.print_test("Invalid JSON rejected", "FAIL",
                              f"Status: {response.status_code}")
                
            # Test with missing required fields
            response = requests.post(f"{self.base_url}/telemetry",
                                   json={},
                                   timeout=5)
            if response.status_code in [400, 401]:
                self.print_test("Missing fields rejected", "PASS")
            else:
                self.print_test("Missing fields rejected", "FAIL",
                              f"Status: {response.status_code}")
        except Exception as e:
            self.print_test("Input validation test", "FAIL", str(e))
            
    def test_logging_consistency(self):
        """Test 3.4: Logging Consistency"""
        self.print_header("TEST 3.4: Logging Consistency")
        
        # Check for print statements in production code
        app_file = Path('../app_final.py')
        if app_file.exists():
            content = app_file.read_text()
            
            # Count print statements (excluding comments)
            lines = content.split('\n')
            print_count = sum(1 for line in lines 
                            if 'print(' in line and not line.strip().startswith('#'))
            
            if print_count > 0:
                self.print_test("No print statements in production", "FAIL",
                              f"Found {print_count} print statements")
                self.add_issue('low', {
                    'issue': 'Print statements in production code',
                    'count': print_count,
                    'fix': 'Replace with telemetry_logger'
                })
            else:
                self.print_test("No print statements in production", "PASS")
                
    def generate_report(self):
        """Generate final report"""
        self.print_header("E2E PROBLEM ANALYSIS REPORT")
        
        print(f"\n{BLUE}Test Summary:{RESET}")
        print(f"  Total Tests: {self.results['total']}")
        print(f"  {GREEN}Passed: {self.results['passed']}{RESET}")
        print(f"  {RED}Failed: {self.results['failed']}{RESET}")
        
        success_rate = (self.results['passed'] / self.results['total'] * 100) if self.results['total'] > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")
        
        print(f"\n{BLUE}Issues by Priority:{RESET}")
        print(f"  {RED}Critical: {len(self.results['critical'])}{RESET}")
        print(f"  {YELLOW}High: {len(self.results['high'])}{RESET}")
        print(f"  {YELLOW}Medium: {len(self.results['medium'])}{RESET}")
        print(f"  {BLUE}Low: {len(self.results['low'])}{RESET}")
        
        if self.results['critical']:
            print(f"\n{RED}CRITICAL ISSUES:{RESET}")
            for i, issue in enumerate(self.results['critical'], 1):
                print(f"\n  {i}. {issue.get('issue', 'Unknown issue')}")
                if 'fix' in issue:
                    print(f"     Fix: {issue['fix']}")
                    
        if self.results['high']:
            print(f"\n{YELLOW}HIGH PRIORITY ISSUES:{RESET}")
            for i, issue in enumerate(self.results['high'], 1):
                print(f"\n  {i}. {issue.get('issue', 'Unknown issue')}")
                if 'fix' in issue:
                    print(f"     Fix: {issue['fix']}")
                    
        # Production readiness assessment
        print(f"\n{BLUE}Production Readiness Assessment:{RESET}")
        if len(self.results['critical']) > 0:
            print(f"  {RED}Status: NOT READY FOR PRODUCTION{RESET}")
            print(f"  {RED}Blockers: {len(self.results['critical'])} critical issues{RESET}")
        elif len(self.results['high']) > 0:
            print(f"  {YELLOW}Status: NEEDS ATTENTION{RESET}")
            print(f"  {YELLOW}Issues: {len(self.results['high'])} high priority issues{RESET}")
        else:
            print(f"  {GREEN}Status: READY FOR PRODUCTION{RESET}")
            
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': self.results['total'],
                'passed': self.results['passed'],
                'failed': self.results['failed'],
                'success_rate': success_rate
            },
            'issues': {
                'critical': self.results['critical'],
                'high': self.results['high'],
                'medium': self.results['medium'],
                'low': self.results['low']
            }
        }
        
        with open('../e2e_analysis_results.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n{GREEN}Report saved to: e2e_analysis_results.json{RESET}")
        
    def run_all_tests(self):
        """Run all E2E problem analysis tests"""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}E2E PROBLEM ANALYSIS TEST SUITE{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")
        print(f"Base URL: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Phase 1: Critical Issues
        self.test_dashboard_endpoint()
        self.test_authentication_bypass()
        self.test_template_files()
        self.test_hardcoded_credentials()
        self.test_error_response_consistency()
        
        # Phase 2: High Priority Issues
        self.test_database_session_management()
        self.test_ssl_configuration()
        self.test_deployment_configuration()
        self.test_environment_files()
        
        # Phase 3: Medium Priority Issues
        self.test_mock_data_in_production()
        self.test_input_validation()
        
        # Phase 4: Low Priority Issues
        self.test_logging_consistency()
        
        # Generate final report
        self.generate_report()
        
        return len(self.results['critical']) == 0

def main():
    """Main entry point"""
    base_url = os.environ.get('API_BASE_URL', 'http://localhost:8000')
    
    analyzer = E2EProblemAnalyzer(base_url)
    
    try:
        success = analyzer.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Test suite failed: {str(e)}{RESET}")
        sys.exit(1)

if __name__ == '__main__':
    main()
