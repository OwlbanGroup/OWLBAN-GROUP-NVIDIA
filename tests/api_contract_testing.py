#!/usr/bin/env python3
"""
API Contract Testing for JPMorgan Financial APIs
Validates API responses against OpenAPI specifications and schemas
"""
import json
import requests
import jsonschema
from openapi_spec_validator import validate_spec
from app_final import app
from test_utils import SAMPLE_TELEMETRY_DATA, LARGE_BATCH_DATA

class APIContractTester:
    """API contract testing utilities"""

    def __init__(self, base_url="http://localhost:5000", spec_file="openapi.yml"):
        self.base_url = base_url
        self.spec_file = spec_file
        self.test_results = []

    def log_result(self, test_name, success, details=None, error=None):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details or {},
            'error': str(error) if error else None
        }
        self.test_results.append(result)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error}")
        return result

    def validate_openapi_spec(self):
        """Validate OpenAPI specification"""
        print("📋 Validating OpenAPI Specification...")

        try:
            with open(self.spec_file, 'r') as f:
                spec = json.load(f)

            # Validate the spec
            validate_spec(spec)
            self.log_result('openapi_spec_validation', True, {'spec_version': spec.get('openapi')})
            return True

        except FileNotFoundError:
            self.log_result('openapi_spec_validation', False, error=f"Spec file not found: {self.spec_file}")
            return False
        except Exception as e:
            self.log_result('openapi_spec_validation', False, error=f"Spec validation failed: {e}")
            return False

    def test_endpoint_response_schemas(self):
        """Test that API responses match expected schemas"""
        print("🔍 Testing Endpoint Response Schemas...")

        endpoints_to_test = [
            {
                'path': '/health',
                'method': 'GET',
                'expected_schema': {
                    'type': 'object',
                    'required': ['status', 'timestamp'],
                    'properties': {
                        'status': {'type': 'string', 'enum': ['healthy', 'unhealthy']},
                        'timestamp': {'type': 'string'},
                        'version': {'type': 'string'}
                    }
                }
            },
            {
                'path': '/telemetry',
                'method': 'POST',
                'data': SAMPLE_TELEMETRY_DATA,
                'expected_schema': {
                    'type': 'object',
                    'required': ['status', 'message', 'timestamp'],
                    'properties': {
                        'status': {'type': 'string', 'enum': ['success']},
                        'message': {'type': 'string'},
                        'timestamp': {'type': 'string'}
                    }
                }
            },
            {
                'path': '/telemetry/batch',
                'method': 'POST',
                'data': LARGE_BATCH_DATA,
                'expected_schema': {
                    'type': 'object',
                    'required': ['status', 'message', 'statistics', 'timestamp'],
                    'properties': {
                        'status': {'type': 'string', 'enum': ['success']},
                        'message': {'type': 'string'},
                        'statistics': {
                            'type': 'object',
                            'properties': {
                                'total_events': {'type': 'integer'},
                                'successful_events': {'type': 'integer'},
                                'failed_events': {'type': 'integer'}
                            }
                        },
                        'timestamp': {'type': 'string'}
                    }
                }
            }
        ]

        success_count = 0
        for endpoint in endpoints_to_test:
            try:
                if endpoint['method'] == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint['path']}")
                elif endpoint['method'] == 'POST':
                    response = requests.post(f"{self.base_url}{endpoint['path']}",
                                            json=endpoint.get('data'))

                if response.status_code != 200:
                    self.log_result(f"schema_test_{endpoint['path']}", False,
                                    error=f"HTTP {response.status_code}")
                    continue

                response_data = response.json()

                # Validate against schema
                jsonschema.validate(instance=response_data, schema=endpoint['expected_schema'])

                self.log_result(f"schema_test_{endpoint['path']}", True,
                                {'response_size': len(json.dumps(response_data))})
                success_count += 1

            except jsonschema.ValidationError as e:
                self.log_result(f"schema_test_{endpoint['path']}", False,
                                error=f"Schema validation failed: {e.message}")
            except Exception as e:
                self.log_result(f"schema_test_{endpoint['path']}", False, error=str(e))

        overall_success = success_count == len(endpoints_to_test)
        self.log_result('endpoint_response_schemas', overall_success,
                        {'tested_endpoints': len(endpoints_to_test), 'successful': success_count})

        return overall_success

    def test_error_response_formats(self):
        """Test that error responses follow consistent format"""
        print("🚨 Testing Error Response Formats...")

        error_scenarios = [
            {
                'description': 'Invalid JSON',
                'method': 'POST',
                'path': '/telemetry',
                'data': 'invalid json',
                'content_type': 'application/json',
                'expected_status': 400
            },
            {
                'description': 'Missing required fields',
                'method': 'POST',
                'path': '/telemetry',
                'data': {},
                'expected_status': 400
            },
            {
                'description': 'Non-existent endpoint',
                'method': 'GET',
                'path': '/nonexistent',
                'expected_status': 404
            }
        ]

        error_schema = {
            'type': 'object',
            'required': ['status', 'error'],
            'properties': {
                'status': {'type': 'string', 'enum': ['error']},
                'error': {'type': 'string'},
                'timestamp': {'type': 'string'}
            }
        }

        success_count = 0
        for scenario in error_scenarios:
            try:
                headers = {'Content-Type': scenario.get('content_type', 'application/json')}

                if scenario['method'] == 'GET':
                    response = requests.get(f"{self.base_url}{scenario['path']}", headers=headers)
                elif scenario['method'] == 'POST':
                    response = requests.post(f"{self.base_url}{scenario['path']}",
                                            data=scenario.get('data'),
                                            headers=headers)

                if response.status_code != scenario['expected_status']:
                    self.log_result(f"error_format_{scenario['description'].replace(' ', '_')}", False,
                                    error=f"Expected {scenario['expected_status']}, got {response.status_code}")
                    continue

                response_data = response.json()

                # Validate error response schema
                jsonschema.validate(instance=response_data, schema=error_schema)

                self.log_result(f"error_format_{scenario['description'].replace(' ', '_')}", True)
                success_count += 1

            except Exception as e:
                self.log_result(f"error_format_{scenario['description'].replace(' ', '_')}", False, error=str(e))

        overall_success = success_count == len(error_scenarios)
        self.log_result('error_response_formats', overall_success,
                        {'tested_scenarios': len(error_scenarios), 'successful': success_count})

        return overall_success

    def test_api_versioning(self):
        """Test API versioning consistency"""
        print("🏷️  Testing API Versioning...")

        try:
            # Check that version is returned in health endpoint
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                self.log_result('api_versioning', False, error=f"Health check failed: {response.status_code}")
                return False

            data = response.json()
            if 'version' not in data:
                self.log_result('api_versioning', False, error="Version not found in health response")
                return False

            version = data['version']
            # Basic version format validation (should be semantic version)
            import re
            if not re.match(r'^\d+\.\d+\.\d+', version):
                self.log_result('api_versioning', False, error=f"Invalid version format: {version}")
                return False

            self.log_result('api_versioning', True, {'version': version})
            return True

        except Exception as e:
            self.log_result('api_versioning', False, error=str(e))
            return False

    def test_content_type_headers(self):
        """Test Content-Type headers are correct"""
        print("📄 Testing Content-Type Headers...")

        endpoints = [
            ('/health', 'GET', 'application/json'),
            ('/telemetry', 'POST', 'application/json'),
            ('/telemetry/metrics', 'GET', 'application/json'),
            ('/telemetry/export', 'GET', 'application/json')
        ]

        success_count = 0
        for path, method, expected_content_type in endpoints:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{path}")
                elif method == 'POST':
                    response = requests.post(f"{self.base_url}{path}", json=SAMPLE_TELEMETRY_DATA)

                if response.status_code not in [200, 400, 404]:  # Accept error responses too
                    continue

                content_type = response.headers.get('Content-Type', '').split(';')[0]  # Remove charset
                if content_type == expected_content_type:
                    success_count += 1
                    self.log_result(f"content_type_{path.replace('/', '_')}", True,
                                    {'content_type': content_type})
                else:
                    self.log_result(f"content_type_{path.replace('/', '_')}", False,
                                    error=f"Expected {expected_content_type}, got {content_type}")

            except Exception as e:
                self.log_result(f"content_type_{path.replace('/', '_')}", False, error=str(e))

        overall_success = success_count == len(endpoints)
        self.log_result('content_type_headers', overall_success,
                        {'tested_endpoints': len(endpoints), 'correct_types': success_count})

        return overall_success

    def test_http_status_codes(self):
        """Test that appropriate HTTP status codes are used"""
        print("🔢 Testing HTTP Status Codes...")

        status_tests = [
            ('/health', 'GET', 200),
            ('/telemetry', 'POST', 200, SAMPLE_TELEMETRY_DATA),
            ('/nonexistent', 'GET', 404),
            ('/telemetry', 'POST', 400, {}),  # Invalid data
        ]

        success_count = 0
        for test_case in status_tests:
            path, method, expected_status = test_case[:3]
            data = test_case[3] if len(test_case) > 3 else None

            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{path}")
                elif method == 'POST':
                    response = requests.post(f"{self.base_url}{path}", json=data)

                if response.status_code == expected_status:
                    success_count += 1
                    self.log_result(f"status_code_{path.replace('/', '_')}_{method}", True,
                                    {'status_code': response.status_code})
                else:
                    self.log_result(f"status_code_{path.replace('/', '_')}_{method}", False,
                                    error=f"Expected {expected_status}, got {response.status_code}")

            except Exception as e:
                self.log_result(f"status_code_{path.replace('/', '_')}_{method}", False, error=str(e))

        overall_success = success_count == len(status_tests)
        self.log_result('http_status_codes', overall_success,
                        {'tested_cases': len(status_tests), 'correct_codes': success_count})

        return overall_success

    def run_contract_tests(self):
        """Run all API contract tests"""
        print("📑 Starting API Contract Testing Suite")
        print("=" * 60)

        tests = [
            self.validate_openapi_spec,
            self.test_endpoint_response_schemas,
            self.test_error_response_formats,
            self.test_api_versioning,
            self.test_content_type_headers,
            self.test_http_status_codes
        ]

        passed_tests = 0
        for test in tests:
            if test():
                passed_tests += 1

        # Summary
        print("\n" + "=" * 60)
        print("📑 API Contract Testing Results")
        print("=" * 60)

        total_tests = len(tests)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(".1f")

        if passed_tests == total_tests:
            print("✅ All contract tests passed - API contracts are valid!")
        else:
            print("⚠️  Some contract tests failed - Review API specifications")

        return passed_tests == total_tests, self.test_results

def run_api_contract_tests():
    """Main function to run API contract tests"""
    contract_tester = APIContractTester()
    success, results = contract_tester.run_contract_tests()

    # Save results
    with open('api_contract_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Contract test results saved to api_contract_test_results.json")
    return success

if __name__ == "__main__":
    success = run_api_contract_tests()
    exit(0 if success else 1)
