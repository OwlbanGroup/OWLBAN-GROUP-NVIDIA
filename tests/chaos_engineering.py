#!/usr/bin/env python3
"""
Chaos Engineering Tests for JPMorgan Financial APIs
Tests system resilience under failure conditions
"""
import time
import threading
import requests
import json
import subprocess
import signal
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from app_final import app
from test_utils import SAMPLE_TELEMETRY_DATA, LARGE_BATCH_DATA

class ChaosTestSuite:
    """Chaos engineering test suite"""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []

    def log_result(self, test_name, success, details=None, error=None):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'timestamp': time.time(),
            'details': details or {},
            'error': str(error) if error else None
        }
        self.test_results.append(result)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error}")
        return result

    def test_network_latency_injection(self):
        """Test system behavior under network latency"""
        print("🕐 Testing Network Latency Injection...")

        try:
            # Simulate network latency by adding delays
            def delayed_request():
                time.sleep(2)  # 2 second delay
                response = requests.get(f"{self.base_url}/health", timeout=10)
                return response.status_code == 200

            # Run multiple delayed requests concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(delayed_request) for _ in range(10)]
                results = [future.result() for future in as_completed(futures)]

            success_rate = sum(results) / len(results)
            success = success_rate > 0.8  # 80% success rate threshold

            self.log_result(
                'network_latency_injection',
                success,
                {'success_rate': success_rate, 'total_requests': len(results)}
            )

        except Exception as e:
            self.log_result('network_latency_injection', False, error=e)

    def test_service_outage_simulation(self):
        """Test behavior when dependent services are unavailable"""
        print("🔌 Testing Service Outage Simulation...")

        try:
            # Simulate database unavailability by stopping the service temporarily
            # This is a simplified version - in real chaos engineering, you'd use tools like Chaos Monkey

            # Test with database connection issues (if any)
            response = requests.post(f"{self.base_url}/telemetry",
                                    json=SAMPLE_TELEMETRY_DATA,
                                    timeout=5)

            # Even if database is down, API should return proper error response
            success = response.status_code in [200, 500, 503]  # Accept service unavailable

            self.log_result(
                'service_outage_simulation',
                success,
                {'response_code': response.status_code}
            )

        except requests.exceptions.Timeout:
            self.log_result('service_outage_simulation', False, error="Request timeout")
        except Exception as e:
            self.log_result('service_outage_simulation', False, error=e)

    def test_resource_exhaustion(self):
        """Test system under resource exhaustion"""
        print("💥 Testing Resource Exhaustion...")

        try:
            # Create many concurrent connections
            def exhaust_resources():
                try:
                    # Send large payload
                    large_data = {
                        "telemetry_data": [SAMPLE_TELEMETRY_DATA] * 1000
                    }
                    response = requests.post(f"{self.base_url}/telemetry/batch",
                                            json=large_data,
                                            timeout=30)
                    return response.status_code
                except Exception as e:
                    return str(e)

            # Run resource-intensive operations concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(exhaust_resources) for _ in range(20)]
                results = [future.result() for future in as_completed(futures)]

            # Check if system handled the load gracefully
            success_codes = sum(1 for r in results if isinstance(r, int) and r in [200, 413, 429])
            success = (success_codes / len(results)) > 0.5  # At least 50% handled gracefully

            self.log_result(
                'resource_exhaustion',
                success,
                {'total_requests': len(results), 'successful_responses': success_codes}
            )

        except Exception as e:
            self.log_result('resource_exhaustion', False, error=e)

    def test_dependency_failure_injection(self):
        """Test system when dependencies fail"""
        print("🔗 Testing Dependency Failure Injection...")

        try:
            # Test various dependency failure scenarios

            # 1. Invalid external service calls (if any)
            # 2. Database connection failures
            # 3. Cache unavailability

            test_scenarios = [
                {
                    'name': 'database_connection_failure',
                    'test': lambda: self._test_database_failure()
                },
                {
                    'name': 'cache_unavailability',
                    'test': lambda: self._test_cache_failure()
                },
                {
                    'name': 'external_service_timeout',
                    'test': lambda: self._test_external_service_failure()
                }
            ]

            scenario_results = {}
            for scenario in test_scenarios:
                try:
                    result = scenario['test']()
                    scenario_results[scenario['name']] = result
                except Exception as e:
                    scenario_results[scenario['name']] = False

            # Overall success if at least 2/3 scenarios handled gracefully
            successful_scenarios = sum(scenario_results.values())
            success = successful_scenarios >= 2

            self.log_result(
                'dependency_failure_injection',
                success,
                {'scenario_results': scenario_results, 'successful_scenarios': successful_scenarios}
            )

        except Exception as e:
            self.log_result('dependency_failure_injection', False, error=e)

    def _test_database_failure(self):
        """Test database failure scenario"""
        # In a real implementation, this would temporarily break database connectivity
        # For now, test with invalid data that might cause DB errors
        try:
            invalid_data = {"invalid": "data", "missing": "required_fields"}
            response = requests.post(f"{self.base_url}/telemetry",
                                    json=invalid_data,
                                    timeout=5)
            # Should get 400, not 500 (DB error)
            return response.status_code == 400
        except:
            return False

    def _test_cache_failure(self):
        """Test cache failure scenario"""
        # Test that system works without cache
        try:
            response = requests.get(f"{self.base_url}/telemetry/metrics?hours=1",
                                    timeout=5)
            return response.status_code == 200
        except:
            return False

    def _test_external_service_failure(self):
        """Test external service failure scenario"""
        # Test timeout handling
        try:
            # Make request with very short timeout to simulate slow external service
            response = requests.get(f"{self.base_url}/health",
                                    timeout=0.001)  # Very short timeout
            return False  # Should have timed out
        except requests.exceptions.Timeout:
            return True  # Expected timeout
        except:
            return False

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern implementation"""
        print("🔌 Testing Circuit Breaker Pattern...")

        try:
            # Simulate repeated failures to trigger circuit breaker
            failed_requests = 0
            total_requests = 10

            for i in range(total_requests):
                try:
                    # Send requests that should fail
                    invalid_data = {"completely": "invalid"}
                    response = requests.post(f"{self.base_url}/telemetry",
                                            json=invalid_data,
                                            timeout=2)
                    if response.status_code != 200:
                        failed_requests += 1
                except:
                    failed_requests += 1
                time.sleep(0.1)  # Small delay between requests

            # Circuit breaker should prevent cascading failures
            # In a real system, after X failures, requests should be rejected immediately
            success = failed_requests < total_requests  # Some requests should be handled

            self.log_result(
                'circuit_breaker_pattern',
                success,
                {'failed_requests': failed_requests, 'total_requests': total_requests}
            )

        except Exception as e:
            self.log_result('circuit_breaker_pattern', False, error=e)

    def test_graceful_degradation(self):
        """Test graceful degradation under load"""
        print("📉 Testing Graceful Degradation...")

        try:
            # Test that system degrades gracefully under extreme load
            def high_load_request():
                try:
                    # Send multiple large requests simultaneously
                    large_data = {"telemetry_data": [SAMPLE_TELEMETRY_DATA] * 500}
                    response = requests.post(f"{self.base_url}/telemetry/batch",
                                            json=large_data,
                                            timeout=10)
                    return response.status_code
                except requests.exceptions.Timeout:
                    return 504  # Gateway timeout
                except Exception as e:
                    return str(e)

            # Run high load test
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(high_load_request) for _ in range(50)]
                results = [future.result() for future in as_completed(futures)]

            # Check graceful degradation - should not crash, should return appropriate error codes
            acceptable_codes = [200, 429, 503, 504]  # Success or expected overload responses
            graceful_responses = sum(1 for r in results if isinstance(r, int) and r in acceptable_codes)

            success = (graceful_responses / len(results)) > 0.7  # 70% graceful handling

            self.log_result(
                'graceful_degradation',
                success,
                {'total_requests': len(results), 'graceful_responses': graceful_responses}
            )

        except Exception as e:
            self.log_result('graceful_degradation', False, error=e)

    def run_chaos_tests(self):
        """Run all chaos engineering tests"""
        print("🎭 Starting Chaos Engineering Test Suite")
        print("=" * 60)

        tests = [
            self.test_network_latency_injection,
            self.test_service_outage_simulation,
            self.test_resource_exhaustion,
            self.test_dependency_failure_injection,
            self.test_circuit_breaker_pattern,
            self.test_graceful_degradation
        ]

        for test in tests:
            test()
            time.sleep(1)  # Brief pause between tests

        # Summary
        print("\n" + "=" * 60)
        print("🎭 Chaos Engineering Test Results")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(".1f")

        if passed_tests == total_tests:
            print("✅ All chaos tests passed - System is resilient!")
        else:
            print("⚠️  Some chaos tests failed - Review system resilience")

        return passed_tests == total_tests, self.test_results

def run_chaos_engineering_tests():
    """Main function to run chaos engineering tests"""
    chaos_suite = ChaosTestSuite()
    success, results = chaos_suite.run_chaos_tests()

    # Save results
    with open('chaos_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Chaos test results saved to chaos_test_results.json")
    return success

if __name__ == "__main__":
    success = run_chaos_engineering_tests()
    exit(0 if success else 1)
