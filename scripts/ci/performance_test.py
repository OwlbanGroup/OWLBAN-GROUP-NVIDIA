#!/usr/bin/env python3
"""
Performance regression testing script for CI/CD pipeline
"""
import requests
import time
import statistics
import sys
import os
from typing import Dict, List, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(level)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Tests API performance and detects regressions"""

    def __init__(self, base_url: str, concurrent_users: int = 10, test_duration: int = 60):
        self.base_url = base_url.rstrip('/')
        self.concurrent_users = concurrent_users
        self.test_duration = test_duration
        self.session = requests.Session()

    def run_load_test(self) -> Dict[str, float]:
        """Run a simple load test on the health endpoint"""
        logger.info(f"Running load test with {self.concurrent_users} concurrent users for {self.test_duration}s")

        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'error_rate': 0.0,
            'avg_response_time': 0.0,
            'median_response_time': 0.0,
            'p95_response_time': 0.0,
            'p99_response_time': 0.0,
            'requests_per_second': 0.0
        }

        start_time = time.time()
        end_time = start_time + self.test_duration

        while time.time() < end_time:
            try:
                request_start = time.time()
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                request_end = time.time()

                results['total_requests'] += 1

                if response.status_code == 200:
                    results['successful_requests'] += 1
                    response_time = request_end - request_start
                    results['response_times'].append(response_time)
                else:
                    results['failed_requests'] += 1

            except Exception as e:
                results['failed_requests'] += 1
                logger.debug(f"Request failed: {e}")

            # Small delay to avoid overwhelming the server
            time.sleep(0.01)

        # Calculate statistics
        if results['response_times']:
            results['avg_response_time'] = statistics.mean(results['response_times'])
            results['median_response_time'] = statistics.median(results['response_times'])
            results['p95_response_time'] = statistics.quantiles(results['response_times'], n=20)[18]  # 95th percentile
            results['p99_response_time'] = statistics.quantiles(results['response_times'], n=100)[98]  # 99th percentile

        if results['total_requests'] > 0:
            results['error_rate'] = (results['failed_requests'] / results['total_requests']) * 100

        actual_duration = time.time() - start_time
        results['requests_per_second'] = results['total_requests'] / actual_duration

        return results

    def check_performance_regression(self, results: Dict[str, float]) -> bool:
        """Check if current performance meets acceptable thresholds"""
        logger.info("Checking performance against thresholds...")

        # Define acceptable thresholds
        thresholds = {
            'max_avg_response_time': 0.5,  # 500ms
            'max_p95_response_time': 1.0,  # 1s
            'max_p99_response_time': 2.0,  # 2s
            'max_error_rate': 5.0,  # 5%
            'min_requests_per_second': 50  # 50 RPS for basic load
        }

        issues = []

        if results['avg_response_time'] > thresholds['max_avg_response_time']:
            issues.append(f"Average response time too high: {results['avg_response_time']:.3f}s (threshold: {thresholds['max_avg_response_time']}s)")

        if results['p95_response_time'] > thresholds['max_p95_response_time']:
            issues.append(f"P95 response time too high: {results['p95_response_time']:.3f}s (threshold: {thresholds['max_p95_response_time']}s)")

        if results['p99_response_time'] > thresholds['max_p99_response_time']:
            issues.append(f"P99 response time too high: {results['p99_response_time']:.3f}s (threshold: {thresholds['max_p99_response_time']}s)")

        if results['error_rate'] > thresholds['max_error_rate']:
            issues.append(f"Error rate too high: {results['error_rate']:.2f}% (threshold: {thresholds['max_error_rate']}%)")

        if results['requests_per_second'] < thresholds['min_requests_per_second']:
            issues.append(f"Throughput too low: {results['requests_per_second']:.1f} RPS (threshold: {thresholds['min_requests_per_second']} RPS)")

        if issues:
            logger.error("❌ Performance regression detected:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False

        logger.info("✅ Performance meets all thresholds")
        return True

    def print_results(self, results: Dict[str, float]):
        """Print performance test results"""
        logger.info("\n📊 Performance Test Results:")
        logger.info(f"  Total Requests: {results['total_requests']}")
        logger.info(f"  Successful Requests: {results['successful_requests']}")
        logger.info(f"  Failed Requests: {results['failed_requests']}")
        logger.info(f"  Error Rate: {results['error_rate']:.2f}%")
        logger.info(f"  Requests/Second: {results['requests_per_second']:.1f}")
        logger.info(f"  Average Response Time: {results['avg_response_time']:.3f}s")
        logger.info(f"  Median Response Time: {results['median_response_time']:.3f}s")
        logger.info(f"  P95 Response Time: {results['p95_response_time']:.3f}s")
        logger.info(f"  P99 Response Time: {results['p99_response_time']:.3f}s")


def main():
    """Main performance testing function"""
    # Get environment variables
    base_url = os.getenv('API_BASE_URL', 'http://localhost:5000')
    concurrent_users = int(os.getenv('CONCURRENT_USERS', '10'))
    test_duration = int(os.getenv('TEST_DURATION', '30'))

    logger.info(f"Running performance tests against: {base_url}")
    logger.info(f"Concurrent users: {concurrent_users}")
    logger.info(f"Test duration: {test_duration}s")

    tester = PerformanceTester(base_url, concurrent_users, test_duration)

    # Run the load test
    results = tester.run_load_test()

    # Print results
    tester.print_results(results)

    # Check for regressions
    if tester.check_performance_regression(results):
        logger.info("✅ Performance test passed!")
        sys.exit(0)
    else:
        logger.error("❌ Performance test failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
