#!/usr/bin/env python3
"""
API Performance Benchmarking Script
Tests response times for key API endpoints to ensure <200ms p95 latency
"""
import sys
import os
import time
import requests
import statistics
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.logger import telemetry_logger
except ImportError:
    telemetry_logger = None

class APIPerformanceBenchmark:
    """API Performance Benchmarking Tool"""

    def __init__(self, base_url='http://localhost:5000', auth_token=None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()
        if auth_token:
            self.session.headers.update({'Authorization': f'Bearer {auth_token}'})

    def make_request(self, endpoint, method='GET', data=None, params=None):
        """Make a single API request and measure response time"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': response.status_code,
                'response_time_ms': response_time,
                'success': response.status_code < 400
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': None,
                'response_time_ms': response_time,
                'success': False,
                'error': str(e)
            }

    def benchmark_endpoint(self, endpoint, method='GET', data=None, params=None, iterations=10):
        """Benchmark a single endpoint with multiple iterations"""
        print(f"📊 Benchmarking {method} {endpoint} ({iterations} iterations)...")

        response_times = []
        successes = 0

        for i in range(iterations):
            result = self.make_request(endpoint, method, data, params)
            response_times.append(result['response_time_ms'])
            if result['success']:
                successes += 1

            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{iterations} requests")

        # Calculate statistics
        stats = {
            'endpoint': endpoint,
            'method': method,
            'iterations': iterations,
            'success_rate': successes / iterations,
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18],  # 95th percentile
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }

        # Check performance targets
        stats['meets_target'] = stats['p95_response_time'] < 200  # <200ms p95 target

        print(f"  ✅ Success rate: {stats['success_rate']:.1%}")
        print(f"  ⏱️  P95 latency: {stats['p95_response_time']:.1f}ms {'✅' if stats['meets_target'] else '❌'}")
        print(f"  📈 Avg latency: {stats['avg_response_time']:.1f}ms")

        return stats

    def run_full_benchmark(self):
        """Run comprehensive benchmark on key endpoints"""
        print("🚀 Starting API Performance Benchmark...")
        print(f"⏰ Start time: {datetime.now(timezone.utc).isoformat()}")
        print(f"🎯 Target: <200ms P95 response time")

        # Key endpoints to benchmark
        endpoints = [
            {'endpoint': '/health', 'method': 'GET'},
            {'endpoint': '/metrics', 'method': 'GET'},
            {'endpoint': '/', 'method': 'GET'},
            {'endpoint': '/telemetry/metrics', 'method': 'GET', 'params': {'hours': 1}},
            {'endpoint': '/businesses', 'method': 'GET'},
            {'endpoint': '/assets', 'method': 'GET'},
            {'endpoint': '/api/jpmorgan-data', 'method': 'GET'},
            {'endpoint': '/private-bank/accounts', 'method': 'GET'},
            {'endpoint': '/revenue/metrics', 'method': 'GET', 'params': {'start_date': '2024-01-01T00:00:00Z', 'end_date': '2024-01-02T00:00:00Z'}},
            {'endpoint': '/payments/dashboard', 'method': 'GET'},
        ]

        results = []
        overall_success = True

        for endpoint_config in endpoints:
            try:
                result = self.benchmark_endpoint(**endpoint_config, iterations=10)
                results.append(result)

                if not result['meets_target']:
                    overall_success = False

                print()  # Add spacing between endpoints

            except Exception as e:
                print(f"❌ Error benchmarking {endpoint_config['endpoint']}: {e}")
                results.append({
                    'endpoint': endpoint_config['endpoint'],
                    'error': str(e),
                    'meets_target': False
                })
                overall_success = False

        # Generate summary
        summary = self.generate_summary(results)

        print("📋 Benchmark Summary:")
        print(f"  Total endpoints tested: {len(results)}")
        print(f"  Endpoints meeting target: {summary['meeting_target']}")
        print(f"  Overall success: {'✅' if overall_success else '❌'}")
        print(f"  Average P95 latency: {summary['avg_p95']:.1f}ms")
        print(f"⏰ End time: {datetime.now(timezone.utc).isoformat()}")

        # Log results
        if telemetry_logger:
            logger = telemetry_logger.get_logger()
            logger.info("API performance benchmark completed", extra={
                'benchmark_results': results,
                'summary': summary,
                'overall_success': overall_success
            })

        return {
            'results': results,
            'summary': summary,
            'overall_success': overall_success
        }

    def generate_summary(self, results):
        """Generate summary statistics from benchmark results"""
        successful_results = [r for r in results if 'p95_response_time' in r]

        if not successful_results:
            return {'meeting_target': 0, 'avg_p95': 0, 'total_tested': len(results)}

        meeting_target = sum(1 for r in successful_results if r['meets_target'])
        avg_p95 = statistics.mean(r['p95_response_time'] for r in successful_results)

        return {
            'meeting_target': meeting_target,
            'total_tested': len(results),
            'avg_p95': avg_p95
        }

def main():
    """Main benchmarking function"""
    # You can set auth_token if needed for authenticated endpoints
    auth_token = os.environ.get('BENCHMARK_AUTH_TOKEN')

    benchmark = APIPerformanceBenchmark(
        base_url='http://localhost:5000',
        auth_token=auth_token
    )

    results = benchmark.run_full_benchmark()

    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)

if __name__ == '__main__':
    main()
