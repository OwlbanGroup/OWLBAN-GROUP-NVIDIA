#!/usr/bin/env python3
"""
Performance Benchmarking Tests for JPMorgan Financial APIs
Tests CPU usage, memory consumption, response times, and throughput
"""
import time
import psutil
import cProfile
import pstats
import io
from functools import wraps
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
from ..app import app
from test_utils import SAMPLE_TELEMETRY_DATA, LARGE_BATCH_DATA

class PerformanceBenchmarker:
    """Performance benchmarking utilities"""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.process = psutil.Process()

    def measure_memory_usage(self, func):
        """Decorator to measure memory usage of a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            execution_time = end_time - start_time

            print(f"Execution Time: {execution_time:.2f}s")
            print(f"Memory Delta: {memory_delta:.2f} MB")
            print(f"Initial Memory: {initial_memory:.2f} MB, Final Memory: {final_memory:.2f} MB")

            return result, {
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'initial_memory': initial_memory,
                'final_memory': final_memory
            }
        return wrapper

    def profile_function(self, func, *args, **kwargs):
        """Profile a function using cProfile"""
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        profile_output = s.getvalue()

        return result, profile_output

    def benchmark_response_time(self, endpoint, method='GET', data=None, headers=None, iterations=100):
        """Benchmark response time for an endpoint"""
        response_times = []

        for _ in range(iterations):
            start_time = time.time()

            if method == 'GET':
                response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
            elif method == 'POST':
                response = requests.post(f"{self.base_url}{endpoint}",
                                       json=data, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            end_time = time.time()
            response_times.append(end_time - start_time)

            if response.status_code != 200:
                print(f"Warning: Request failed with status {response.status_code}")

        avg_time = statistics.mean(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        p95 = statistics.quantiles(response_times, n=20)[18]  # 95th percentile

        print(f"\nResponse Time Benchmark for {endpoint} ({method}):")
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Min: {min_time:.4f}s")
        print(f"  Max: {max_time:.4f}s")
        print(f"  Std Dev: {std_dev:.4f}s")
        print(f"  95th Percentile: {p95:.4f}s")

        return {
            'endpoint': endpoint,
            'method': method,
            'iterations': iterations,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_dev,
            'p95': p95
        }

    def benchmark_throughput(self, endpoint, method='GET', data=None, headers=None,
                           concurrent_users=10, duration=60):
        """Benchmark throughput under concurrent load"""
        results = []

        def make_request():
            start_time = time.time()
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
                elif method == 'POST':
                    response = requests.post(f"{self.base_url}{endpoint}",
                                           json=data, headers=headers)
                end_time = time.time()

                return {
                    'success': response.status_code == 200,
                    'response_time': end_time - start_time,
                    'status_code': response.status_code
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'success': False,
                    'response_time': end_time - start_time,
                    'error': str(e)
                }

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            while time.time() - start_time < duration:
                futures.append(executor.submit(make_request))

            for future in as_completed(futures):
                results.append(future.result())

        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        throughput = total_requests / duration
        avg_response_time = statistics.mean(r['response_time'] for r in results)
        success_rate = (successful_requests / total_requests) * 100

        print(f"\nThroughput Benchmark for {endpoint} ({method}):")
        print(f"  Duration: {duration}s")
        print(f"  Concurrent Users: {concurrent_users}")
        print(f"  Total Requests: {total_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Avg Response Time: {avg_response_time:.4f}s")
        print(f"  Success Rate: {success_rate:.2f}%")

        return {
            'endpoint': endpoint,
            'method': method,
            'duration': duration,
            'concurrent_users': concurrent_users,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'throughput': throughput,
            'avg_response_time': avg_response_time,
            'success_rate': success_rate
        }

def benchmark_telemetry_processing():
    """Benchmark telemetry processing performance"""
    print("🚀 Starting Telemetry Processing Performance Benchmarks")

    benchmarker = PerformanceBenchmarker()

    # Benchmark single telemetry processing
    print("\n📊 Benchmarking Single Telemetry Processing...")
    single_results = benchmarker.benchmark_response_time(
        '/telemetry', 'POST', SAMPLE_TELEMETRY_DATA, iterations=50
    )

    # Benchmark batch telemetry processing
    print("\n📊 Benchmarking Batch Telemetry Processing...")
    batch_results = benchmarker.benchmark_response_time(
        '/telemetry/batch', 'POST', LARGE_BATCH_DATA, iterations=20
    )

    # Benchmark throughput
    print("\n📊 Benchmarking Throughput (Concurrent Users)...")
    throughput_results = benchmarker.benchmark_throughput(
        '/telemetry', 'POST', SAMPLE_TELEMETRY_DATA,
        concurrent_users=5, duration=30
    )

    return {
        'single_telemetry': single_results,
        'batch_telemetry': batch_results,
        'throughput': throughput_results
    }

def benchmark_ml_operations():
    """Benchmark ML operations performance"""
    print("🤖 Starting ML Operations Performance Benchmarks")

    benchmarker = PerformanceBenchmarker()

    # Prepare training data
    training_data = [
        [10, 50, 20, 30, 15, 40, 5], [12, 52, 22, 32, 17, 42, 6],
        [8, 48, 18, 28, 13, 38, 4], [15, 55, 25, 35, 20, 45, 7],
        [9, 49, 19, 29, 14, 39, 5], [11, 51, 21, 31, 16, 41, 6],
        [13, 53, 23, 33, 18, 43, 7], [7, 47, 17, 27, 12, 37, 4],
        [14, 54, 24, 34, 19, 44, 6], [16, 56, 26, 36, 21, 46, 8]
    ]

    train_payload = {
        'training_data': training_data,
        'contamination': 0.1
    }

    anomaly_payload = {
        'telemetry_data': [SAMPLE_TELEMETRY_DATA]
    }

    # Benchmark ML training
    print("\n📊 Benchmarking ML Model Training...")
    train_results = benchmarker.benchmark_response_time(
        '/ml/train', 'POST', train_payload, iterations=5
    )

    # Benchmark anomaly detection
    print("\n📊 Benchmarking ML Anomaly Detection...")
    anomaly_results = benchmarker.benchmark_response_time(
        '/ml/anomalies', 'POST', anomaly_payload, iterations=20
    )

    return {
        'ml_training': train_results,
        'anomaly_detection': anomaly_results
    }

def benchmark_memory_usage():
    """Benchmark memory usage for various operations"""
    print("💾 Starting Memory Usage Benchmarks")

    benchmarker = PerformanceBenchmarker()

    @benchmarker.measure_memory_usage
    def process_large_batch():
        """Process a large batch of telemetry data"""
        # Simulate processing 1000 telemetry events
        large_batch = {
            'telemetry_data': [SAMPLE_TELEMETRY_DATA] * 1000
        }

        with app.test_client() as client:
            response = client.post('/telemetry/batch',
                                 data=json.dumps(large_batch),
                                 content_type='application/json')
            return response

    @benchmarker.measure_memory_usage
    def train_ml_model():
        """Train ML model with large dataset"""
        training_data = [[i % 100, (i + 10) % 100, (i + 20) % 100,
                         (i + 30) % 100, (i + 40) % 100, (i + 50) % 100, i % 10]
                        for i in range(1000)]

        payload = {
            'training_data': training_data,
            'contamination': 0.1
        }

        with app.test_client() as client:
            response = client.post('/ml/train', json=payload)
            return response

    print("\n📊 Benchmarking Large Batch Processing Memory Usage...")
    batch_result, batch_metrics = process_large_batch()

    print("\n📊 Benchmarking ML Training Memory Usage...")
    train_result, train_metrics = train_ml_model()

    return {
        'batch_processing': batch_metrics,
        'ml_training': train_metrics
    }

def run_comprehensive_performance_benchmarks():
    """Run all performance benchmarks"""
    print("🏃‍♂️ Starting Comprehensive Performance Benchmarking Suite")
    print("=" * 70)

    results = {}

    try:
        # Telemetry processing benchmarks
        results['telemetry'] = benchmark_telemetry_processing()

        # ML operations benchmarks
        results['ml'] = benchmark_ml_operations()

        # Memory usage benchmarks
        results['memory'] = benchmark_memory_usage()

        print("\n" + "=" * 70)
        print("✅ Performance Benchmarking Complete!")
        print("📊 Results Summary:")
        print(f"  - Telemetry Processing: {results['telemetry']['single_telemetry']['avg_time']:.4f}s avg response time")
        print(f"  - ML Training: {results['ml']['ml_training']['avg_time']:.4f}s avg response time")
        print(f"  - Throughput: {results['telemetry']['throughput']['throughput']:.2f} req/s")
        print(f"  - Memory Usage (Batch): {results['memory']['batch_processing']['memory_delta']:.2f} MB delta")

        return True, results

    except Exception as e:
        print(f"\n❌ Performance Benchmarking Failed: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    success, results = run_comprehensive_performance_benchmarks()
    if success:
        # Save results to file
        with open('performance_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n💾 Results saved to performance_benchmark_results.json")
    exit(0 if success else 1)
