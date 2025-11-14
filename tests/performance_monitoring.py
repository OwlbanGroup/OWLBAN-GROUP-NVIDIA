#!/usr/bin/env python3
"""
Performance Monitoring Tests for JPMorgan Financial APIs
Monitors system performance during test execution
"""
import time
import psutil
import threading
import json
from datetime import datetime
from collections import defaultdict
from app_final import app
from test_utils import SAMPLE_TELEMETRY_DATA, LARGE_BATCH_DATA

class PerformanceMonitor:
    """Performance monitoring utilities"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval=1.0):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self, interval):
        """Main monitoring loop"""
        while self.monitoring:
            timestamp = time.time()

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics['cpu_percent'].append((timestamp, cpu_percent))

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_percent'].append((timestamp, memory.percent))
            self.metrics['memory_used'].append((timestamp, memory.used / 1024 / 1024))  # MB

            # Disk I/O (if available)
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_read'].append((timestamp, disk_io.read_bytes / 1024 / 1024))  # MB
                    self.metrics['disk_write'].append((timestamp, disk_io.write_bytes / 1024 / 1024))  # MB
            except:
                pass

            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    self.metrics['net_sent'].append((timestamp, net_io.bytes_sent / 1024 / 1024))  # MB
                    self.metrics['net_recv'].append((timestamp, net_io.bytes_recv / 1024 / 1024))  # MB
            except:
                pass

            time.sleep(interval)

    def get_summary_stats(self):
        """Get summary statistics from collected metrics"""
        summary = {}

        for metric_name, data_points in self.metrics.items():
            if not data_points:
                continue

            values = [point[1] for point in data_points]
            summary[metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1] if values else None
            }

        return summary

    def save_metrics(self, filename):
        """Save metrics to JSON file"""
        data = {
            'metrics': dict(self.metrics),
            'summary': self.get_summary_stats(),
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

class TestPerformanceMonitor:
    """Monitor performance during test execution"""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.test_results = []

    def run_test_with_monitoring(self, test_name, test_func, *args, **kwargs):
        """Run a test function with performance monitoring"""
        print(f"📊 Starting monitored test: {test_name}")

        # Start monitoring
        self.monitor.start_monitoring(interval=0.5)

        start_time = time.time()
        try:
            # Run the test
            result = test_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.time()
        execution_time = end_time - start_time

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Record test result
        test_result = {
            'test_name': test_name,
            'success': success,
            'execution_time': execution_time,
            'error': error,
            'performance_summary': self.monitor.get_summary_stats()
        }

        self.test_results.append(test_result)

        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name} ({execution_time:.2f}s)")

        if error:
            print(f"   Error: {error}")

        return test_result

    def run_telemetry_load_test(self, num_requests=100):
        """Run telemetry load test with monitoring"""
        def load_test():
            with app.test_client() as client:
                for i in range(num_requests):
                    response = client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
                    if response.status_code != 200:
                        raise Exception(f"Request {i+1} failed with status {response.status_code}")

        return self.run_test_with_monitoring(
            f'telemetry_load_test_{num_requests}_requests',
            load_test
        )

    def run_batch_processing_test(self, batch_sizes=[100, 500, 1000]):
        """Run batch processing tests with different sizes"""
        results = []

        for batch_size in batch_sizes:
            def batch_test():
                batch_data = {
                    'telemetry_data': [SAMPLE_TELEMETRY_DATA] * batch_size
                }

                with app.test_client() as client:
                    response = client.post('/telemetry/batch', json=batch_data)
                    if response.status_code != 200:
                        raise Exception(f"Batch processing failed with status {response.status_code}")

            result = self.run_test_with_monitoring(
                f'batch_processing_test_{batch_size}_items',
                batch_test
            )
            results.append(result)

        return results

    def run_ml_operations_test(self):
        """Run ML operations test with monitoring"""
        def ml_test():
            # Prepare training data
            training_data = [
                [10, 50, 20, 30, 15, 40, 5], [12, 52, 22, 32, 17, 42, 6],
                [8, 48, 18, 28, 13, 38, 4], [15, 55, 25, 35, 20, 45, 7]
            ]

            train_payload = {
                'training_data': training_data,
                'contamination': 0.1
            }

            anomaly_payload = {
                'telemetry_data': [SAMPLE_TELEMETRY_DATA]
            }

            with app.test_client() as client:
                # Train model
                train_response = client.post('/ml/train', json=train_payload)
                if train_response.status_code != 200:
                    raise Exception(f"ML training failed: {train_response.status_code}")

                # Detect anomalies
                anomaly_response = client.post('/ml/anomalies', json=anomaly_payload)
                if anomaly_response.status_code != 200:
                    raise Exception(f"Anomaly detection failed: {anomaly_response.status_code}")

        return self.run_test_with_monitoring('ml_operations_test', ml_test)

    def run_concurrent_operations_test(self, num_threads=10, requests_per_thread=20):
        """Run concurrent operations test"""
        def concurrent_test():
            def worker_thread(thread_id):
                with app.test_client() as client:
                    for i in range(requests_per_thread):
                        # Mix of different operations
                        if i % 3 == 0:
                            response = client.get('/health')
                        elif i % 3 == 1:
                            response = client.post('/telemetry', json=SAMPLE_TELEMETRY_DATA)
                        else:
                            response = client.get('/telemetry/metrics?hours=1')

                        if response.status_code not in [200, 404]:  # 404 is ok for metrics if no data
                            raise Exception(f"Thread {thread_id}, request {i+1} failed: {response.status_code}")

            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        return self.run_test_with_monitoring(
            f'concurrent_operations_test_{num_threads}_threads_{requests_per_thread}_requests',
            concurrent_test
        )

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'test_results': self.test_results,
            'overall_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results if r['success']),
                'failed_tests': sum(1 for r in self.test_results if not r['success']),
                'total_execution_time': sum(r['execution_time'] for r in self.test_results)
            },
            'performance_insights': self._analyze_performance()
        }

        return report

    def _analyze_performance(self):
        """Analyze performance data for insights"""
        insights = {}

        if not self.test_results:
            return insights

        # CPU analysis
        cpu_peaks = []
        memory_peaks = []

        for result in self.test_results:
            perf = result.get('performance_summary', {})
            if 'cpu_percent' in perf:
                cpu_peaks.append(perf['cpu_percent']['max'])
            if 'memory_percent' in perf:
                memory_peaks.append(perf['memory_percent']['max'])

        if cpu_peaks:
            insights['avg_cpu_peak'] = sum(cpu_peaks) / len(cpu_peaks)
            insights['max_cpu_peak'] = max(cpu_peaks)

        if memory_peaks:
            insights['avg_memory_peak'] = sum(memory_peaks) / len(memory_peaks)
            insights['max_memory_peak'] = max(memory_peaks)

        # Performance bottlenecks
        slow_tests = [r for r in self.test_results if r['execution_time'] > 5.0]  # Tests taking > 5 seconds
        insights['slow_tests'] = len(slow_tests)
        insights['slow_test_names'] = [r['test_name'] for r in slow_tests]

        return insights

def run_performance_monitoring_tests():
    """Run comprehensive performance monitoring tests"""
    print("📈 Starting Performance Monitoring Test Suite")
    print("=" * 60)

    monitor = TestPerformanceMonitor()

    # Run various performance tests
    tests = [
        lambda: monitor.run_telemetry_load_test(50),
        lambda: monitor.run_batch_processing_test([50, 100, 200]),
        lambda: monitor.run_ml_operations_test(),
        lambda: monitor.run_concurrent_operations_test(5, 10)
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed: {e}")

    # Generate report
    report = monitor.generate_performance_report()

    print("\n" + "=" * 60)
    print("📈 Performance Monitoring Results")
    print("=" * 60)

    summary = report['overall_summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(".2f")

    insights = report.get('performance_insights', {})
    if insights:
        print("\n🔍 Performance Insights:")
        if 'avg_cpu_peak' in insights:
            print(".1f")
        if 'max_cpu_peak' in insights:
            print(".1f")
        if 'avg_memory_peak' in insights:
            print(".1f")
        if 'slow_tests' in insights:
            print(f"Slow Tests (>5s): {insights['slow_tests']}")

    # Save detailed report
    with open('performance_monitoring_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n💾 Detailed report saved to performance_monitoring_report.json")

    success = summary['failed_tests'] == 0
    return success

if __name__ == "__main__":
    success = run_performance_monitoring_tests()
    exit(0 if success else 1)
