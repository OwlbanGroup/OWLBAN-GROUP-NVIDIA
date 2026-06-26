"""
Custom Metrics Collector for JPMorgan Financial APIs
Phase 8 - Post-Deployment Monitoring

This module provides custom metrics collection for comprehensive monitoring
of API performance, business metrics, and system health.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
from functools import wraps
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricsDataPoint:
    """Represents a single metrics data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_name: str = ""


@dataclass
class AggregatedMetrics:
    """Represents aggregated metrics."""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """
    Custom metrics collector for the JPMorgan API.
    
    Provides:
    - Request metrics (latency, throughput, errors)
    - Business metrics (transactions, revenue, users)
    - System metrics (CPU, memory, connections)
    - Custom application metrics
    """
    
    def __init__(self, flush_interval: int = 60):
        """
        Initialize the metrics collector.
        
        Args:
            flush_interval: Interval in seconds to flush metrics
        """
        self.flush_interval = flush_interval
        self._lock = threading.Lock()
        self._metrics: Dict[str, List[MetricsDataPoint]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._start_time = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Business metrics
        self._transactions_total = 0.0
        self._revenue_total = 0.0
        self._active_users = 0
        self._api_keys_used = 0
        
        logger.info("Metrics collector initialized")
    
    def start(self):
        """Start the metrics collector background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        logger.info("Metrics collector started")
    
    def stop(self):
        """Stop the metrics collector."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Metrics collector stopped")
    
    def _flush_loop(self):
        """Background loop to flush metrics."""
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_metrics()
    
    def _flush_metrics(self):
        """Flush metrics to storage."""
        with self._lock:
            # Export Prometheus format metrics
            metrics_output = self._export_prometheus()
            
            # Log metrics
            logger.debug(f"Flushed metrics: {len(metrics_output)} lines")
            
            # Clear old histograms
            for key in self._histograms:
                self._histograms[key].clear()
    
    def _export_prometheus(self) -> List[str]:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(time.time() * 1000)
        
        # Export counters
        for name, value in self._counters.items():
            lines.append(f"{name}_total {value} {timestamp}")
        
        # Export gauges
        for name, value in self._gauges.items():
            lines.append(f"{name} {value} {timestamp}")
        
        return lines
    
    # Counter methods
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by
            labels: Optional labels
        """
        with self._lock:
            key = self._build_key(name, labels)
            self._counters[key] += value
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        with self._lock:
            key = self._build_key(name, labels)
            return self._counters.get(key, 0.0)
    
    # Gauge methods
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric.
        
        Args:
            name: Gauge name
            value: Value to set
            labels: Optional labels
        """
        with self._lock:
            key = self._build_key(name, labels)
            self._gauges[key] = value
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        with self._lock:
            key = self._build_key(name, labels)
            return self._gauges.get(key, 0.0)
    
    # Histogram methods
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Add observation to histogram.
        
        Args:
            name: Histogram name
            value: Observed value
            labels: Optional labels
        """
        with self._lock:
            key = self._build_key(name, labels)
            self._histograms[key].append(value)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> AggregatedMetrics:
        """Get histogram statistics."""
        with self._lock:
            key = self._build_key(name, labels)
            values = sorted(self._histograms.get(key, []))
            
            if not values:
                return AggregatedMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            count = len(values)
            total = sum(values)
            
            return AggregatedMetrics(
                count=count,
                sum=total,
                min=values[0],
                max=values[-1],
                avg=total / count,
                p50=self._percentile(values, 0.50),
                p95=self._percentile(values, 0.95),
                p99=self._percentile(values, 0.99),
            )
    
    # Business metrics methods
    def track_transaction(self, amount: float, currency: str = "USD"):
        """Track a transaction."""
        with self._lock:
            self._transactions_total += 1
            self._revenue_total += amount
        
        self.increment_counter("transactions_total")
        self.increment_counter(f"transactions_{currency}_total", amount)
    
    def track_api_key_usage(self):
        """Track API key usage."""
        with self._lock:
            self._api_keys_used += 1
        self.increment_counter("api_keys_used_total")
    
    def set_active_users(self, count: int):
        """Set active users count."""
        with self._lock:
            self._active_users = count
        self.set_gauge("active_users", count)
    
    def get_business_metrics(self) -> Dict[str, Any]:
        """Get business metrics summary."""
        with self._lock:
            return {
                "transactions_total": self._transactions_total,
                "revenue_total": self._revenue_total,
                "active_users": self._active_users,
                "api_keys_used": self._api_keys_used,
                "uptime_seconds": time.time() - self._start_time,
            }
    
    # Record metrics for specific operations
    def record_request(self, path: str, method: str, duration_ms: float, status_code: int):
        """Record API request metrics."""
        labels = {"path": path, "method": method, "status": str(status_code)}
        
        self.increment_counter("http_requests_total", labels=labels)
        self.observe_histogram("http_request_duration_seconds", duration_ms / 1000, labels=labels)
        
        if status_code >= 400:
            self.increment_counter("http_errors_total", labels=labels)
    
    def record_database_query(self, query_type: str, duration_ms: float):
        """Record database query metrics."""
        labels = {"query_type": query_type}
        
        self.increment_counter("db_queries_total", labels=labels)
        self.observe_histogram("db_query_duration_seconds", duration_ms / 1000, labels=labels)
        
        if duration_ms > 1000:
            self.increment_counter("db_slow_queries_total", labels=labels)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics."""
        labels = {"operation": operation}
        
        self.increment_counter("cache_operations_total", labels=labels)
        
        if hit:
            self.increment_counter("cache_hits_total", labels=labels)
        else:
            self.increment_counter("cache_misses_total", labels=labels)
    
    def _build_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Build metric key with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        index = int(len(sorted_values) * percentile)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_request_metrics(path: str, method: str):
    """
    Decorator to track request metrics.
    
    Usage:
        @track_request_metrics("/api/v1/accounts", "GET")
        async def get_accounts():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                get_metrics_collector().record_request(path, method, duration_ms, status_code)
        
        return wrapper
    return decorator


def metric_timer(metric_name: str):
    """
    Decorator to time metric execution.
    
    Usage:
        @metric_timer("db_query_duration")
        async def query_database():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                get_metrics_collector().observe_histogram(metric_name, duration_ms / 1000)
        
        return wrapper
    return decorator


# Prometheus format exporter
class PrometheusExporter:
    """Export metrics in Prometheus format."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def export(self) -> str:
        """Export all metrics as Prometheus text format."""
        lines = ["# HELP jpmorgan_api_metrics JPMorgan Financial API Metrics"]
        lines.append("# TYPE jpmorgan_api_metrics gauge")
        
        # Business metrics
        business_metrics = self.collector.get_business_metrics()
        
        for key, value in business_metrics.items():
            lines.append(f"jpmorgan_{key} {value}")
        
        return "\n".join(lines) + "\n"


# Health check metrics
def get_health_metrics() -> Dict[str, Any]:
    """Get health check metrics."""
    collector = get_metrics_collector()
    
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - collector._start_time,
        "counters_count": len(collector._counters),
        "gauges_count": len(collector._gauges),
        "histograms_count": len(collector._histograms),
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    collector = MetricsCollector()
    collector.start()
    
    # Track some metrics
    collector.increment_counter("test_requests_total")
    collector.set_gauge("test_gauge", 42.0)
    collector.observe_histogram("test_histogram", 0.1)
    collector.observe_histogram("test_histogram", 0.2)
    collector.observe_histogram("test_histogram", 0.3)
    
    # Track business metrics
    collector.track_transaction(100.50, "USD")
    collector.track_transaction(250.00, "USD")
    collector.set_active_users(1500)
    collector.track_api_key_usage()
    
    # Get statistics
    stats = collector.get_histogram_stats("test_histogram")
    print(f"Histogram stats: {stats}")
    
    business = collector.get_business_metrics()
    print(f"Business metrics: {business}")
    
    # Export Prometheus format
    exporter = PrometheusExporter(collector)
    print(exporter.export())
    
    collector.stop()
    print("Metrics collector example completed")
