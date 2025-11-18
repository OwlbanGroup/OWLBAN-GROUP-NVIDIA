"""
Monitoring and metrics collection for JP Morgan Financial APIs
"""
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()


class MetricsCollector:
    """Centralized metrics collection"""

    def __init__(self) -> None:
        """Initialize metrics collector"""
        # Request metrics
        self.request_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['service', 'method', 'endpoint', 'status']
        )

        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['service', 'method', 'endpoint']
        )

        # Business metrics
        self.business_operations = Counter(
            'business_operations_total',
            'Total business operations',
            ['service', 'operation', 'status']
        )

        # System metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['service']
        )

        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['service', 'cache_type']
        )

        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['service', 'cache_type']
        )

    def record_request(
        self,
        service: str,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ) -> None:
        """Record API request metrics"""
        self.request_counter.labels(
            service=service,
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()

        self.request_duration.labels(
            service=service,
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_business_operation(
        self,
        service: str,
        operation: str,
        status: str
    ) -> None:
        """Record business operation metrics"""
        self.business_operations.labels(
            service=service,
            operation=operation,
            status=status
        ).inc()

    def set_active_connections(self, service: str, count: int) -> None:
        """Set active connections gauge"""
        self.active_connections.labels(service=service).set(count)

    def record_cache_hit(self, service: str, cache_type: str) -> None:
        """Record cache hit"""
        self.cache_hits.labels(service=service, cache_type=cache_type).inc()

    def record_cache_miss(self, service: str, cache_type: str) -> None:
        """Record cache miss"""
        self.cache_misses.labels(service=service, cache_type=cache_type).inc()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        return {
            "requests": "tracked",
            "business_operations": "tracked",
            "cache": "tracked",
            "connections": "tracked"
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()
