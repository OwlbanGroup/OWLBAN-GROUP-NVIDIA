"""
Comprehensive monitoring and alerting system
"""
import time
import psutil
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import json
import asyncio
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST
)
import structlog

from .logger import telemetry_logger
from .database import db_manager
from .async_utils import async_health_check

logger = structlog.get_logger()

# Custom registry for application metrics
app_registry = CollectorRegistry()

# Application Metrics
APP_START_TIME = Gauge('app_start_time_seconds', 'Application start time', registry=app_registry)
APP_UPTIME = Gauge('app_uptime_seconds', 'Application uptime in seconds', registry=app_registry)
APP_VERSION = Gauge('app_version', 'Application version', ['version'], registry=app_registry)

# System Metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage', registry=app_registry)
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage', registry=app_registry)
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage', ['mount_point'], registry=app_registry)
SYSTEM_NETWORK_IO = Counter('system_network_io_bytes', 'System network I/O bytes', ['direction'], registry=app_registry)

# Business Metrics
TELEMETRY_EVENTS_TOTAL = Counter('telemetry_events_total', 'Total telemetry events processed', ['status', 'event_type'], registry=app_registry)
TELEMETRY_PROCESSING_LATENCY = Histogram('telemetry_processing_latency_seconds', 'Telemetry processing latency', ['operation'], registry=app_registry)
BATCH_PROCESSING_SIZE = Histogram('batch_processing_size', 'Batch processing size distribution', registry=app_registry)
ANOMALY_DETECTION_ACCURACY = Gauge('anomaly_detection_accuracy', 'Anomaly detection accuracy percentage', registry=app_registry)

# Error Metrics
ERRORS_TOTAL = Counter('errors_total', 'Total errors by type', ['error_type', 'endpoint'], registry=app_registry)
VALIDATION_ERRORS = Counter('validation_errors_total', 'Validation errors', ['field', 'error_type'], registry=app_registry)

# Performance Metrics
DATABASE_CONNECTIONS_ACTIVE = Gauge('db_connections_active', 'Active database connections', registry=app_registry)
CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['cache_type'], registry=app_registry)
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses', ['cache_type'], registry=app_registry)

# External Service Metrics
EXTERNAL_API_CALLS = Counter('external_api_calls_total', 'External API calls', ['service', 'method', 'status'], registry=app_registry)
EXTERNAL_API_LATENCY = Histogram('external_api_latency_seconds', 'External API call latency', ['service', 'method'], registry=app_registry)

# Alert thresholds
ALERT_THRESHOLDS = {
    'cpu_usage_percent': 85.0,
    'memory_usage_percent': 90.0,
    'disk_usage_percent': 95.0,
    'error_rate_per_minute': 10.0,
    'response_time_seconds': 5.0,
    'db_connection_pool_usage': 0.9
}


class MetricsCollector:
    """Collect and expose application metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.last_collection = 0
        self.collection_interval = 30  # seconds
        self.alerts = []

        # Initialize start time metric
        APP_START_TIME.set(self.start_time)

    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)

            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    SYSTEM_DISK_USAGE.labels(mount_point=partition.mountpoint).set(usage.percent)
                except PermissionError:
                    continue

            # Network I/O (simplified)
            net_io = psutil.net_io_counters()
            SYSTEM_NETWORK_IO.labels(direction='sent')._value_set(net_io.bytes_sent)
            SYSTEM_NETWORK_IO.labels(direction='recv')._value_set(net_io.bytes_recv)

        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))

    def collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Uptime
            uptime = time.time() - self.start_time
            APP_UPTIME.set(uptime)

            # Database connection stats
            if hasattr(db_manager, 'get_connection_stats'):
                db_stats = db_manager.get_connection_stats()
                if db_stats:
                    DATABASE_CONNECTIONS_ACTIVE.set(db_stats.get('checkedout', 0))

        except Exception as e:
            logger.error("Failed to collect application metrics", error=str(e))

    def record_telemetry_event(self, status: str, event_type: str = 'unknown'):
        """Record telemetry event processing"""
        TELEMETRY_EVENTS_TOTAL.labels(status=status, event_type=event_type).inc()

    def record_processing_latency(self, operation: str, duration: float):
        """Record processing latency"""
        TELEMETRY_PROCESSING_LATENCY.labels(operation=operation).observe(duration)

    def record_batch_size(self, size: int):
        """Record batch processing size"""
        BATCH_PROCESSING_SIZE.observe(size)

    def record_error(self, error_type: str, endpoint: str = 'unknown'):
        """Record application error"""
        ERRORS_TOTAL.labels(error_type=error_type, endpoint=endpoint).inc()

    def record_validation_error(self, field: str, error_type: str):
        """Record validation error"""
        VALIDATION_ERRORS.labels(field=field, error_type=error_type).inc()

    def record_cache_operation(self, hit: bool, cache_type: str = 'default'):
        """Record cache operation"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()

    def record_external_api_call(self, service: str, method: str, status: int, latency: float):
        """Record external API call"""
        EXTERNAL_API_CALLS.labels(service=service, method=method, status=str(status)).inc()
        EXTERNAL_API_LATENCY.labels(service=service, method=method).observe(latency)

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []

        # CPU usage alert
        cpu_usage = SYSTEM_CPU_USAGE._value
        if cpu_usage and cpu_usage > ALERT_THRESHOLDS['cpu_usage_percent']:
            alerts.append({
                'type': 'cpu_usage_high',
                'severity': 'warning',
                'message': f'CPU usage is {cpu_usage:.1f}%, threshold: {ALERT_THRESHOLDS["cpu_usage_percent"]}%',
                'value': cpu_usage,
                'threshold': ALERT_THRESHOLDS['cpu_usage_percent']
            })

        # Memory usage alert
        memory_usage = SYSTEM_MEMORY_USAGE._value
        if memory_usage and memory_usage > ALERT_THRESHOLDS['memory_usage_percent']:
            alerts.append({
                'type': 'memory_usage_high',
                'severity': 'critical',
                'message': f'Memory usage is {memory_usage:.1f}%, threshold: {ALERT_THRESHOLDS["memory_usage_percent"]}%',
                'value': memory_usage,
                'threshold': ALERT_THRESHOLDS['memory_usage_percent']
            })

        # Error rate alert (simplified check)
        # In a real implementation, you'd track error rates over time windows

        return alerts

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(app_registry).decode('utf-8')

    async def collect_all_metrics(self):
        """Collect all metrics asynchronously"""
        current_time = time.time()

        # Only collect system metrics at intervals
        if current_time - self.last_collection > self.collection_interval:
            self.collect_system_metrics()
            self.last_collection = current_time

        self.collect_application_metrics()

        # Check for alerts
        alerts = self.check_alerts()
        if alerts:
            for alert in alerts:
                logger.warning("Alert triggered", **alert)

        return {
            'metrics_collected': True,
            'alerts': alerts,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
        self.health_status = {}

    async def perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks"""
        current_time = time.time()

        # Only perform full health checks at intervals
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                # Async health checks
                async_results = await async_health_check()

                # Synchronous health checks
                sync_results = self._perform_sync_health_checks()

                # Combine results
                self.health_status = {
                    **async_results,
                    **sync_results,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

                self.last_health_check = current_time

            except Exception as e:
                logger.error("Health check failed", error=str(e))
                self.health_status = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        else:
            # Return cached results
            self.health_status['cached'] = True

        return self.health_status

    def _perform_sync_health_checks(self) -> Dict[str, Any]:
        """Perform synchronous health checks"""
        results = {}

        # System resources
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            results['system'] = {
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'healthy': memory.percent < 95 and disk.percent < 95
            }
        except Exception as e:
            results['system'] = {'healthy': False, 'error': str(e)}

        # Configuration
        try:
            results['configuration'] = {
                'env_vars_present': bool(os.getenv('SECRET_KEY')),
                'healthy': True
            }
        except Exception as e:
            results['configuration'] = {'healthy': False, 'error': str(e)}

        return results

    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        return {
            'overall_health': self.health_status.get('healthy', False),
            'checks': self.health_status,
            'version': os.getenv('APP_VERSION', '1.0.0'),
            'uptime_seconds': time.time() - APP_START_TIME._value if APP_START_TIME._value else 0
        }


class AlertManager:
    """Alert management and notification system"""

    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.max_history_size = 1000

    def process_alerts(self, alerts: List[Dict[str, Any]]):
        """Process and manage alerts"""
        for alert in alerts:
            alert_id = f"{alert['type']}_{int(time.time())}"

            # Add to active alerts
            self.active_alerts[alert_id] = {
                **alert,
                'id': alert_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'status': 'active'
            }

            # Add to history
            self.alert_history.append(self.active_alerts[alert_id])
            if len(self.alert_history) > self.max_history_size:
                self.alert_history.pop(0)

            # Log alert
            logger.warning("Alert generated", alert_id=alert_id, **alert)

            # In a real implementation, you'd send notifications here
            # self._send_notification(alert)

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['status'] = 'resolved'
            self.active_alerts[alert_id]['resolved_at'] = datetime.now(timezone.utc).isoformat()
            logger.info("Alert resolved", alert_id=alert_id)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values() if alert['status'] == 'active']

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return self.alert_history[-limit:]

    def _send_notification(self, alert: Dict[str, Any]):
        """Send alert notification (placeholder)"""
        # In a real implementation, this would send emails, Slack messages, etc.
        print(f"ALERT: {alert['message']}")


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
alert_manager = AlertManager()
