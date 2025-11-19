"""
Handler for processing and storing telemetry data with optimized database operations
"""
import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psycopg2
import psycopg2.pool
from psycopg2.extras import execute_batch

from .data_processor import prepare_for_ml
from .logger import telemetry_logger
from .ml_model import AnomalyDetector
from .telemetry_parser import TelemetryEvent, TelemetryParser

try:
    from config import config  # type: ignore
except ImportError:
    # Fallback config if module not found
    class Config:
        DATABASE_URL = 'sqlite:///telemetry.db'
        TELEMETRY_BATCH_SIZE = 100
    config = Config()

# Query performance tracking
query_performance_stats: Dict[str, Dict[str, Union[int, float]]] = defaultdict(
    lambda: {
        'count': 0,
        'total_time': 0.0,
        'max_time': 0.0,
        'min_time': float('inf')
    }
)
query_performance_lock = threading.Lock()


def track_query_performance(query_name: str):
    """Decorator to track query execution time and performance metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Update performance stats
                with query_performance_lock:
                    stats = query_performance_stats[query_name]
                    stats['count'] += 1
                    stats['total_time'] += execution_time
                    stats['max_time'] = max(stats['max_time'], execution_time)
                    stats['min_time'] = min(stats['min_time'], execution_time)
                    stats['avg_time'] = stats['total_time'] / stats['count']

                # Log slow queries (> 1 second)
                if execution_time > 1.0:
                    telemetry_logger.get_logger().warning(
                        "Slow query detected: %s took %.3fs",
                        query_name, execution_time
                    )
                elif execution_time > 0.5:
                    telemetry_logger.get_logger().info(
                        "Query %s took %.3fs",
                        query_name, execution_time
                    )

                return result
            except Exception as e:
                execution_time = time.time() - start_time
                telemetry_logger.log_error(
                    e,
                    {
                        'context': f'query_{query_name}',
                        'execution_time': execution_time
                    }
                )
                raise
        return wrapper
    return decorator


def get_query_performance_stats() -> Dict[str, Any]:
    """Get query performance statistics"""
    with query_performance_lock:
        return {
            query_name: {
                'count': stats['count'],
                'avg_time': round(stats['avg_time'], 4),
                'max_time': round(stats['max_time'], 4),
                'min_time': (round(stats['min_time'], 4)
                            if stats['min_time'] != float('inf') else 0),
                'total_time': round(stats['total_time'], 2)
            }
            for query_name, stats in query_performance_stats.items()
        }


class TelemetryDatabase:
    """
    Optimized database handler for telemetry data with PostgreSQL
    and SQLite support
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_url = db_path or config.DATABASE_URL
        self.is_postgres = (
            self.db_url.startswith('postgresql://') or
            self.db_url.startswith('postgres://')
        )

        if self.is_postgres:
            # Initialize PostgreSQL connection pool
            try:
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=20,
                    dsn=self.db_url
                )
                telemetry_logger.get_logger().info(
                    "PostgreSQL connection pool initialized"
                )
            except Exception as e:
                telemetry_logger.log_error(
                    e, {'context': 'postgres_pool_init'}
                )
                raise
        else:
            # SQLite setup
            self.db_path = self.db_url.replace('sqlite:///', '')
            self.connection_pool = None

        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with connection pooling"""
        if self.is_postgres:
            conn = self.connection_pool.getconn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self.connection_pool.putconn(conn)
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def init_database(self):
        """Initialize the database and create tables with optimized indexes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Drop existing tables (for development - remove in production)
            cursor.execute('DROP TABLE IF EXISTS telemetry_events')
            cursor.execute('DROP TABLE IF EXISTS telemetry_metrics')

            # Create telemetry_events table with appropriate syntax
            if self.is_postgres:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS telemetry_events (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        operation VARCHAR(255) NOT NULL,
                        pfn VARCHAR(255) NOT NULL,
                        version VARCHAR(50),
                        event_name VARCHAR(255),
                        shell_id INTEGER,
                        event_flags INTEGER,
                        pg_name VARCHAR(255),
                        dvc_sample REAL,
                        flags INTEGER,
                        edition INTEGER,
                        epoch VARCHAR(50),
                        seq INTEGER,
                        data_type INTEGER,
                        is_required BOOLEAN,
                        data_category INTEGER,
                        product INTEGER,
                        priv_tags INTEGER,
                        policies INTEGER,
                        cv VARCHAR(255),
                        boot_id INTEGER,
                        os_name VARCHAR(100),
                        os_version VARCHAR(100),
                        exp_id VARCHAR(255),
                        app_id VARCHAR(255),
                        app_version VARCHAR(50),
                        is_1p INTEGER,
                        as_id INTEGER,
                        local_id VARCHAR(255),
                        device_class VARCHAR(100),
                        dev_make VARCHAR(100),
                        dev_model VARCHAR(100),
                        ticket_keys TEXT,
                        user_local_id VARCHAR(255),
                        tz VARCHAR(50),
                        pn1 VARCHAR(255),
                        p1 TEXT,
                        pn2 VARCHAR(255),
                        p2 TEXT,
                        pn3 VARCHAR(255),
                        p3 TEXT,
                        pn4 VARCHAR(255),
                        p4 TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create optimized indexes for PostgreSQL
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp '
                    'ON telemetry_events(timestamp DESC)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_operation '
                    'ON telemetry_events(operation)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_pfn '
                    'ON telemetry_events(pfn)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_device_class '
                    'ON telemetry_events(device_class)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_os_name '
                    'ON telemetry_events(os_name)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_composite '
                    'ON telemetry_events(operation, timestamp DESC)'
                )

            else:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS telemetry_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        pfn TEXT NOT NULL,
                        version TEXT,
                        event_name TEXT,
                        shell_id INTEGER,
                        event_flags INTEGER,
                        pg_name TEXT,
                        dvc_sample REAL,
                        flags INTEGER,
                        edition INTEGER,
                        epoch TEXT,
                        seq INTEGER,
                        data_type INTEGER,
                        is_required BOOLEAN,
                        data_category INTEGER,
                        product INTEGER,
                        priv_tags INTEGER,
                        policies INTEGER,
                        cv TEXT,
                        boot_id INTEGER,
                        os_name TEXT,
                        os_version TEXT,
                        exp_id TEXT,
                        app_id TEXT,
                        app_version TEXT,
                        is_1p INTEGER,
                        as_id INTEGER,
                        local_id TEXT,
                        device_class TEXT,
                        dev_make TEXT,
                        dev_model TEXT,
                        ticket_keys TEXT,
                        user_local_id TEXT,
                        tz TEXT,
                        pn1 TEXT,
                        p1 TEXT,
                        pn2 TEXT,
                        p2 TEXT,
                        pn3 TEXT,
                        p3 TEXT,
                        pn4 TEXT,
                        p4 TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create optimized indexes for SQLite
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp '
                    'ON telemetry_events(timestamp DESC)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_operation '
                    'ON telemetry_events(operation)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_pfn '
                    'ON telemetry_events(pfn)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_device_class '
                    'ON telemetry_events(device_class)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_os_name '
                    'ON telemetry_events(os_name)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_telemetry_composite '
                    'ON telemetry_events(operation, timestamp DESC)'
                )

            # Create telemetry_metrics table
            if self.is_postgres:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS telemetry_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(255) NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        operation VARCHAR(255),
                        pfn VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_metrics_timestamp '
                    'ON telemetry_metrics(timestamp DESC)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_metrics_name '
                    'ON telemetry_metrics(metric_name)'
                )
            else:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS telemetry_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        operation TEXT,
                        pfn TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_metrics_timestamp '
                    'ON telemetry_metrics(timestamp DESC)'
                )
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_metrics_name '
                    'ON telemetry_metrics(metric_name)'
                )

            db_type = 'PostgreSQL' if self.is_postgres else 'SQLite'
            telemetry_logger.get_logger().info(
                "Database initialized successfully (%s)", db_type
            )

    @track_query_performance('store_event')
    def store_event(self, event: TelemetryEvent) -> bool:
        """Store a telemetry event in the database with optimized query"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Use parameterized query for both PostgreSQL and SQLite
                placeholder = '%s' if self.is_postgres else '?'
                query = f'''
                    INSERT INTO telemetry_events (
                        timestamp, operation, pfn, version, event_name,
                        shell_id, event_flags, pg_name, dvc_sample, flags,
                        edition, epoch, seq, data_type, is_required,
                        data_category, product, priv_tags, policies, cv,
                        boot_id, os_name, os_version, exp_id, app_id,
                        app_version, is_1p, as_id, local_id, device_class,
                        dev_make, dev_model, ticket_keys, user_local_id, tz,
                        pn1, p1, pn2, p2, pn3, p3, pn4, p4
                    ) VALUES (
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}
                    )
                '''
                cursor.execute(query, (
                    event.timestamp, event.operation, event.pfn,
                    event.version, event.event_name, event.shell_id,
                    event.event_flags, event.pg_name, event.dvc_sample,
                    event.flags, event.edition, event.epoch, event.seq,
                    event.data_type, event.is_required, event.data_category,
                    event.product, event.priv_tags, event.policies, event.cv,
                    event.boot_id, event.os_name, event.os_version,
                    event.exp_id, event.app_id, event.app_version,
                    event.is_1p, event.as_id, event.local_id,
                    event.device_class, event.dev_make, event.dev_model,
                    json.dumps(event.ticket_keys), event.user_local_id,
                    event.tz, event.pn1, event.p1, event.pn2, event.p2,
                    event.pn3, event.p3, event.pn4, event.p4
                ))
                return True
        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'storing_telemetry_event'}
            )
            return False

    @track_query_performance('store_events_batch')
    def store_events_batch(self, events: List[TelemetryEvent]) -> int:
        """
        Store multiple telemetry events in a single batch operation
        for better performance
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                values = [
                    (
                        event.timestamp, event.operation, event.pfn,
                        event.version, event.event_name, event.shell_id,
                        event.event_flags, event.pg_name, event.dvc_sample,
                        event.flags, event.edition, event.epoch, event.seq,
                        event.data_type, event.is_required,
                        event.data_category, event.product, event.priv_tags,
                        event.policies, event.cv, event.boot_id,
                        event.os_name, event.os_version, event.exp_id,
                        event.app_id, event.app_version, event.is_1p,
                        event.as_id, event.local_id, event.device_class,
                        event.dev_make, event.dev_model,
                        json.dumps(event.ticket_keys), event.user_local_id,
                        event.tz, event.pn1, event.p1, event.pn2, event.p2,
                        event.pn3, event.p3, event.pn4, event.p4
                    )
                    for event in events
                ]

                placeholder = '%s' if self.is_postgres else '?'
                query = f'''
                    INSERT INTO telemetry_events (
                        timestamp, operation, pfn, version, event_name,
                        shell_id, event_flags, pg_name, dvc_sample, flags,
                        edition, epoch, seq, data_type, is_required,
                        data_category, product, priv_tags, policies, cv,
                        boot_id, os_name, os_version, exp_id, app_id,
                        app_version, is_1p, as_id, local_id, device_class,
                        dev_make, dev_model, ticket_keys, user_local_id, tz,
                        pn1, p1, pn2, p2, pn3, p3, pn4, p4
                    ) VALUES (
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}
                    )
                '''

                if self.is_postgres:
                    execute_batch(cursor, query, values, page_size=100)
                else:
                    cursor.executemany(query, values)

                return len(events)
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'storing_batch_events'})
            return 0

    def get_events_by_operation(
        self, operation: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get telemetry events by operation with optimized query and caching"""
        try:
            from app import cache_database_query
            return cache_database_query(expiration=300)(
                self._get_events_by_operation_uncached
            )(operation, limit)
        except ImportError:
            # Fallback if app module not available
            return self._get_events_by_operation_uncached(operation, limit)

    @track_query_performance('get_events_by_operation')
    def _get_events_by_operation_uncached(
        self, operation: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get telemetry events by operation without caching"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Use parameterized query with proper placeholder
                placeholder = '%s' if self.is_postgres else '?'
                cursor.execute(f'''
                    SELECT * FROM telemetry_events
                    WHERE operation = {placeholder}
                    ORDER BY timestamp DESC
                    LIMIT {placeholder}
                ''', (operation, limit))

                if self.is_postgres:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'getting_events_by_operation'}
            )
            return []

    @track_query_performance('get_metrics_summary')
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours with optimized queries"""
        try:
            cutoff_time = (
                datetime.utcnow() - timedelta(hours=hours)
            ).isoformat()
            placeholder = '%s' if self.is_postgres else '?'

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Count events by operation (optimized with index)
                cursor.execute(f'''
                    SELECT operation, COUNT(*) as count
                    FROM telemetry_events
                    WHERE timestamp > {placeholder}
                    GROUP BY operation
                    ORDER BY count DESC
                    LIMIT 50
                ''', (cutoff_time,))
                operation_counts = {row[0]: row[1] for row in cursor.fetchall()}

                # Count events by device class (optimized with index)
                cursor.execute(f'''
                    SELECT device_class, COUNT(*) as count
                    FROM telemetry_events
                    WHERE timestamp > {placeholder}
                    GROUP BY device_class
                    ORDER BY count DESC
                    LIMIT 50
                ''', (cutoff_time,))
                device_counts = {row[0]: row[1] for row in cursor.fetchall()}

                # Total events (optimized with index on timestamp)
                cursor.execute(f'''
                    SELECT COUNT(*) FROM telemetry_events
                    WHERE timestamp > {placeholder}
                ''', (cutoff_time,))
                total_events = cursor.fetchone()[0]

                # Average events per hour
                cursor.execute(f'''
                    SELECT COUNT(*) / {hours} as avg_per_hour
                    FROM telemetry_events
                    WHERE timestamp > {placeholder}
                ''', (cutoff_time,))
                avg_per_hour = cursor.fetchone()[0]

                return {
                    'total_events': total_events,
                    'operation_counts': operation_counts,
                    'device_counts': device_counts,
                    'time_period_hours': hours,
                    'avg_events_per_hour': (
                        round(avg_per_hour, 2) if avg_per_hour else 0
                    )
                }
        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'getting_metrics_summary'}
            )
            return {}

    @track_query_performance('cleanup_old_events')
    def cleanup_old_events(self, days: int = 30) -> int:
        """Clean up events older than specified days for database maintenance"""
        try:
            cutoff_time = (
                datetime.utcnow() - timedelta(days=days)
            ).isoformat()
            placeholder = '%s' if self.is_postgres else '?'

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    DELETE FROM telemetry_events
                    WHERE timestamp < {placeholder}
                ''', (cutoff_time,))
                deleted_count = cursor.rowcount

                telemetry_logger.get_logger().info(
                    "Cleaned up %d old events (older than %d days)",
                    deleted_count, days
                )
                return deleted_count
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'cleanup_old_events'})
            return 0

    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics for monitoring"""
        if not self.is_postgres or not self.connection_pool:
            return {'type': 'sqlite', 'pooling': False}

        try:
            # Get pool statistics
            connection_pool = self.connection_pool
            return {
                'type': 'postgresql',
                'pooling': True,
                'min_connections': connection_pool.minconn,
                'max_connections': connection_pool.maxconn,
                'closed': connection_pool.closed
            }
        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'get_connection_pool_stats'}
            )
            return {'type': 'postgresql', 'pooling': True, 'error': str(e)}


class TelemetryHandler:
    """Main handler for processing telemetry data with optimized batch operations"""

    def __init__(self):
        self.parser = TelemetryParser()
        self.database = TelemetryDatabase()
        self.batch_queue = deque(maxlen=config.TELEMETRY_BATCH_SIZE)
        self.lock = threading.Lock()
        self.anomaly_detector = AnomalyDetector()
        self._batch_insert_threshold = 10  # Use batch insert for 10+ events

    def process_single_event(self, raw_data: Dict[str, Any]) -> bool:
        """
        Process a single telemetry event

        Args:
            raw_data: Raw telemetry JSON data

        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Validate the data
            if not self.parser.validate_telemetry_data(raw_data):
                return False

            # Parse the data
            event = self.parser.parse_telemetry_data(raw_data)
            if not event:
                return False

            # Store in database
            if not self.database.store_event(event):
                return False

            return True

        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'processing_single_event'})
            return False

    def process_batch(
        self, telemetry_data_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a batch of telemetry events with async processing

        Args:
            telemetry_data_list: List of raw telemetry data

        Returns:
            Dictionary with processing statistics
        """
        stats: Dict[str, Any] = {
            'total': len(telemetry_data_list),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        try:
            # Try async batch processing for better performance
            import asyncio
            from .async_utils import process_batch_async

            async_results = asyncio.run(
                process_batch_async(telemetry_data_list)
            )

            # Count successful vs failed based on processing status
            for result in async_results:
                if result.get('_processing_status') == 'completed':
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1

        except (ImportError, Exception) as e:
            telemetry_logger.get_logger().error(
                "Async batch processing failed: %s", str(e)
            )
            # Fallback to synchronous processing
            for raw_data in telemetry_data_list:
                if self.process_single_event(raw_data):
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1

        # Log batch processing results
        telemetry_logger.get_logger().info(
            "Batch processing completed: %d/%d events processed successfully",
            stats['successful'], stats['total']
        )

        return stats

    def add_to_batch_queue(self, raw_data: Dict[str, Any]) -> bool:
        """
        Add telemetry data to batch processing queue

        Args:
            raw_data: Raw telemetry JSON data

        Returns:
            True if added successfully, False otherwise
        """
        try:
            with self.lock:
                self.batch_queue.append(raw_data)

                # Process batch if queue is full
                if len(self.batch_queue) >= config.TELEMETRY_BATCH_SIZE:
                    self._process_batch_queue()

                return True

        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'adding_to_batch_queue'}
            )
            return False

    def _process_batch_queue(self) -> Dict[str, Any]:
        """Process the current batch queue"""
        try:
            with self.lock:
                if not self.batch_queue:
                    return {'total': 0, 'successful': 0, 'failed': 0}

                batch_data = list(self.batch_queue)
                self.batch_queue.clear()

            return self.process_batch(batch_data)

        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'processing_batch_queue'}
            )
            return {'total': 0, 'successful': 0, 'failed': 0}

    def get_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get telemetry metrics for the specified time period"""
        return self.database.get_metrics_summary(hours)

    def export_events(
        self, operation: Optional[str] = None, limit: int = 1000,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Export telemetry events to file or return as list

        Args:
            operation: Filter by operation (optional)
            limit: Maximum number of events to export
            output_file: File path to export to (optional)

        Returns:
            List of event dictionaries
        """
        try:
            if operation:
                events = self.database.get_events_by_operation(
                    operation, limit
                )
            else:
                # Get all events (limited)
                with sqlite3.connect(self.database.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT * FROM telemetry_events
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                    events = [dict(row) for row in cursor.fetchall()]

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(events, f, indent=2, default=str)
                telemetry_logger.get_logger().info(
                    "Exported %d events to %s", len(events), output_file
                )

            return events

        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'exporting_events'})
            return []

    def detect_anomalies_in_batch(
        self, telemetry_data_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies in a batch of telemetry data using ML

        Args:
            telemetry_data_list: List of raw telemetry data

        Returns:
            Dictionary with anomaly detection results
        """
        try:
            if not telemetry_data_list:
                return {'anomalies': [], 'total': 0}

            # Prepare data for ML
            feature_matrix, features_df = prepare_for_ml(telemetry_data_list)

            if feature_matrix.shape[0] < 10:
                return {
                    'anomalies': [],
                    'total': 0,
                    'message': 'Not enough data for anomaly detection'
                }

            # Train model if not trained
            if not self.anomaly_detector.is_trained:
                self.anomaly_detector.train(feature_matrix)

            # Predict anomalies
            anomalies = self.anomaly_detector.predict(feature_matrix)

            # Get anomaly indices
            anomaly_indices = np.where(anomalies == 1)[0]

            # Log anomalies
            telemetry_logger.get_logger().info(
                "Detected %d anomalies in %d events",
                len(anomaly_indices), len(telemetry_data_list)
            )

            return {
                'total': len(telemetry_data_list),
                'anomalies_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_data': features_df.iloc[anomaly_indices].to_dict(
                    'records'
                )
            }

        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'detecting_anomalies'}
            )
            return {'error': str(e), 'total': len(telemetry_data_list)}

    def train_anomaly_model(
        self, telemetry_data_list: List[Dict[str, Any]]
    ) -> bool:
        """
        Train the anomaly detection model with provided data

        Args:
            telemetry_data_list: List of raw telemetry data for training

        Returns:
            True if training successful, False otherwise
        """
        try:
            feature_matrix, _ = prepare_for_ml(telemetry_data_list)
            self.anomaly_detector.train(feature_matrix)
            telemetry_logger.get_logger().info(
                "Anomaly detection model trained successfully"
            )
            return True
        except Exception as e:
            telemetry_logger.log_error(
                e, {'context': 'training_anomaly_model'}
            )
            return False


# Global handler instance
telemetry_handler = TelemetryHandler()
