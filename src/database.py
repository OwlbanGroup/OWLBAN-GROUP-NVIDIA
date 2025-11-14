"""
Database connection and session management with SQLAlchemy
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import os
from datetime import datetime, timezone
from config import config

Base = declarative_base()

class TelemetryEventModel(Base):
    """SQLAlchemy model for telemetry events"""
    __tablename__ = 'telemetry_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    operation = Column(String, nullable=False)
    pfn = Column(String, nullable=False)
    version = Column(String)
    event_name = Column(String)
    shell_id = Column(Integer)
    event_flags = Column(Integer)
    pg_name = Column(String)
    dvc_sample = Column(Float)
    flags = Column(Integer)
    edition = Column(Integer)
    epoch = Column(String)
    seq = Column(Integer)
    data_type = Column(Integer)
    is_required = Column(Boolean)
    data_category = Column(Integer)
    product = Column(Integer)
    priv_tags = Column(Integer)
    policies = Column(Integer)
    cv = Column(String)
    boot_id = Column(Integer)
    os_name = Column(String)
    os_version = Column(String)
    exp_id = Column(String)
    app_id = Column(String)
    app_version = Column(String)
    is_1p = Column(Integer)
    as_id = Column(Integer)
    local_id = Column(String)
    device_class = Column(String)
    dev_make = Column(String)
    dev_model = Column(String)
    ticket_keys = Column(Text)  # JSON string
    user_local_id = Column(String)
    tz = Column(String)
    pn1 = Column(String)
    p1 = Column(String)
    pn2 = Column(String)
    p2 = Column(String)
    pn3 = Column(String)
    p3 = Column(String)
    pn4 = Column(String)
    p4 = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TelemetryMetricsModel(Base):
    """SQLAlchemy model for telemetry metrics"""
    __tablename__ = 'telemetry_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metadata = Column(Text)  # JSON string


class DatabaseManager:
    """Database connection manager with connection pooling and encryption"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self.encryption_key = config.SECRET_KEY[:32] if config.SECRET_KEY else None  # Use first 32 chars for AES-256
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling"""
        # Convert SQLite URL to SQLAlchemy format
        if self.database_url.startswith('sqlite:///'):
            db_path = self.database_url.replace('sqlite:///', '')
            self.database_url = f'sqlite:///{db_path}'

        # Create engine with connection pooling
        connect_args = {}
        if 'sqlite' in self.database_url:
            connect_args = {"check_same_thread": False}

        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,  # Maximum number of connections
            max_overflow=20,  # Additional connections beyond pool_size
            pool_timeout=30,  # Timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections after 1 hour
            connect_args=connect_args,
            echo=False  # Set to True for SQL debugging
        )

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create scoped session for thread-local sessions
        self.session = scoped_session(self.SessionLocal)

        # Create tables
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception:
            return False

    def start_health_monitoring(self, interval_seconds: int = 60):
        """Start background health monitoring"""
        import threading
        import time

        def monitor_health():
            while True:
                try:
                    healthy = self.health_check()
                    stats = self.get_connection_stats()

                    if not healthy:
                        telemetry_logger.get_logger().error("Database health check failed")
                    else:
                        telemetry_logger.get_logger().debug(f"Database healthy - Pool stats: {stats}")

                    # Update Prometheus metrics if available
                    try:
                        from prometheus_client import Gauge
                        db_health_metric = Gauge('database_health_status', 'Database health status (1=healthy, 0=unhealthy)')
                        db_health_metric.set(1 if healthy else 0)

                        if stats:
                            pool_size_metric = Gauge('database_pool_size', 'Database connection pool size')
                            pool_size_metric.set(stats.get('pool_size', 0))
                    except ImportError:
                        pass

                except Exception as e:
                    telemetry_logger.get_logger().error(f"Health monitoring error: {e}")

                time.sleep(interval_seconds)

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_health, daemon=True, name="db-health-monitor")
        monitor_thread.start()
        telemetry_logger.get_logger().info(f"Database health monitoring started (interval: {interval_seconds}s)")

    def get_connection_stats(self) -> dict:
        """Get connection pool statistics"""
        if hasattr(self.engine.pool, '_pool'):
            pool = self.engine.pool._pool
            return {
                'pool_size': getattr(pool, 'size', 0),
                'checkedin': getattr(pool, 'checkedin', 0),
                'checkedout': getattr(pool, 'checkedout', 0),
                'invalid': getattr(pool, 'invalid', 0),
                'overflow': getattr(pool, 'overflow', 0)
            }
        return {}

    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()
