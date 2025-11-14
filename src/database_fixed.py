"""
Database connection and session management with SQLAlchemy
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session, relationship
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import os
from datetime import datetime, timezone
try:
    from config import config
except ImportError:
    class Config:
        DATABASE_URL = 'sqlite:///telemetry.db'
        TELEMETRY_BATCH_SIZE = 100
    config = Config()

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


class BusinessModel(Base):
    """SQLAlchemy model for businesses"""
    __tablename__ = 'businesses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    registration_number = Column(String)
    address = Column(Text)
    contact_info = Column(Text)  # JSON string
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AssetModel(Base):
    """SQLAlchemy model for assets"""
    __tablename__ = 'assets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    business_id = Column(Integer, ForeignKey('businesses.id'), nullable=False)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    acquisition_date = Column(DateTime)
    current_value = Column(Float)
    ownership_percentage = Column(Float, default=100.0)
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    business = relationship("BusinessModel", backref="assets")


class DatabaseManager:
    """Database connection manager with connection pooling"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or getattr(config, 'DATABASE_URL', 'sqlite:///telemetry.db')
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling"""
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        # Create tables
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
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

    def get_connection_stats(self) -> dict:
        """Get connection pool statistics"""
        return {}

    def close(self):
        """Close database connections"""
        if self.SessionLocal:
            self.SessionLocal.remove()
        if self.engine:
            self.engine.dispose()

class DummyQuery:
    def __init__(self, items):
        self.items = items
    def all(self):
        return self.items
    def filter(self, condition):
        return self
    def order_by(self, order):
        return self
    def limit(self, limit):
        return self
    def first(self):
        return self.items[0] if self.items else None


# Global database manager instance
db_manager = DatabaseManager()
