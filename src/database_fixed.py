"""
Database connection and session management with SQLAlchemy
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session, relationship
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, List, Optional
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

# Import models after Base is defined
try:
    from src.models.audit_log import AuditLogModel
except ImportError:
    # AuditLogModel will be imported later if not available
    AuditLogModel = None

try:
    from src.models.revenue import RevenueTransaction, RevenueMetrics
except ImportError:
    # Revenue models will be imported later if not available
    RevenueTransaction = None
    RevenueMetrics = None

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
        
        # Create tables (including audit_logs and revenue models if available)
        Base.metadata.create_all(bind=self.engine)

        # Create audit_logs table if AuditLogModel is available
        if AuditLogModel is not None:
            try:
                AuditLogModel.__table__.create(bind=self.engine, checkfirst=True)
            except Exception as e:
                print(f"Note: Audit log table creation skipped: {e}")

        # Create revenue tables if models are available
        if RevenueTransaction is not None:
            try:
                RevenueTransaction.__table__.create(bind=self.engine, checkfirst=True)
                RevenueMetrics.__table__.create(bind=self.engine, checkfirst=True)
            except Exception as e:
                print(f"Note: Revenue table creation skipped: {e}")

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
    
    # Audit Log Methods
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List:
        """
        Get audit logs with filters
        
        Args:
            user_id: Filter by user ID
            action: Filter by action
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of audit log records
        """
        if AuditLogModel is None:
            return []
        
        try:
            with self.get_session() as session:
                query = session.query(AuditLogModel)
                
                if user_id:
                    query = query.filter(AuditLogModel.user_id == user_id)
                if action:
                    query = query.filter(AuditLogModel.action == action)
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                
                query = query.order_by(AuditLogModel.timestamp.desc())
                query = query.limit(limit).offset(offset)
                
                return query.all()
        except Exception as e:
            print(f"Failed to get audit logs: {e}")
            return []
    
    def get_audit_log_count(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> int:
        """
        Get count of audit logs matching filters
        
        Args:
            user_id: Filter by user ID
            action: Filter by action
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Count of matching audit logs
        """
        if AuditLogModel is None:
            return 0
        
        try:
            with self.get_session() as session:
                query = session.query(AuditLogModel)
                
                if user_id:
                    query = query.filter(AuditLogModel.user_id == user_id)
                if action:
                    query = query.filter(AuditLogModel.action == action)
                if start_date:
                    query = query.filter(AuditLogModel.timestamp >= start_date)
                if end_date:
                    query = query.filter(AuditLogModel.timestamp <= end_date)
                
                return query.count()
        except Exception as e:
            print(f"Failed to get audit log count: {e}")
            return 0
    
    def cleanup_old_audit_logs(self, retention_days: int = 90) -> int:
        """
        Clean up audit logs older than retention period
        
        Args:
            retention_days: Number of days to retain logs
            
        Returns:
            Number of logs deleted
        """
        if AuditLogModel is None:
            return 0
        
        try:
            from datetime import timedelta
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            
            with self.get_session() as session:
                deleted = session.query(AuditLogModel).filter(
                    AuditLogModel.timestamp < cutoff_date
                ).delete()
                session.commit()
                return deleted
        except Exception as e:
            print(f"Failed to cleanup audit logs: {e}")
            return 0

    # Business Management Methods
    def create_business(self, business_data: dict) -> BusinessModel:
        """Create a new business"""
        try:
            with self.get_session() as session:
                # Convert contact_info to JSON string if it's a dict
                if 'contact_info' in business_data and isinstance(business_data['contact_info'], dict):
                    import json
                    business_data['contact_info'] = json.dumps(business_data['contact_info'])

                business = BusinessModel(**business_data)
                session.add(business)
                session.commit()
                session.refresh(business)
                return business
        except Exception as e:
            print(f"Failed to create business: {e}")
            raise

    def get_business_by_id(self, business_id: int) -> Optional[BusinessModel]:
        """Get business by ID"""
        try:
            with self.get_session() as session:
                return session.query(BusinessModel).filter(BusinessModel.id == business_id).first()
        except Exception as e:
            print(f"Failed to get business: {e}")
            return None

    def get_all_businesses(self) -> List[BusinessModel]:
        """Get all businesses"""
        try:
            with self.get_session() as session:
                return session.query(BusinessModel).all()
        except Exception as e:
            print(f"Failed to get businesses: {e}")
            return []

    def update_business(self, business_id: int, update_data: dict) -> Optional[BusinessModel]:
        """Update business details"""
        try:
            with self.get_session() as session:
                business = session.query(BusinessModel).filter(BusinessModel.id == business_id).first()
                if not business:
                    return None
                for key, value in update_data.items():
                    if hasattr(business, key):
                        setattr(business, key, value)
                session.commit()
                session.refresh(business)
                return business
        except Exception as e:
            print(f"Failed to update business: {e}")
            return None

    def delete_business(self, business_id: int) -> bool:
        """Delete a business"""
        try:
            with self.get_session() as session:
                business = session.query(BusinessModel).filter(BusinessModel.id == business_id).first()
                if not business:
                    return False
                session.delete(business)
                session.commit()
                return True
        except Exception as e:
            print(f"Failed to delete business: {e}")
            return False

    # Asset Management Methods
    def create_asset(self, asset_data: dict) -> AssetModel:
        """Create a new asset"""
        try:
            with self.get_session() as session:
                # Convert acquisition_date from string to datetime if needed
                if 'acquisition_date' in asset_data and isinstance(asset_data['acquisition_date'], str):
                    try:
                        asset_data['acquisition_date'] = datetime.fromisoformat(asset_data['acquisition_date'].replace('Z', '+00:00'))
                    except ValueError:
                        # If conversion fails, set to None
                        asset_data['acquisition_date'] = None

                asset = AssetModel(**asset_data)
                session.add(asset)
                session.commit()
                session.refresh(asset)
                return asset
        except Exception as e:
            print(f"Failed to create asset: {e}")
            raise

    def get_asset_by_id(self, asset_id: int) -> Optional[AssetModel]:
        """Get asset by ID"""
        try:
            with self.get_session() as session:
                return session.query(AssetModel).filter(AssetModel.id == asset_id).first()
        except Exception as e:
            print(f"Failed to get asset: {e}")
            return None

    def get_all_assets(self) -> List[AssetModel]:
        """Get all assets"""
        try:
            with self.get_session() as session:
                return session.query(AssetModel).all()
        except Exception as e:
            print(f"Failed to get assets: {e}")
            return []

    def get_assets_by_business_id(self, business_id: int) -> List[AssetModel]:
        """Get assets for a specific business"""
        try:
            with self.get_session() as session:
                return session.query(AssetModel).filter(AssetModel.business_id == business_id).all()
        except Exception as e:
            print(f"Failed to get assets for business: {e}")
            return []

    def update_asset(self, asset_id: int, update_data: dict) -> Optional[AssetModel]:
        """Update asset details"""
        try:
            with self.get_session() as session:
                asset = session.query(AssetModel).filter(AssetModel.id == asset_id).first()
                if not asset:
                    return None
                for key, value in update_data.items():
                    if hasattr(asset, key):
                        setattr(asset, key, value)
                session.commit()
                session.refresh(asset)
                return asset
        except Exception as e:
            print(f"Failed to update asset: {e}")
            return None

    def delete_asset(self, asset_id: int) -> bool:
        """Delete an asset"""
        try:
            with self.get_session() as session:
                asset = session.query(AssetModel).filter(AssetModel.id == asset_id).first()
                if not asset:
                    return False
                session.delete(asset)
                session.commit()
                return True
        except Exception as e:
            print(f"Failed to delete asset: {e}")
            return False


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
