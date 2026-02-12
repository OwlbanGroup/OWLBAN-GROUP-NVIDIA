"""
Async database connection and session management with SQLAlchemy
"""
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, text, select, update, delete, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import contextlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Dict, Any, Generator
import os
from datetime import datetime, timezone
try:
    from config import config
except ImportError:
    class Config:
        DATABASE_URL = 'sqlite:///telemetry.db'
        TELEMETRY_BATCH_SIZE = 100
    config = Config()

# Import shared Base and models
from src.models.base import Base
from src.models.audit_log import AuditLogModel
from src.models.revenue import RevenueTransaction, RevenueMetrics

class TelemetryEventModel(Base):
    """SQLAlchemy model for telemetry events"""
    __tablename__ = 'telemetry_events'
    __table_args__ = {'extend_existing': True}

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
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DBBusinessModel(Base):
    """SQLAlchemy model for businesses"""
    __tablename__ = 'businesses'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    registration_number = Column(String)
    address = Column(Text)
    contact_info = Column(Text)  # JSON string
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class DBAssetModel(Base):
    """SQLAlchemy model for assets"""
    __tablename__ = 'assets'
    __table_args__ = {'extend_existing': True}

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

    business = relationship("DBBusinessModel", back_populates="assets", overlaps="business")


class DBOrganizationModel(Base):
    """SQLAlchemy model for organizations"""
    __tablename__ = 'organizations'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    registration_number = Column(String)
    address = Column(Text)
    contact_info = Column(Text)  # JSON string
    owner_id = Column(String, nullable=False)  # Docker ID of the owner
    subscription_type = Column(String, default='team')  # team, business, etc.
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    members = relationship("DBOrganizationMemberModel", back_populates="organization", overlaps="organization")


class DBOrganizationMemberModel(Base):
    """SQLAlchemy model for organization members"""
    __tablename__ = 'organization_members'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    user_id = Column(String, nullable=False)  # Docker ID
    role = Column(String, default='member')  # owner, admin, member
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    organization = relationship("DBOrganizationModel", back_populates="members", overlaps="organization")

# Define relationships after both classes are defined
DBBusinessModel.assets = relationship("DBAssetModel", back_populates="business", overlaps="business")


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

    @contextlib.contextmanager
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
                session.execute(text("SELECT 1"))
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

    def initialize_database(self):
        """Initialize database tables in correct order"""
        try:
            # Create base tables first
            Base.metadata.create_all(bind=self.engine)

            # Create businesses table explicitly first
            DBBusinessModel.__table__.create(bind=self.engine, checkfirst=True)

            # Create audit_logs table if available
            if AuditLogModel is not None:
                try:
                    AuditLogModel.__table__.create(bind=self.engine, checkfirst=True)
                except Exception as e:
                    print(f"Note: Audit log table creation skipped: {e}")

            # Create revenue tables after businesses table
            if RevenueTransaction is not None:
                try:
                    RevenueTransaction.__table__.create(bind=self.engine, checkfirst=True)
                    RevenueMetrics.__table__.create(bind=self.engine, checkfirst=True)
                except Exception as e:
                    print(f"Note: Revenue table creation skipped: {e}")

            print("Database initialized successfully")
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            raise
    
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
    def create_business(self, business_data: dict) -> DBBusinessModel:
        """Create a new business"""
        try:
            with self.get_session() as session:
                # Convert contact_info to JSON string if it's a dict
                if 'contact_info' in business_data and isinstance(business_data['contact_info'], dict):
                    import json
                    business_data['contact_info'] = json.dumps(business_data['contact_info'])

                business = DBBusinessModel(**business_data)
                session.add(business)
                session.commit()
                session.refresh(business)
                return business
        except Exception as e:
            print(f"Failed to create business: {e}")
            raise

    def get_business_by_id(self, business_id: int) -> Optional[DBBusinessModel]:
        """Get business by ID"""
        try:
            with self.get_session() as session:
                return session.query(DBBusinessModel).filter(DBBusinessModel.id == business_id).first()
        except Exception as e:
            print(f"Failed to get business: {e}")
            return None

    def get_all_businesses(self) -> List[DBBusinessModel]:
        """Get all businesses"""
        try:
            with self.get_session() as session:
                return session.query(DBBusinessModel).all()
        except Exception as e:
            print(f"Failed to get businesses: {e}")
            return []

    def update_business(self, business_id: int, update_data: dict) -> Optional[DBBusinessModel]:
        """Update business details"""
        try:
            with self.get_session() as session:
                business = session.query(DBBusinessModel).filter(DBBusinessModel.id == business_id).first()
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
                business = session.query(DBBusinessModel).filter(DBBusinessModel.id == business_id).first()
                if not business:
                    return False
                session.delete(business)
                session.commit()
                return True
        except Exception as e:
            print(f"Failed to delete business: {e}")
            return False

    # Asset Management Methods
    def create_asset(self, asset_data: dict) -> DBAssetModel:
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

                asset = DBAssetModel(**asset_data)
                session.add(asset)
                session.commit()
                session.refresh(asset)
                return asset
        except Exception as e:
            print(f"Failed to create asset: {e}")
            raise

    def get_asset_by_id(self, asset_id: int) -> Optional[DBAssetModel]:
        """Get asset by ID"""
        try:
            with self.get_session() as session:
                return session.query(DBAssetModel).filter(DBAssetModel.id == asset_id).first()
        except Exception as e:
            print(f"Failed to get asset: {e}")
            return None

    def get_all_assets(self) -> List[DBAssetModel]:
        """Get all assets"""
        try:
            with self.get_session() as session:
                return session.query(DBAssetModel).all()
        except Exception as e:
            print(f"Failed to get assets: {e}")
            return []

    def get_assets_by_business_id(self, business_id: int) -> List[DBAssetModel]:
        """Get assets for a specific business"""
        try:
            with self.get_session() as session:
                return session.query(DBAssetModel).filter(DBAssetModel.business_id == business_id).all()
        except Exception as e:
            print(f"Failed to get assets for business: {e}")
            return []

    def update_asset(self, asset_id: int, update_data: dict) -> Optional[DBAssetModel]:
        """Update asset details"""
        try:
            with self.get_session() as session:
                asset = session.query(DBAssetModel).filter(DBAssetModel.id == asset_id).first()
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
                asset = session.query(DBAssetModel).filter(DBAssetModel.id == asset_id).first()
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


class AsyncDatabaseManager:
    """Async database connection manager with connection pooling"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or getattr(config, 'DATABASE_URL', 'sqlite:///telemetry.db')
        # Convert to async URL for PostgreSQL
        if self.database_url.startswith('postgresql://'):
            self.database_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
        elif self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql+asyncpg://')

        self.engine = None
        self.AsyncSessionLocal = None
        self.is_sqlite = 'sqlite' in self.database_url

        # Skip initialization for SQLite as it doesn't support async
        if self.is_sqlite:
            return

        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize async SQLAlchemy engine with connection pooling"""
        self.engine = create_async_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            echo=False
        )
        self.AsyncSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            class_=AsyncSession
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup"""
        session = self.AsyncSessionLocal()
        try:
            yield session
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """Check database connectivity"""
        if self.is_sqlite:
            # For SQLite, use sync manager
            return db_manager.health_check()
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {}

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()

    async def initialize_database(self):
        """Initialize database tables in correct order"""
        if self.is_sqlite:
            # For SQLite, use sync manager
            db_manager.initialize_database()
            return

        try:
            # Create base tables first
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Create businesses table explicitly first
            async with self.engine.begin() as conn:
                await conn.run_sync(DBBusinessModel.__table__.create, checkfirst=True)

            # Create organizations table after businesses table
            async with self.engine.begin() as conn:
                await conn.run_sync(DBOrganizationModel.__table__.create, checkfirst=True)
                await conn.run_sync(DBOrganizationMemberModel.__table__.create, checkfirst=True)

            # Create audit_logs table if available
            if AuditLogModel is not None:
                try:
                    async with self.engine.begin() as conn:
                        await conn.run_sync(AuditLogModel.__table__.create, checkfirst=True)
                except Exception as e:
                    print(f"Note: Audit log table creation skipped: {e}")

            # Create revenue tables after businesses table
            if RevenueTransaction is not None:
                try:
                    async with self.engine.begin() as conn:
                        await conn.run_sync(RevenueTransaction.__table__.create, checkfirst=True)
                        await conn.run_sync(RevenueMetrics.__table__.create, checkfirst=True)
                except Exception as e:
                    print(f"Note: Revenue table creation skipped: {e}")

            print("Database initialized successfully")
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            raise

    # Business Management Methods
    async def create_business(self, business_data: Dict[str, Any]) -> DBBusinessModel:
        """Create a new business"""
        try:
            async with self.get_session() as session:
                # Convert contact_info to JSON string if it's a dict
                if 'contact_info' in business_data and isinstance(business_data['contact_info'], dict):
                    import json
                    business_data['contact_info'] = json.dumps(business_data['contact_info'])

                business = DBBusinessModel(**business_data)
                session.add(business)
                await session.commit()
                await session.refresh(business)
                return business
        except Exception as e:
            print(f"Failed to create business: {e}")
            raise

    async def get_business_by_id(self, business_id: int) -> Optional[DBBusinessModel]:
        """Get business by ID"""
        try:
            async with self.get_session() as session:
                query = select(DBBusinessModel).where(DBBusinessModel.id == business_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            print(f"Failed to get business: {e}")
            return None

    async def get_all_businesses(self) -> List[DBBusinessModel]:
        """Get all businesses"""
        try:
            async with self.get_session() as session:
                query = select(DBBusinessModel)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            print(f"Failed to get businesses: {e}")
            return []

    async def update_business(self, business_id: int, update_data: Dict[str, Any]) -> Optional[DBBusinessModel]:
        """Update business details"""
        try:
            async with self.get_session() as session:
                query = select(DBBusinessModel).where(DBBusinessModel.id == business_id)
                result = await session.execute(query)
                business = result.scalar_one_or_none()
                if not business:
                    return None
                for key, value in update_data.items():
                    if hasattr(business, key):
                        setattr(business, key, value)
                await session.commit()
                await session.refresh(business)
                return business
        except Exception as e:
            print(f"Failed to update business: {e}")
            return None

    async def delete_business(self, business_id: int) -> bool:
        """Delete a business"""
        try:
            async with self.get_session() as session:
                query = select(DBBusinessModel).where(DBBusinessModel.id == business_id)
                result = await session.execute(query)
                business = result.scalar_one_or_none()
                if not business:
                    return False
                await session.delete(business)
                await session.commit()
                return True
        except Exception as e:
            print(f"Failed to delete business: {e}")
            return False

    # Asset Management Methods
    async def create_asset(self, asset_data: Dict[str, Any]) -> DBAssetModel:
        """Create a new asset"""
        try:
            async with self.get_session() as session:
                # Convert acquisition_date from string to datetime if needed
                if 'acquisition_date' in asset_data and isinstance(asset_data['acquisition_date'], str):
                    try:
                        asset_data['acquisition_date'] = datetime.fromisoformat(asset_data['acquisition_date'].replace('Z', '+00:00'))
                    except ValueError:
                        # If conversion fails, set to None
                        asset_data['acquisition_date'] = None

                asset = DBAssetModel(**asset_data)
                session.add(asset)
                await session.commit()
                await session.refresh(asset)
                return asset
        except Exception as e:
            print(f"Failed to create asset: {e}")
            raise

    async def get_asset_by_id(self, asset_id: int) -> Optional[DBAssetModel]:
        """Get asset by ID"""
        try:
            async with self.get_session() as session:
                query = select(DBAssetModel).where(DBAssetModel.id == asset_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            print(f"Failed to get asset: {e}")
            return None

    async def get_all_assets(self) -> List[DBAssetModel]:
        """Get all assets"""
        try:
            async with self.get_session() as session:
                query = select(DBAssetModel)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            print(f"Failed to get assets: {e}")
            return []

    async def get_assets_by_business_id(self, business_id: int) -> List[DBAssetModel]:
        """Get assets for a specific business"""
        try:
            async with self.get_session() as session:
                query = select(DBAssetModel).where(DBAssetModel.business_id == business_id)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            print(f"Failed to get assets for business: {e}")
            return []

    async def update_asset(self, asset_id: int, update_data: Dict[str, Any]) -> Optional[DBAssetModel]:
        """Update asset details"""
        try:
            async with self.get_session() as session:
                query = select(DBAssetModel).where(DBAssetModel.id == asset_id)
                result = await session.execute(query)
                asset = result.scalar_one_or_none()
                if not asset:
                    return None
                for key, value in update_data.items():
                    if hasattr(asset, key):
                        setattr(asset, key, value)
                await session.commit()
                await session.refresh(asset)
                return asset
        except Exception as e:
            print(f"Failed to update asset: {e}")
            return None

    async def delete_asset(self, asset_id: int) -> bool:
        """Delete an asset"""
        try:
            async with self.get_session() as session:
                query = select(DBAssetModel).where(DBAssetModel.id == asset_id)
                result = await session.execute(query)
                asset = result.scalar_one_or_none()
                if not asset:
                    return False
                await session.delete(asset)
                await session.commit()
                return True
        except Exception as e:
            print(f"Failed to delete asset: {e}")
            return False

    # Organization Management Methods
    async def create_organization(self, organization_data: Dict[str, Any]) -> DBOrganizationModel:
        """Create a new organization"""
        try:
            async with self.get_session() as session:
                # Convert contact_info to JSON string if it's a dict
                if 'contact_info' in organization_data and isinstance(organization_data['contact_info'], dict):
                    import json
                    organization_data['contact_info'] = json.dumps(organization_data['contact_info'])

                organization = DBOrganizationModel(**organization_data)
                session.add(organization)
                await session.commit()
                await session.refresh(organization)
                return organization
        except Exception as e:
            print(f"Failed to create organization: {e}")
            raise

    async def get_organization_by_id(self, organization_id: int) -> Optional[DBOrganizationModel]:
        """Get organization by ID"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationModel).where(DBOrganizationModel.id == organization_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            print(f"Failed to get organization: {e}")
            return None

    async def get_organization_by_owner_id(self, owner_id: str) -> Optional[DBOrganizationModel]:
        """Get organization by owner ID"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationModel).where(DBOrganizationModel.owner_id == owner_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            print(f"Failed to get organization by owner: {e}")
            return None

    async def get_all_organizations(self) -> List[DBOrganizationModel]:
        """Get all organizations"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationModel)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            print(f"Failed to get organizations: {e}")
            return []

    async def update_organization(self, organization_id: int, update_data: Dict[str, Any]) -> Optional[DBOrganizationModel]:
        """Update organization details"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationModel).where(DBOrganizationModel.id == organization_id)
                result = await session.execute(query)
                organization = result.scalar_one_or_none()
                if not organization:
                    return None
                for key, value in update_data.items():
                    if hasattr(organization, key):
                        setattr(organization, key, value)
                await session.commit()
                await session.refresh(organization)
                return organization
        except Exception as e:
            print(f"Failed to update organization: {e}")
            return None

    async def delete_organization(self, organization_id: int) -> bool:
        """Delete an organization"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationModel).where(DBOrganizationModel.id == organization_id)
                result = await session.execute(query)
                organization = result.scalar_one_or_none()
                if not organization:
                    return False
                await session.delete(organization)
                await session.commit()
                return True
        except Exception as e:
            print(f"Failed to delete organization: {e}")
            return False

    # Organization Member Management Methods
    async def add_organization_member(self, member_data: Dict[str, Any]) -> DBOrganizationMemberModel:
        """Add a member to an organization"""
        try:
            async with self.get_session() as session:
                member = DBOrganizationMemberModel(**member_data)
                session.add(member)
                await session.commit()
                await session.refresh(member)
                return member
        except Exception as e:
            print(f"Failed to add organization member: {e}")
            raise

    async def get_organization_members(self, organization_id: int) -> List[DBOrganizationMemberModel]:
        """Get all members of an organization"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationMemberModel).where(DBOrganizationMemberModel.organization_id == organization_id)
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            print(f"Failed to get organization members: {e}")
            return []

    async def get_member_role(self, organization_id: int, user_id: str) -> Optional[str]:
        """Get a member's role in an organization"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationMemberModel).where(
                    DBOrganizationMemberModel.organization_id == organization_id,
                    DBOrganizationMemberModel.user_id == user_id
                )
                result = await session.execute(query)
                member = result.scalar_one_or_none()
                return member.role if member else None
        except Exception as e:
            print(f"Failed to get member role: {e}")
            return None

    async def update_member_role(self, organization_id: int, user_id: str, new_role: str) -> bool:
        """Update a member's role in an organization"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationMemberModel).where(
                    DBOrganizationMemberModel.organization_id == organization_id,
                    DBOrganizationMemberModel.user_id == user_id
                )
                result = await session.execute(query)
                member = result.scalar_one_or_none()
                if not member:
                    return False
                member.role = new_role
                await session.commit()
                return True
        except Exception as e:
            print(f"Failed to update member role: {e}")
            return False

    async def remove_organization_member(self, organization_id: int, user_id: str) -> bool:
        """Remove a member from an organization"""
        try:
            async with self.get_session() as session:
                query = select(DBOrganizationMemberModel).where(
                    DBOrganizationMemberModel.organization_id == organization_id,
                    DBOrganizationMemberModel.user_id == user_id
                )
                result = await session.execute(query)
                member = result.scalar_one_or_none()
                if not member:
                    return False
                await session.delete(member)
                await session.commit()
                return True
        except Exception as e:
            print(f"Failed to remove organization member: {e}")
            return False

    # User to Organization Conversion
    async def convert_user_to_organization(self, user_id: str, organization_data: Dict[str, Any]) -> DBOrganizationModel:
        """Convert a user account to an organization"""
        try:
            async with self.get_session() as session:
                # Check if user already has an organization
                existing_org = await self.get_organization_by_owner_id(user_id)
                if existing_org:
                    raise ValueError(f"User {user_id} already owns an organization")

                # Create the organization
                org_data = organization_data.copy()
                org_data['owner_id'] = user_id

                organization = DBOrganizationModel(**org_data)
                session.add(organization)
                await session.commit()
                await session.refresh(organization)

                # Add the owner as the first member with owner role
                owner_member = DBOrganizationMemberModel(
                    organization_id=organization.id,
                    user_id=user_id,
                    role='owner'
                )
                session.add(owner_member)
                await session.commit()

                return organization
        except Exception as e:
            print(f"Failed to convert user to organization: {e}")
            raise


# Global database manager instances
async_db_manager = AsyncDatabaseManager()  # New async version
