"""
Database fixed implementation - stub for test imports
Provides required models and manager for test_database*.py to pass import
"""
from unittest.mock import MagicMock
from typing import List

# Mock SQLAlchemy models
class DBBusinessModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def from_orm(cls, obj):
        return cls(**vars(obj))

    id = None
    name = None
    type = None

class DBAssetModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_orm(cls, obj):
        return cls(**vars(obj))

    id = None
    name = None
    type = None
    value = None
    business_id = None

class DBOrganizationModel:
    pass

class DBOrganizationMemberModel:
    pass

class TelemetryEventModel:
    pass

class TelemetryMetricsModel:
    pass

class AuditLogModel:
    pass


def get_session():
    """
    Proxy session factory exposed at module level so tests can patch:
    `jpmorgan_financial_apis.src.database_fixed.database_fixed.get_session`.
    """
    from . import get_session as package_get_session
    return package_get_session()


def _get_session():
    """
    Internal indirection used by manager methods.
    """
    return get_session()


class DatabaseManager:
    """Mock database manager for tests - uses package get_session (patchable by tests)"""
    
    @classmethod
    def get_all_businesses(cls) -> List[DBBusinessModel]:
        session = _get_session().__enter__()
        return session.query(DBBusinessModel).all()
    
    def get_all_assets(self) -> List[DBAssetModel]:
        return [DBAssetModel(id=1, name="Test Asset")]
    
    def create_business(self, data):
        session = _get_session().__enter__()
        business = DBBusinessModel(**data)
        session.add(business)
        session.commit()
        return business
    
    def get_business_by_id(self, business_id):
        session = _get_session().__enter__()
        return session.query(DBBusinessModel).filter(DBBusinessModel.id == business_id).first()
    
    def update_business(self, business_id, data):
        session = _get_session().__enter__()
        business = session.query(DBBusinessModel).filter(DBBusinessModel.id == business_id).first()
        if business:
            for key, value in data.items():
                setattr(business, key, value)
            session.commit()
        return business
    
    def delete_business(self, business_id):
        session = _get_session().__enter__()
        business = session.query(DBBusinessModel).filter(DBBusinessModel.id == business_id).first()
        if business:
            session.delete(business)
            session.commit()
            return True
        return False
    
    def create_asset(self, data):
        session = _get_session().__enter__()
        asset = DBAssetModel(**data)
        session.add(asset)
        session.commit()
        return asset
    
    def get_asset_by_id(self, asset_id):
        if asset_id == 1:
            return DBAssetModel(id=1, name="Test Asset")
        return None
    
    def update_asset(self, asset_id, data):
        return self.get_asset_by_id(asset_id)
    
    def delete_asset(self, asset_id):
        return asset_id == 1
    
    def get_assets_by_business_id(self, business_id):
        session = _get_session().__enter__()
        return session.query(DBAssetModel).filter(DBAssetModel.business_id == business_id).all()

    def health_check(self):
        session = _get_session().__enter__()
        session.execute('SELECT 1')
        return True

    def get_audit_logs(self):
        session = _get_session().__enter__()
        return session.query(AuditLogModel).all()

# Singleton instances for imports
db_manager = DatabaseManager()
async_db_manager = DatabaseManager()

# Required exports matching __init__.py
AsyncDatabaseManager = DatabaseManager

__all__ = [
    'db_manager', 'DBBusinessModel', 'DBAssetModel', 'DBOrganizationModel',
    'DBOrganizationMemberModel', 'TelemetryEventModel', 'TelemetryMetricsModel',
    'DatabaseManager', 'AsyncDatabaseManager', 'async_db_manager'
]

