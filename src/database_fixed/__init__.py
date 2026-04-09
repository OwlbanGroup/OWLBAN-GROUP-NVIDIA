"""
Database fixed module - direct imports from main file
"""
# Fixed circular imports - direct import

from .database_fixed import (
    db_manager,
    DBBusinessModel,
    DBAssetModel,
    DBOrganizationModel,
    DBOrganizationMemberModel,
    TelemetryEventModel,
    TelemetryMetricsModel,
    AuditLogModel,
    DatabaseManager,
    AsyncDatabaseManager,
    async_db_manager
)
from .db_manager import get_session
from .database_fixed import db_manager as _db_manager_instance

db_manager = _db_manager_instance

# Removed builtins hack to fix module path mismatch in tests
# async_db_manager and AsyncDatabaseManager now imported explicitly in tests

__all__ = [
    'db_manager',
    'DBBusinessModel',
    'DBAssetModel',
    'DBOrganizationModel',
    'DBOrganizationMemberModel',
    'TelemetryEventModel',
    'TelemetryMetricsModel',
    'DatabaseManager',
    'AsyncDatabaseManager',
    'async_db_manager',
    'get_session'
]
