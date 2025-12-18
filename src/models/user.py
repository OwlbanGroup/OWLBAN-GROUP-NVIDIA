"""
User Model for Database-Backed Authentication with RBAC
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from .base import Base
import json
from enum import Enum


class UserRole(str, Enum):
    """User role enumeration"""
    USER = "user"
    AUDITOR = "auditor"
    MANAGER = "manager"
    ADMIN = "admin"


class RolePermission(str, Enum):
    """Role permission enumeration"""
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    DELETE_USERS = "delete:users"
    READ_BUSINESSES = "read:businesses"
    WRITE_BUSINESSES = "write:businesses"
    DELETE_BUSINESSES = "delete:businesses"
    READ_ASSETS = "read:assets"
    WRITE_ASSETS = "write:assets"
    DELETE_ASSETS = "delete:assets"
    READ_TELEMETRY = "read:telemetry"
    WRITE_TELEMETRY = "write:telemetry"
    READ_AUDIT = "read:audit"
    WRITE_AUDIT = "write:audit"
    READ_REVENUE = "read:revenue"
    WRITE_REVENUE = "write:revenue"
    READ_PRIVATE_BANK = "read:private_bank"
    WRITE_PRIVATE_BANK = "write:private_bank"
    READ_ML = "read:ml"
    WRITE_ML = "write:ml"
    READ_METRICS = "read:metrics"
    WRITE_METRICS = "write:metrics"

class User(Base):
    """User model for authentication with role-based access control"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default='user')  # admin, manager, user, auditor
    permissions = Column(Text, nullable=True)  # JSON string of permissions
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    token = Column(String(255), nullable=True, index=True)  # Legacy token for backward compatibility
    token_created_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                       onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'permissions': self.get_permissions(),
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'token_created_at': self.token_created_at.isoformat() if self.token_created_at else None
        }

    def get_permissions(self):
        """Get permissions as a list"""
        if self.permissions:
            try:
                return json.loads(self.permissions)
            except json.JSONDecodeError:
                return []
        return []

    def set_permissions(self, permissions_list):
        """Set permissions from a list"""
        self.permissions = json.dumps(permissions_list) if permissions_list else None

    def has_permission(self, permission):
        """Check if user has a specific permission"""
        permissions = self.get_permissions()
        return permission in permissions

    def has_role(self, required_role):
        """Check if user has required role or higher"""
        role_hierarchy = {
            'user': 1,
            'auditor': 2,
            'manager': 3,
            'admin': 4
        }

        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level

    @staticmethod
    def get_default_permissions_for_role(role):
        """Get default permissions for a role"""
        role_permissions = {
            'user': [
                'read:own_profile',
                'read:businesses',
                'read:assets',
                'read:telemetry',
                'read:metrics'
            ],
            'auditor': [
                'read:own_profile',
                'read:businesses',
                'read:assets',
                'read:telemetry',
                'read:metrics',
                'read:audit_logs',
                'read:compliance_reports'
            ],
            'manager': [
                'read:own_profile',
                'read:businesses',
                'read:assets',
                'read:telemetry',
                'read:metrics',
                'read:audit_logs',
                'read:compliance_reports',
                'write:businesses',
                'write:assets',
                'manage:users'
            ],
            'admin': [
                'read:own_profile',
                'read:businesses',
                'read:assets',
                'read:telemetry',
                'read:metrics',
                'read:audit_logs',
                'read:compliance_reports',
                'write:businesses',
                'write:assets',
                'manage:users',
                'manage:system',
                'delete:businesses',
                'delete:assets'
            ]
        }
        return role_permissions.get(role, [])
