"""
JWT Authentication and RBAC Service for JPMorgan Financial APIs
"""
import jwt
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from functools import wraps
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

from src.logger import telemetry_logger
from src.database_fixed import db_manager
from src.models.user import User, UserRole, RolePermission


class AuthService:
    """JWT Authentication and Role-Based Access Control Service"""

    def __init__(self, secret_key: str = None, token_expiry_hours: int = 24):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry_hours = token_expiry_hours
        self.logger = telemetry_logger.get_logger()

        # Define role permissions
        self.role_permissions = {
            UserRole.ADMIN.value: [
                'read:users', 'write:users', 'delete:users',
                'read:businesses', 'write:businesses', 'delete:businesses',
                'read:assets', 'write:assets', 'delete:assets',
                'read:telemetry', 'write:telemetry',
                'read:audit', 'write:audit',
                'read:revenue', 'write:revenue',
                'read:private_bank', 'write:private_bank',
                'read:ml', 'write:ml',
                'read:metrics', 'write:metrics'
            ],
            UserRole.MANAGER.value: [
                'read:businesses', 'write:businesses',
                'read:assets', 'write:assets',
                'read:telemetry', 'write:telemetry',
                'read:audit',
                'read:revenue', 'write:revenue',
                'read:private_bank',
                'read:metrics'
            ],
            UserRole.USER.value: [
                'read:businesses',
                'read:assets',
                'read:telemetry',
                'read:revenue',
                'read:private_bank',
                'read:metrics'
            ],
            UserRole.AUDITOR.value: [
                'read:audit',
                'read:metrics'
            ]
        }

    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return generate_password_hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return check_password_hash(hashed, password)

    def create_user(self, username: str, password: str, email: str = None,
                   role: UserRole = UserRole.USER, business_id: int = None) -> User:
        """Create a new user"""
        try:
            hashed_password = self.hash_password(password)

            user = User(
                username=username,
                password_hash=hashed_password,
                email=email,
                role=role,
                business_id=business_id,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            with db_manager.get_session() as session:
                session.add(user)
                session.commit()
                session.refresh(user)

            self.logger.info(f"Created user: {username} with role: {role.value}")
            return user

        except Exception as e:
            self.logger.error(f"Failed to create user {username}: {e}")
            raise

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(username=username, is_active=True).first()

                if user and self.verify_password(password, user.password_hash):
                    # Update last login
                    user.last_login_at = datetime.now(timezone.utc)
                    user.updated_at = datetime.now(timezone.utc)
                    session.commit()

                    self.logger.info(f"User authenticated: {username}")
                    return user

                self.logger.warning(f"Failed authentication attempt for user: {username}")
                return None

        except Exception as e:
            self.logger.error(f"Authentication error for {username}: {e}")
            return None

    def generate_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'business_id': user.business_id,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(hours=self.token_expiry_hours)
        }

        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        self.logger.info(f"Generated token for user: {user.username}")
        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])

            # Check if token is expired
            exp = datetime.fromtimestamp(payload['exp'], timezone.utc)
            if exp < datetime.now(timezone.utc):
                self.logger.warning("Token expired")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token"""
        payload = self.verify_token(token)
        if not payload:
            return None

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(
                    id=payload['user_id'],
                    is_active=True
                ).first()
                return user
        except Exception as e:
            self.logger.error(f"Error getting user from token: {e}")
            return None

    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has a specific permission"""
        if not user or not user.is_active:
            return False

        user_permissions = self.role_permissions.get(user.role, [])
        return permission in user_permissions

    def require_auth(self, permissions: List[str] = None):
        """Decorator for JWT authentication with optional RBAC"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Skip auth in testing mode
                if hasattr(request, 'app') and request.app.config.get('TESTING'):
                    return f(*args, **kwargs)

                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({
                        'error': 'Missing or invalid authorization header',
                        'status': 'error'
                    }), 401

                token = auth_header.split(' ')[1]
                user = self.get_user_from_token(token)

                if not user:
                    return jsonify({
                        'error': 'Invalid or expired token',
                        'status': 'error'
                    }), 401

                # Check permissions if specified
                if permissions:
                    has_required_permissions = any(
                        self.has_permission(user, perm) for perm in permissions
                    )
                    if not has_required_permissions:
                        return jsonify({
                            'error': 'Insufficient permissions',
                            'status': 'error'
                        }), 403

                # Store user in Flask g object for use in route
                g.user = user
                g.user_permissions = self.role_permissions.get(user.role, [])

                return f(*args, **kwargs)
            return decorated_function
        return decorator

    def get_current_user(self) -> Optional[User]:
        """Get current user from Flask g object"""
        return getattr(g, 'user', None)

    def get_current_user_permissions(self) -> List[str]:
        """Get current user's permissions"""
        return getattr(g, 'user_permissions', [])

    def update_user_role(self, user_id: int, new_role: UserRole) -> bool:
        """Update a user's role"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    return False

                user.role = new_role
                user.updated_at = datetime.now(timezone.utc)
                session.commit()

                self.logger.info(f"Updated role for user {user.username} to {new_role.value}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to update user role: {e}")
            return False

    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    return False

                user.is_active = False
                user.updated_at = datetime.now(timezone.utc)
                session.commit()

                self.logger.info(f"Deactivated user: {user.username}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to deactivate user: {e}")
            return False


# Global auth service instance
auth_service = AuthService()
