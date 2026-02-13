"""
Authentication and Authorization Module for JPMorgan Financial APIs
Provides decorators for token authentication, role-based access control, and rate limiting.
"""

from functools import wraps
from flask import request, jsonify, g
from typing import Optional, List, Callable
import os
import secrets

# Try to import from existing modules
try:
    from src.logger import telemetry_logger
except ImportError:
    # Fallback if logger not available
    class FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg}")
    telemetry_logger = FallbackLogger()


# =============================================================================
# TOKEN AUTHENTICATION
# =============================================================================

def token_auth_required(f: Callable) -> Callable:
    """
    Decorator to require valid authentication token for endpoint access.
    Checks for Bearer token in Authorization header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication in testing mode
        testing = os.environ.get('TESTING') == '1'
        if testing:
            # Set a default user_id for testing
            g.user_id = 'test_user'
            g.user_role = 'admin'
            return f(*args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            telemetry_logger.log_info("Missing authorization header", {'context': 'token_auth'})
            return jsonify({'error': 'Missing authorization header', 'status': 'error'}), 401
        
        if not auth_header.startswith('Bearer '):
            telemetry_logger.log_info("Invalid authorization header format", {'context': 'token_auth'})
            return jsonify({'error': 'Invalid authorization header format. Use: Bearer <token>', 'status': 'error'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Validate token against stored tokens
        user_data = validate_token(token)
        if not user_data:
            telemetry_logger.log_info("Invalid or expired token", {'context': 'token_auth'})
            return jsonify({'error': 'Invalid or expired token', 'status': 'error'}), 401
        
        # Set user info in Flask's g object
        g.user_id = user_data.get('user_id')
        g.user_role = user_data.get('role', 'user')
        g.user_data = user_data
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_auth(f: Callable) -> Callable:
    """
    Alternative decorator for requiring authentication.
    Similar to token_auth_required but with slightly different error handling.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication in testing mode
        testing = os.environ.get('TESTING') == '1'
        if testing:
            g.user_id = 'test_user'
            g.user_role = 'admin'
            return f(*args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header', 'status': 'error'}), 401
        
        token = auth_header.split(' ')[1]
        user_data = validate_token(token)
        
        if not user_data:
            return jsonify({'error': 'Invalid or expired token', 'status': 'error'}), 401
        
        g.user_id = user_data.get('user_id')
        g.user_role = user_data.get('role', 'user')
        g.user_data = user_data
        
        return f(*args, **kwargs)
    
    return decorated_function


# =============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# =============================================================================

def role_required(allowed_roles: List[str]):
    """
    Decorator factory to require specific roles for endpoint access.
    
    Usage:
        @role_required(['admin', 'manager'])
        def my_endpoint():
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip in testing mode
            testing = os.environ.get('TESTING') == '1'
            if testing:
                return f(*args, **kwargs)
            
            # Get user role from g object (set by token_auth_required)
            user_role = getattr(g, 'user_role', None)
            
            if not user_role:
                return jsonify({'error': 'Authentication required', 'status': 'error'}), 401
            
            if user_role not in allowed_roles:
                telemetry_logger.log_info(
                    f"Access denied: user role '{user_role}' not in allowed roles {allowed_roles}",
                    {'context': 'role_required', 'user_role': user_role}
                )
                return jsonify({
                    'error': 'Insufficient permissions. Required roles: ' + ', '.join(allowed_roles),
                    'status': 'error'
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def admin_required(f: Callable) -> Callable:
    """
    Decorator to require admin role for endpoint access.
    """
    return role_required(['admin'])(f)


def manager_required(f: Callable) -> Callable:
    """
    Decorator to require admin or manager role for endpoint access.
    """
    return role_required(['admin', 'manager'])(f)


# =============================================================================
# TOKEN MANAGEMENT
# =============================================================================

# In-memory token storage (replace with database in production)
_valid_tokens = {}


def validate_token(token: str) -> Optional[dict]:
    """
    Validate a Bearer token and return user data if valid.
    
    Args:
        token: The Bearer token to validate
        
    Returns:
        User data dict if token is valid, None otherwise
    """
    # Check in-memory store
    if token in _valid_tokens:
        user_data = _valid_tokens[token]
        # Check expiration
        if 'expires_at' in user_data:
            from datetime import datetime, timezone
            if user_data['expires_at'] < datetime.now(timezone.utc):
                # Token expired
                del _valid_tokens[token]
                return None
        return user_data
    
    # Check environment for test tokens
    test_token = os.environ.get('TEST_TOKEN')
    if test_token and token == test_token:
        return {
            'user_id': 'test_user',
            'role': 'admin',
            'email': 'test@example.com'
        }
    
    return None


def create_token(user_id: str, role: str = 'user', expires_in_hours: int = 24) -> str:
    """
    Create a new authentication token for a user.
    
    Args:
        user_id: The user's ID
        role: The user's role (default: 'user')
        expires_in_hours: Token expiration time in hours
        
    Returns:
        The generated token
    """
    from datetime import datetime, timezone, timedelta
    
    token = secrets.token_hex(32)
    
    _valid_tokens[token] = {
        'user_id': user_id,
        'role': role,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'expires_at': datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    }
    
    telemetry_logger.log_info(f"Token created for user: {user_id}", {'context': 'create_token'})
    
    return token


def revoke_token(token: str) -> bool:
    """
    Revoke a token.
    
    Args:
        token: The token to revoke
        
    Returns:
        True if token was revoked, False if not found
    """
    if token in _valid_tokens:
        del _valid_tokens[token]
        telemetry_logger.log_info("Token revoked", {'context': 'revoke_token'})
        return True
    return False


# =============================================================================
# USER ROLES AND PERMISSIONS
# =============================================================================

# Define roles and their permissions
ROLE_PERMISSIONS = {
    'admin': [
        'read_all', 'write_all', 'delete_all',
        'manage_users', 'manage_roles', 'view_analytics',
        'access_settings', 'export_data'
    ],
    'manager': [
        'read_all', 'write_own', 'read_employees', 'write_employees',
        'view_analytics', 'access_reports'
    ],
    'employee': [
        'read_own', 'write_own', 'read_payroll', 'write_timesheet'
    ],
    'customer': [
        'read_own', 'write_own', 'read_account', 'make_payment'
    ],
    'user': [
        'read_own', 'write_own'
    ]
}


def check_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.
    
    Args:
        role: The user's role
        permission: The permission to check
        
    Returns:
        True if role has permission, False otherwise
    """
    permissions = ROLE_PERMISSIONS.get(role, [])
    return permission in permissions


def get_user_permissions(role: str) -> List[str]:
    """
    Get all permissions for a role.
    
    Args:
        role: The user's role
        
    Returns:
        List of permission strings
    """
    return ROLE_PERMISSIONS.get(role, [])


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

def api_key_required(f: Callable) -> Callable:
    """
    Decorator to require valid API key for endpoint access.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'Missing API key', 'status': 'error'}), 401
        
        # Validate API key (in production, check against database)
        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key', 'status': 'error'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    # In production, check against database
    valid_keys = os.environ.get('API_KEYS', '').split(',')
    return api_key in valid_keys


# =============================================================================
# CURRENT USER HELPER
# =============================================================================

def get_current_user_id() -> str:
    """Get the current authenticated user's ID from Flask g object."""
    return getattr(g, 'user_id', None)


def get_current_user_role() -> str:
    """Get the current authenticated user's role from Flask g object."""
    return getattr(g, 'user_role', 'guest')


def get_current_user_data() -> dict:
    """Get the current authenticated user's full data from Flask g object."""
    return getattr(g, 'user_data', {})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'token_auth_required',
    'require_auth',
    'role_required',
    'admin_required',
    'manager_required',
    'api_key_required',
    'validate_token',
    'create_token',
    'revoke_token',
    'check_permission',
    'get_user_permissions',
    'ROLE_PERMISSIONS',
    'get_current_user_id',
    'get_current_user_role',
    'get_current_user_data'
]
