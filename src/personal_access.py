"""
Personal Access Module for JPMorgan Financial APIs
Provides API key management and personal access token functionality.
"""

from functools import wraps
from flask import request, jsonify, g
from typing import Optional, List, Dict, Any
import secrets
import os
from datetime import datetime, timezone, timedelta

# Try to import from existing modules
try:
    from src.logger import telemetry_logger
except ImportError:
    class FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg}")
    telemetry_logger = FallbackLogger()


# =============================================================================
# API KEY STORAGE
# =============================================================================

# In-memory storage (replace with database in production)
_api_keys = {}
_access_tokens = {}


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def generate_api_key(prefix: str = "sk_live_") -> str:
    """
    Generate a new API key with a given prefix.
    """
    return f"{prefix}{secrets.token_hex(32)}"


def create_api_key(
    user_id: str,
    name: str,
    permissions: List[str] = None,
    expires_in_days: int = 365,
    rate_limit: str = "1000 per day"
) -> Dict[str, Any]:
    """
    Create a new API key for a user.
    
    Args:
        user_id: The user's ID
        name: A name/description for the API key
        permissions: List of permissions (e.g., ['read', 'write'])
        expires_in_days: Days until key expires
        rate_limit: Rate limit string
        
    Returns:
        Dict with API key details (including the secret key)
    """
    api_key = generate_api_key()
    key_id = secrets.token_hex(8)
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
    
    key_data = {
        'key_id': key_id,
        'user_id': user_id,
        'name': name,
        'key_prefix': api_key[:12] + "...",
        'permissions': permissions or ['read'],
        'rate_limit': rate_limit,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'expires_at': expires_at.isoformat(),
        'last_used': None,
        'is_active': True
    }
    
    _api_keys[api_key] = key_data
    telemetry_logger.log_info(f"API key created for user {user_id}: {name}")
    
    # Return the full key only once
    return {
        **key_data,
        'api_key': api_key  # Only returned on creation
    }


def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Validate an API key and return its data if valid.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Key data dict if valid, None otherwise
    """
    if api_key not in _api_keys:
        return None
    
    key_data = _api_keys[api_key]
    
    # Check if key is active
    if not key_data.get('is_active', False):
        return None
    
    # Check expiration
    expires_at = datetime.fromisoformat(key_data['expires_at'])
    if expires_at < datetime.now(timezone.utc):
        return None
    
    # Update last used
    key_data['last_used'] = datetime.now(timezone.utc).isoformat()
    
    return key_data


def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key.
    
    Args:
        api_key: The API key to revoke
        
    Returns:
        True if revoked, False if not found
    """
    if api_key in _api_keys:
        _api_keys[api_key]['is_active'] = False
        telemetry_logger.log_info(f"API key revoked: {_api_keys[api_key].get('key_id')}")
        return True
    return False


def list_api_keys(user_id: str) -> List[Dict[str, Any]]:
    """
    List all API keys for a user (without showing the full key).
    """
    keys = []
    for key_data in _api_keys.values():
        if key_data.get('user_id') == user_id:
            # Remove the actual key from the response
            key_copy = {k: v for k, v in key_data.items() if k != 'api_key'}
            keys.append(key_copy)
    return keys


# =============================================================================
# PERSONAL ACCESS TOKENS
# =============================================================================

def generate_access_token() -> str:
    """
    Generate a new personal access token.
    """
    return secrets.token_hex(40)


def create_access_token(
    user_id: str,
    name: str,
    scopes: List[str] = None,
    expires_in_days: int = 30
) -> Dict[str, Any]:
    """
    Create a new personal access token.
    """
    token = generate_access_token()
    token_id = secrets.token_hex(8)
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
    
    token_data = {
        'token_id': token_id,
        'user_id': user_id,
        'name': name,
        'scopes': scopes or ['read'],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'expires_at': expires_at.isoformat(),
        'last_used': None,
        'is_active': True
    }
    
    _access_tokens[token] = token_data
    telemetry_logger.log_info(f"Access token created for user {user_id}: {name}")
    
    return {
        **token_data,
        'token': token  # Only returned on creation
    }


def validate_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a personal access token.
    """
    if token not in _access_tokens:
        return None
    
    token_data = _access_tokens[token]
    
    if not token_data.get('is_active', False):
        return None
    
    expires_at = datetime.fromisoformat(token_data['expires_at'])
    if expires_at < datetime.now(timezone.utc):
        return None
    
    token_data['last_used'] = datetime.now(timezone.utc).isoformat()
    
    return token_data


def revoke_access_token(token: str) -> bool:
    """
    Revoke a personal access token.
    """
    if token in _access_tokens:
        _access_tokens[token]['is_active'] = False
        telemetry_logger.log_info(f"Access token revoked: {_access_tokens[token].get('token_id')}")
        return True
    return False


def list_access_tokens(user_id: str) -> List[Dict[str, Any]]:
    """
    List all access tokens for a user (without showing the token).
    """
    tokens = []
    for token_data in _access_tokens.values():
        if token_data.get('user_id') == user_id:
            token_copy = {k: v for k, v in token_data.items() if k != 'token'}
            tokens.append(token_copy)
    return tokens


# =============================================================================
# DECORATORS
# =============================================================================

def api_key_required(f):
    """
    Decorator to require valid API key for endpoint access.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip in testing mode
        if os.environ.get('TESTING') == '1':
            g.user_id = 'test_user'
            g.api_key_data = {'permissions': ['read', 'write']}
            return f(*args, **kwargs)
        
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'Missing API key', 'status': 'error'}), 401
        
        key_data = validate_api_key(api_key)
        if not key_data:
            return jsonify({'error': 'Invalid or expired API key', 'status': 'error'}), 401
        
        g.user_id = key_data.get('user_id')
        g.api_key_data = key_data
        
        return f(*args, **kwargs)
    
    return decorated_function


def scope_required(required_scopes: List[str]):
    """
    Decorator factory to require specific scopes for endpoint access.
    
    Usage:
        @scope_required(['read', 'write'])
        def my_endpoint():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get scopes from either API key or access token
            scopes = []
            
            if hasattr(g, 'api_key_data'):
                scopes = g.api_key_data.get('permissions', [])
            elif hasattr(g, 'access_token_data'):
                scopes = g.access_token_data.get('scopes', [])
            
            # Check if user has all required scopes
            missing_scopes = [s for s in required_scopes if s not in scopes]
            if missing_scopes:
                return jsonify({
                    'error': f'Missing required scopes: {", ".join(missing_scopes)}',
                    'status': 'error'
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def access_token_required(f):
    """
    Decorator to require valid access token for endpoint access.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip in testing mode
        if os.environ.get('TESTING') == '1':
            g.user_id = 'test_user'
            g.access_token_data = {'scopes': ['read', 'write']}
            return f(*args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header', 'status': 'error'}), 401
        
        token = auth_header.split(' ')[1]
        token_data = validate_access_token(token)
        
        if not token_data:
            return jsonify({'error': 'Invalid or expired access token', 'status': 'error'}), 401
        
        g.user_id = token_data.get('user_id')
        g.access_token_data = token_data
        
        return f(*args, **kwargs)
    
    return decorated_function


# =============================================================================
# USER PREFERENCES
# =============================================================================

# Mock user preferences storage
_user_preferences = {}


def get_user_preferences(user_id: str) -> Dict[str, Any]:
    """
    Get user preferences.
    """
    return _user_preferences.get(user_id, {
        'user_id': user_id,
        'dashboard_layout': 'default',
        'notifications': {
            'email': True,
            'sms': False,
            'push': True
        },
        'currency': 'USD',
        'timezone': 'UTC',
        'theme': 'light'
    })


def update_user_preferences(user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update user preferences.
    """
    current = get_user_preferences(user_id)
    current.update(preferences)
    _user_preferences[user_id] = current
    
    telemetry_logger.log_info(f"User preferences updated for {user_id}")
    
    return current


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'generate_api_key',
    'create_api_key',
    'validate_api_key',
    'revoke_api_key',
    'list_api_keys',
    'generate_access_token',
    'create_access_token',
    'validate_access_token',
    'revoke_access_token',
    'list_access_tokens',
    'api_key_required',
    'access_token_required',
    'scope_required',
    'get_user_preferences',
    'update_user_preferences'
]
