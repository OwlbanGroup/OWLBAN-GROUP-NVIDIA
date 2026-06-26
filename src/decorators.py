"""
Decorators for authentication, rate limiting, and other middleware in JPMorgan Financial APIs
"""

import os
from functools import wraps
from flask import request, jsonify, g
from typing import Callable

# Import rate limiting module
try:
    from src.rate_limiting import (
        init_limiter,
        get_limiter,
        conditional_limit,
        auth_limit,
        api_limit,
        payment_limit,
        FLASK_LIMITER_AVAILABLE
    )
except ImportError:
    # Fallback if module not available
    init_limiter = None
    get_limiter = None
    conditional_limit = lambda limit_str: lambda f: f
    auth_limit = lambda f: f
    api_limit = lambda f: f
    payment_limit = lambda f: f
    FLASK_LIMITER_AVAILABLE = False


def token_auth_required(f: Callable) -> Callable:
    """Decorator requiring valid Bearer token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        token = auth_header.split(' ')[1]
        # Simple token validation (replace with JWT in production)
        # For now, accept any token starting with 'test_' or 'david_'
        if not (token.startswith('test_') or token.startswith('david_')):
            return jsonify({'error': 'Invalid or expired token'}), 401
        g.user_id = 'test_user'  # Mock user ID
        return f(*args, **kwargs)
    return decorated_function


def conditional_limit(limit_str: str):
    """
    Conditional rate limiter decorator - disabled in testing mode
    
    Args:
        limit_str: Rate limit string (e.g., "10 per minute")
    """
    def decorator(f: Callable) -> Callable:
        # Skip rate limiting in testing mode
        if os.environ.get('TESTING') or os.environ.get('FLASK_TESTING'):
            return f
        
        # Use the rate limiting module
        if FLASK_LIMITER_AVAILABLE:
            return conditional_limit(limit_str)(f)
        else:
            # Fallback without rate limiting
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
    return decorator


def auth_limit():
    """Rate limit decorator for authentication endpoints (5/min)"""
    def decorator(f: Callable) -> Callable:
        if os.environ.get('TESTING') or os.environ.get('FLASK_TESTING'):
            return f
        if FLASK_LIMITER_AVAILABLE:
            return auth_limit(f)
        else:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
    return decorator


def api_limit():
    """Rate limit decorator for general API endpoints (60/min)"""
    def decorator(f: Callable) -> Callable:
        if os.environ.get('TESTING') or os.environ.get('FLASK_TESTING'):
            return f
        if FLASK_LIMITER_AVAILABLE:
            return api_limit(f)
        else:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
    return decorator


def payment_limit():
    """Rate limit decorator for payment endpoints (stricter: 10/min)"""
    def decorator(f: Callable) -> Callable:
        if os.environ.get('TESTING') or os.environ.get('FLASK_TESTING'):
            return f
        if FLASK_LIMITER_AVAILABLE:
            return payment_limit(f)
        else:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
    return decorator


def init_rate_limiting(app):
    """Initialize rate limiting with Flask app"""
    return init_limiter(app)

