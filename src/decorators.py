"""
Decorators for authentication, rate limiting, and other middleware in JPMorgan Financial APIs
"""

from functools import wraps
from flask import request, jsonify, g
from flask_limiter import Limiter
from typing import Callable

def token_auth_required(f: Callable) -> Callable:
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
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'TESTING' in os.environ:
                return f(*args, **kwargs)
            # Apply limiter - assumes limiter is configured globally
            return f(*args, **kwargs)  # Mock - real limiter needs app context
        return decorated_function
    return decorator

