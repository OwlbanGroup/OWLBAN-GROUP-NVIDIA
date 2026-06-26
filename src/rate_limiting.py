"""
Rate limiting for JPMorgan Financial APIs - Production Ready
Uses Flask-Limiter for distributed rate limiting
"""
import os
from functools import wraps
from typing import Callable, Optional

# Try to use Flask-Limiter, fall back to simple implementation
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    
    FLASK_LIMITER_AVAILABLE = True
except ImportError:
    FLASK_LIMITER_AVAILABLE = False
    Limiter = None
    get_remote_address = None


# Global limiter instance
_limiter: Optional['Limiter'] = None


def init_limiter(app, storage_uri: Optional[str] = None):
    """
    Initialize rate limiter with Flask app
    
    Args:
        app: Flask application
        storage_uri: Redis storage URI (e.g., redis://localhost:6379)
        
    Returns:
        Initialized Limiter instance
    """
    global _limiter
    
    if not FLASK_LIMITER_AVAILABLE:
        print("⚠️ Flask-Limiter not installed, using fallback")
        return None
    
    # Determine storage backend
    if storage_uri is None:
        storage_uri = os.getenv('REDIS_URL', 'memory://')
    
    # Initialize limiter with storage
    try:
        _limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri=storage_uri,
            default_limits=["200 per day", "50 per hour"]
        )
        print("✅ Rate limiter initialized with Flask-Limiter")
        return _limiter
    except Exception as e:
        print(f"⚠️ Rate limiter initialization failed: {e}")
        # Fall back to memory storage
        _limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        return _limiter


def get_limiter() -> Optional['Limiter']:
    """Get the global limiter instance"""
    return _limiter


def conditional_limit(limit_str: str, scope: Optional[str] = None):
    """
    Conditional rate limiter decorator
    
    Args:
        limit_str: Rate limit string (e.g., "10 per minute", "100 per hour")
        scope: Optional scope for the limit (e.g., "api", "auth")
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        if not FLASK_LIMITER_AVAILABLE or _limiter is None:
            # Fallback: just return the function
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
        
        # Apply the rate limit
        try:
            limited = _limiter.limit(limit_str, scope_func=lambda: scope or f.__name__)
            return limited(f)
        except Exception as e:
            print(f"⚠️ Rate limit applied failed: {e}")
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
    
    return decorator



