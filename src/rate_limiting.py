"""
Rate limiting utilities for JPMorgan Financial APIs
Provides conditional rate limiting decorators that can be used in blueprints
"""

from functools import wraps
import os

# Global limiter instance (will be set when app is initialized)
_limiter = None
_app = None

def init_rate_limiter(limiter, app):
    """
    Initialize the rate limiter with the Flask limiter and app instances
    
    Args:
        limiter: Flask-Limiter instance
        app: Flask application instance
    """
    global _limiter, _app
    _limiter = limiter
    _app = app

def conditional_limit(limit_str):
    """
    Conditional rate limiter decorator that can be used in blueprints.
    Skips rate limiting in testing mode.
    
    Args:
        limit_str: Rate limit string (e.g., "5 per minute", "10 per hour")
    
    Returns:
        Decorator function that applies rate limiting unless in testing mode
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if we're in testing mode
            testing = os.environ.get('TESTING') == '1'
            
            # Also check app config if app is available
            if _app and _app.config.get('TESTING'):
                testing = True
            
            # Skip rate limiting in testing mode
            if testing:
                return f(*args, **kwargs)
            
            # Apply rate limiting if limiter is available
            if _limiter:
                return _limiter.limit(limit_str)(f)(*args, **kwargs)
            
            # If no limiter is available, still run the function
            return f(*args, **kwargs)
        
        # Copy over the original function's attributes for Flask-Limiter compatibility
        decorated_function.__name__ = f.__name__
        decorated_function.__doc__ = f.__doc__
        
        return decorated_function
    return decorator

def get_limiter():
    """
    Get the current limiter instance
    
    Returns:
        The Flask-Limiter instance or None if not initialized
    """
    return _limiter

def get_app():
    """
    Get the current app instance
    
    Returns:
        The Flask app instance or None if not initialized
    """
    return _app
