"""
Rate limiting utilities for JPMorgan Financial APIs
Provides conditional rate limiting decorators that can be used in blueprints
with configurable thresholds from config
"""

from functools import wraps
import os
from typing import Optional, Any, Union

# Try to import config - handle both cases where config might be available
try:
    from config import config as _config
except ImportError:
    _config = None


class RateLimiterState:
    """Class to hold rate limiter state instead of using global variables"""

    _limiter = None
    _app = None

    @classmethod
    def init(cls, limiter, app):
        """Initialize the rate limiter state with Flask limiter and app instances"""
        cls._limiter = limiter
        cls._app = app

    @classmethod
    def get_limiter(cls):
        """Get the current limiter instance"""
        return cls._limiter

    @classmethod
    def get_app(cls):
        """Get the current app instance"""
        return cls._app


# Default rate limits (can be overridden by config)
DEFAULT_RATE_LIMITS = {
    'default': '100 per minute',
    'auth': '5 per minute',
    'api': '100 per minute',
    'payments': '50 per minute',
    'telemetry': '200 per minute',
    'admin': '20 per minute'
}


def init_rate_limiter(limiter, app):
    """
    Initialize the rate limiter with the Flask limiter and app instances

    Args:
        limiter: Flask-Limiter instance
        app: Flask application instance
    """
    RateLimiterState.init(limiter, app)


def get_rate_limit(limit_type: str = 'default') -> str:
    """
    Get the rate limit for a specific type from configuration or defaults.

    Args:
        limit_type: Type of rate limit ('default', 'auth', 'api', 'payments', 'telemetry', 'admin')

    Returns:
        Rate limit string (e.g., "5 per minute")
    """
    if _config is not None:
        # Get rate limit from config based on type
        config_mapping: dict[str, Optional[Any]] = {
            'default': getattr(_config, 'RATE_LIMIT_DEFAULT', None),
            'auth': getattr(_config, 'RATE_LIMIT_AUTH', None),
            'api': getattr(_config, 'RATE_LIMIT_API', None),
            'payments': getattr(_config, 'RATE_LIMIT_PAYMENTS', None),
            'telemetry': getattr(_config, 'RATE_LIMIT_TELEMETRY', None),
            'admin': getattr(_config, 'RATE_LIMIT_ADMIN', None)
        }

        # Return config value if available and not None
        if limit_type in config_mapping:
            config_value: Optional[Any] = config_mapping.get(limit_type)
            if config_value:
                return str(config_value)

    # Fall back to defaults
    default_value: str = DEFAULT_RATE_LIMITS.get(limit_type, DEFAULT_RATE_LIMITS['default'])
    return default_value


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
            app = RateLimiterState.get_app()
            if app and app.config.get('TESTING'):
                testing = True

            # Skip rate limiting in testing mode
            if testing:
                return f(*args, **kwargs)

            # Apply rate limiting if limiter is available
            limiter = RateLimiterState.get_limiter()
            if limiter:
                return limiter.limit(limit_str)(f)(*args, **kwargs)

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
    return RateLimiterState.get_limiter()


def get_app():
    """
    Get the current app instance

    Returns:
        The Flask app instance or None if not initialized
    """
    return RateLimiterState.get_app()
