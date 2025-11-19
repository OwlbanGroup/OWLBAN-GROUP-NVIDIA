"""
Enhanced error handling and logging for the API
"""
import traceback
import json
from typing import Dict, Any, Optional, Union
from flask import request, jsonify, current_app
from werkzeug.exceptions import HTTPException
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from redis.exceptions import RedisError
import structlog
from datetime import datetime, timezone

from .logger import telemetry_logger
from .schemas import ErrorResponse


class APIError(Exception):
    """Base exception for API errors"""

    def __init__(self, message: str, status_code: int = 500, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload


class ValidationAPIError(APIError):
    """Exception for validation errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, payload=details)


class AuthenticationAPIError(APIError):
    """Exception for authentication errors"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401)


class AuthorizationAPIError(APIError):
    """Exception for authorization errors"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)


class NotFoundAPIError(APIError):
    """Exception for resource not found errors"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class RateLimitAPIError(APIError):
    """Exception for rate limiting errors"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class ErrorHandler:
    """Centralized error handling and logging"""

    @staticmethod
    def log_error(error: Exception, context: Optional[Dict[str, Any]] = None,
                    level: str = "error") -> None:
        """Log error with structured logging"""
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'url': request.url if request else None,
            'method': request.method if request else None,
            'user_agent': request.headers.get('User-Agent') if request else None,
            'remote_addr': request.remote_addr if request else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        if context:
            error_context.update(context)

        # Use structlog for structured logging
        logger = structlog.get_logger()
        log_method = getattr(logger, level, logger.error)
        log_method("API Error", **error_context)

        # Also log to telemetry logger for backward compatibility
        telemetry_logger.log_error(error, error_context)

    @staticmethod
    def handle_api_error(error: APIError) -> tuple:
        """Handle custom API errors"""
        response = ErrorResponse(
            error=error.message,
            details=error.payload
        )

        ErrorHandler.log_error(error, {
            'status_code': error.status_code,
            'endpoint': request.endpoint if request else None
        })

        return jsonify(response.dict()), error.status_code

    @staticmethod
    def handle_validation_error(error: Union[PydanticValidationError, Exception]) -> tuple:
        """Handle validation errors"""
        if isinstance(error, PydanticValidationError):
            details = {
                'validation_errors': [
                    {
                        'field': '.'.join(str(loc) for loc in err['loc']),
                        'message': err['msg'],
                        'type': err['type']
                    }
                    for err in error.errors()
                ]
            }
            message = "Validation failed"
        else:
            details = {'original_error': str(error)}
            message = "Input validation error"

        api_error = ValidationAPIError(message, details)
        return ErrorHandler.handle_api_error(api_error)

    @staticmethod
    def handle_sqlalchemy_error(error: SQLAlchemyError) -> tuple:
        """Handle database errors"""
        if isinstance(error, OperationalError):
            message = "Database connection error"
            status_code = 503
        elif isinstance(error, IntegrityError):
            message = "Data integrity violation"
            status_code = 409
        else:
            message = "Database error"
            status_code = 500

        api_error = APIError(message, status_code, {'db_error': str(error)})
        return ErrorHandler.handle_api_error(api_error)

    @staticmethod
    def handle_redis_error(error: RedisError) -> tuple:
        """Handle Redis/cache errors"""
        message = "Cache service unavailable"
        api_error = APIError(message, 503, {'cache_error': str(error)})
        return ErrorHandler.handle_api_error(api_error)

    @staticmethod
    def handle_http_error(error: HTTPException) -> tuple:
        """Handle HTTP exceptions"""
        response = ErrorResponse(
            error=error.description or "HTTP error",
            details={'code': error.code}
        )

        ErrorHandler.log_error(error, {
            'status_code': error.code,
            'http_error': True
        })

        return jsonify(response.dict()), error.code

    @staticmethod
    def handle_generic_error(error: Exception) -> tuple:
        """Handle unexpected errors"""
        # Don't expose internal error details in production
        if current_app.config.get('ENV') == 'production':
            message = "Internal server error"
            details = None
        else:
            message = str(error)
            details = {'traceback': traceback.format_exc()}

        api_error = APIError(message, 500, details)
        return ErrorHandler.handle_api_error(api_error)


def register_error_handlers(app):
    """Register error handlers with Flask app"""

    @app.errorhandler(APIError)
    def handle_api_error(error):
        return ErrorHandler.handle_api_error(error)

    @app.errorhandler(PydanticValidationError)
    def handle_validation_error(error):
        return ErrorHandler.handle_validation_error(error)

    @app.errorhandler(SQLAlchemyError)
    def handle_sqlalchemy_error(error):
        return ErrorHandler.handle_sqlalchemy_error(error)

    @app.errorhandler(RedisError)
    def handle_redis_error(error):
        return ErrorHandler.handle_redis_error(error)

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        return ErrorHandler.handle_http_error(error)

    @app.errorhandler(Exception)
    def handle_generic_error(error):
        return ErrorHandler.handle_generic_error(error)


# Utility functions for error handling in routes
def handle_route_errors(func):
    """Decorator to handle common route errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationAPIError as e:
            raise e
        except AuthenticationAPIError as e:
            raise e
        except AuthorizationAPIError as e:
            raise e
        except NotFoundAPIError as e:
            raise e
        except RateLimitAPIError as e:
            raise e
        except Exception as e:
            ErrorHandler.log_error(e, {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            })
            raise APIError("An unexpected error occurred", 500)
    return wrapper
