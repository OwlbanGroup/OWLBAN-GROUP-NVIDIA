"""
Structured JSON Logger
Provides consistent, structured logging throughout the application
"""
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import traceback

class StructuredLogger:
    """Structured JSON logger for consistent logging"""

    def __init__(self, name: str = 'jpmorgan_api', level: str = 'INFO'):
        """
        Initialize structured logger

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Create console handler with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)

    def _create_log_entry(self, level: str, message: str,
                        context: Optional[Dict[str, Any]] = None,
                        error: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Create structured log entry

        Args:
            level: Log level
            message: Log message
            context: Additional context
            error: Exception object if applicable

        Returns:
            dict: Structured log entry
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message,
            'logger': self.logger.name
        }

        if context:
            entry['context'] = context

        if error:
            entry['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }

        return entry

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        entry = self._create_log_entry('DEBUG', message, context)
        self.logger.debug(json.dumps(entry))

    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message"""
        entry = self._create_log_entry('INFO', message, context)
        self.logger.info(json.dumps(entry))

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        entry = self._create_log_entry('WARNING', message, context)
        self.logger.warning(json.dumps(entry))

    def error(self, message: str, context: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None):
        """Log error message"""
        entry = self._create_log_entry('ERROR', message, context, error)
        self.logger.error(json.dumps(entry))

    def critical(self, message: str, context: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None):
        """Log critical message"""
        entry = self._create_log_entry('CRITICAL', message, context, error)
        self.logger.critical(json.dumps(entry))

    def log_request(self, method: str, path: str, status_code: int,
                    duration_ms: float, user_id: Optional[str] = None):
        """
        Log HTTP request

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            user_id: Optional user ID
        """
        context = {
            'request': {
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration_ms': duration_ms
            }
        }

        if user_id:
            context['user_id'] = user_id

        self.info(f"{method} {path} - {status_code}", context)

    def log_database_query(self, query: str, duration_ms: float,
                            rows_affected: Optional[int] = None):
        """
        Log database query

        Args:
            query: SQL query
            duration_ms: Query duration in milliseconds
            rows_affected: Number of rows affected
        """
        context = {
            'database': {
                'query': query[:200],  # Truncate long queries
                'duration_ms': duration_ms
            }
        }

        if rows_affected is not None:
            context['database']['rows_affected'] = rows_affected

        self.debug("Database query executed", context)

    def log_api_call(self, service: str, endpoint: str, status_code: int,
                    duration_ms: float):
        """
        Log external API call

        Args:
            service: Service name
            endpoint: API endpoint
            status_code: Response status code
            duration_ms: Call duration in milliseconds
        """
        context = {
            'api_call': {
                'service': service,
                'endpoint': endpoint,
                'status_code': status_code,
                'duration_ms': duration_ms
            }
        }

        self.info(f"API call to {service}", context)

    def log_authentication(self, username: str, success: bool,
                            reason: Optional[str] = None):
        """
        Log authentication attempt

        Args:
            username: Username
            success: Whether authentication succeeded
            reason: Optional reason for failure
        """
        context = {
            'authentication': {
                'username': username,
                'success': success
            }
        }

        if reason:
            context['authentication']['reason'] = reason

        level = 'info' if success else 'warning'
        message = f"Authentication {'succeeded' if success else 'failed'} for {username}"

        if success:
            self.info(message, context)
        else:
            self.warning(message, context)

    def log_security_event(self, event_type: str, severity: str,
                            details: Dict[str, Any]):
        """
        Log security event

        Args:
            event_type: Type of security event
            severity: Severity level
            details: Event details
        """
        context = {
            'security': {
                'event_type': event_type,
                'severity': severity,
                'details': details
            }
        }

        if severity.upper() in ['HIGH', 'CRITICAL']:
            self.error(f"Security event: {event_type}", context)
        else:
            self.warning(f"Security event: {event_type}", context)

    def log_performance_metric(self, metric_name: str, value: float,
                                unit: str = 'ms'):
        """
        Log performance metric

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
        """
        context = {
            'performance': {
                'metric': metric_name,
                'value': value,
                'unit': unit
            }
        }

        self.info(f"Performance metric: {metric_name}", context)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for log records"""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON

        Args:
            record: Log record

        Returns:
            str: JSON formatted log entry
        """
        # If message is already JSON, return it
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Create JSON log entry
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': self.formatException(record.exc_info)
                }

            return json.dumps(log_entry)

# Global logger instance
app_logger = StructuredLogger('jpmorgan_api')

# Convenience functions
def log_info(message: str, context: Optional[Dict[str, Any]] = None):
    """Log info message"""
    app_logger.info(message, context)

def log_error(message: str, context: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None):
    """Log error message"""
    app_logger.error(message, context, error)

def log_warning(message: str, context: Optional[Dict[str, Any]] = None):
    """Log warning message"""
    app_logger.warning(message, context)

def log_debug(message: str, context: Optional[Dict[str, Any]] = None):
    """Log debug message"""
    app_logger.debug(message, context)

def log_request(method: str, path: str, status_code: int, duration_ms: float):
    """Log HTTP request"""
    app_logger.log_request(method, path, status_code, duration_ms)
