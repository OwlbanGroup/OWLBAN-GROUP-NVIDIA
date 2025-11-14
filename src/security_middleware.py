"""
Security middleware for input sanitization and validation
"""
import re
from typing import Dict, Any, List
from flask import request, jsonify
from werkzeug.exceptions import BadRequest
from .logger import telemetry_logger


class SecurityMiddleware:
    """Security middleware for input sanitization and validation"""

    def __init__(self, app):
        self.app = app
        self.sanitization_rules = {
            'sql_injection': self._sanitize_sql_injection,
            'xss': self._sanitize_xss,
            'command_injection': self._sanitize_command_injection,
            'path_traversal': self._sanitize_path_traversal
        }
        self.validation_rules = {
            'json_schema': self._validate_json_schema,
            'input_length': self._validate_input_length,
            'allowed_chars': self._validate_allowed_chars
        }

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data"""
        if isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        elif isinstance(data, str):
            return self._apply_sanitization_rules(data)
        else:
            return data

    def _apply_sanitization_rules(self, input_str: str) -> str:
        """Apply all sanitization rules to input string"""
        sanitized = input_str

        for rule_name, rule_func in self.sanitization_rules.items():
            try:
                sanitized = rule_func(sanitized)
            except Exception as e:
                telemetry_logger.get_logger().warning(f"Sanitization rule {rule_name} failed: {e}")

        return sanitized

    def _sanitize_sql_injection(self, input_str: str) -> str:
        """Sanitize SQL injection attempts"""
        # Remove common SQL injection patterns
        patterns = [
            r';\s*--',  # Semicolon followed by comment
            r';\s*/\*',  # Semicolon followed by block comment
            r'union\s+select',  # UNION SELECT
            r'/\*.*?\*/',  # Block comments
            r'--.*?$',  # Line comments
        ]

        for pattern in patterns:
            input_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE | re.MULTILINE)

        return input_str

    def _sanitize_xss(self, input_str: str) -> str:
        """Sanitize XSS attempts"""
        # Remove dangerous HTML/JS tags and attributes
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'<iframe[^>]*>.*?</iframe>',  # Iframe tags
            r'<object[^>]*>.*?</object>',  # Object tags
            r'<embed[^>]*>.*?</embed>',  # Embed tags
            r'on\w+\s*=',  # Event handlers
            r'javascript:',  # JavaScript URLs
            r'vbscript:',  # VBScript URLs
            r'data:',  # Data URLs (potentially dangerous)
        ]

        for pattern in dangerous_patterns:
            input_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE)

        # HTML entity encode dangerous characters
        input_str = input_str.replace('<', '<').replace('>', '>')
        input_str = input_str.replace('"', '"').replace("'", '&#x27;')

        return input_str

    def _sanitize_command_injection(self, input_str: str) -> str:
        """Sanitize command injection attempts"""
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '\n', '\r']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')

        # Remove common command injection patterns
        patterns = [
            r'\|\|',  # OR operator
            r'&&',    # AND operator
            r';\s*rm',  # rm command
            r';\s*wget',  # wget command
            r';\s*curl',  # curl command
        ]

        for pattern in patterns:
            input_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE)

        return input_str

    def _sanitize_path_traversal(self, input_str: str) -> str:
        """Sanitize path traversal attempts"""
        # Remove path traversal patterns
        patterns = [
            r'\.\./',  # Parent directory
            r'\.\.\\',  # Parent directory (Windows)
            r'~',      # Home directory
            r'/',      # Root directory (if not expected)
        ]

        # Only remove if they appear in dangerous contexts
        for pattern in patterns:
            # Be more conservative - only remove if followed by suspicious patterns
            if re.search(r'\.\./.*\.(txt|log|db|sql|php|asp)', input_str, re.IGNORECASE):
                input_str = re.sub(pattern, '', input_str)

        return input_str

    def validate_input(self, data: Any, rules: Dict[str, Any] = None) -> bool:
        """Validate input data against rules"""
        if rules is None:
            rules = {}

        try:
            for rule_name, rule_config in rules.items():
                if rule_name in self.validation_rules:
                    if not self.validation_rules[rule_name](data, rule_config):
                        telemetry_logger.get_logger().warning(f"Validation failed for rule: {rule_name}")
                        return False
            return True
        except Exception as e:
            telemetry_logger.get_logger().error(f"Input validation error: {e}")
            return False

    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data against JSON schema"""
        try:
            # Simple schema validation (can be enhanced with jsonschema library)
            if 'type' in schema:
                expected_type = schema['type']
                if expected_type == 'string' and not isinstance(data, str):
                    return False
                elif expected_type == 'number' and not isinstance(data, (int, float)):
                    return False
                elif expected_type == 'boolean' and not isinstance(data, bool):
                    return False
                elif expected_type == 'array' and not isinstance(data, list):
                    return False
                elif expected_type == 'object' and not isinstance(data, dict):
                    return False

            if 'maxLength' in schema and isinstance(data, str):
                if len(data) > schema['maxLength']:
                    return False

            if 'minLength' in schema and isinstance(data, str):
                if len(data) < schema['minLength']:
                    return False

            return True
        except Exception:
            return False

    def _validate_input_length(self, data: Any, config: Dict[str, Any]) -> bool:
        """Validate input length"""
        try:
            if isinstance(data, str):
                length = len(data)
                if 'max' in config and length > config['max']:
                    return False
                if 'min' in config and length < config['min']:
                    return False
            elif isinstance(data, (list, dict)):
                length = len(data)
                if 'max' in config and length > config['max']:
                    return False
                if 'min' in config and length < config['min']:
                    return False
            return True
        except Exception:
            return False

    def _validate_allowed_chars(self, data: Any, config: Dict[str, Any]) -> bool:
        """Validate allowed characters"""
        try:
            if not isinstance(data, str):
                return True

            allowed_pattern = config.get('pattern', r'^[a-zA-Z0-9\s\-_\.]+$')
            if not re.match(allowed_pattern, data):
                return False

            disallowed_chars = config.get('disallowed', [])
            for char in disallowed_chars:
                if char in data:
                    return False

            return True
        except Exception:
            return False


# Global security middleware instance
security_middleware = SecurityMiddleware(None)


def sanitize_request_data():
    """Flask before_request handler to sanitize input data"""
    try:
        if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
            data = request.get_json()
            if data:
                sanitized_data = security_middleware.sanitize_input(data)

                # Validate sanitized data
                validation_rules = {
                    'input_length': {'max': 10000},  # Max 10KB of data
                    'allowed_chars': {'disallowed': ['<', '>', '&', '"', "'"]}
                }

                if not security_middleware.validate_input(sanitized_data, validation_rules):
                    telemetry_logger.get_logger().warning("Input validation failed for request")
                    return jsonify({
                        'error': 'Invalid input data',
                        'status': 'error'
                    }), 400

                # Replace request data with sanitized version
                # Note: This is a simplified approach. In production, you might want to use a custom request class
                request._cached_json = (sanitized_data, sanitized_data)

    except Exception as e:
        telemetry_logger.get_logger().error(f"Request sanitization error: {e}")
        # Don't block the request, just log the error


def audit_request():
    """Audit incoming requests for security monitoring"""
    try:
        # Log security-relevant request information
        audit_data = {
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'content_length': request.content_length,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Check for suspicious patterns
        suspicious_indicators = []

        if request.path and '..' in request.path:
            suspicious_indicators.append('path_traversal_attempt')

        if request.headers.get('User-Agent', '').lower() in ['curl', 'wget', 'python-requests']:
            suspicious_indicators.append('automated_request')

        if len(request.args) > 10:  # Too many query parameters
            suspicious_indicators.append('excessive_parameters')

        if suspicious_indicators:
            audit_data['suspicious_indicators'] = suspicious_indicators
            telemetry_logger.get_logger().warning("Suspicious request detected", extra=audit_data)
        else:
            telemetry_logger.get_logger().debug("Request audit", extra=audit_data)

    except Exception as e:
        telemetry_logger.get_logger().error(f"Request audit error: {e}")


# Import here to avoid circular imports
from datetime import datetime, timezone
