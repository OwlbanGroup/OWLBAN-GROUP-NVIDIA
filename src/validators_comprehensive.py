"""
Comprehensive Input Validation
Provides extensive validation for all API endpoints
"""
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class ComprehensiveValidators:
    """Comprehensive validation utilities for all endpoints"""

    # Email regex pattern
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    # Phone regex pattern (international format)
    PHONE_PATTERN = re.compile(r'^\+?[1-9]\d{6,14}$')

    # URL regex pattern
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format

        Args:
            email: Email address to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise ValidationError("Email must be a non-empty string")

        if not ComprehensiveValidators.EMAIL_PATTERN.match(email):
            raise ValidationError(f"Invalid email format: {email}")

        if len(email) > 254:  # RFC 5321
            raise ValidationError("Email address too long (max 254 characters)")

        return True

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format

        Args:
            phone: Phone number to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If phone is invalid
        """
        if not phone or not isinstance(phone, str):
            raise ValidationError("Phone must be a non-empty string")

        # Remove common separators
        cleaned = phone.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')

        if not ComprehensiveValidators.PHONE_PATTERN.match(cleaned):
            raise ValidationError(f"Invalid phone format: {phone}")

        return True

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format

        Args:
            url: URL to validate

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If URL is invalid
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        if not ComprehensiveValidators.URL_PATTERN.match(url):
            raise ValidationError(f"Invalid URL format: {url}")

        if len(url) > 2048:
            raise ValidationError("URL too long (max 2048 characters)")

        return True

    @staticmethod
    def validate_string(value: str, field_name: str, min_length: int = 1,
                        max_length: int = 255, pattern: Optional[str] = None) -> bool:
        """
        Validate string field

        Args:
            value: String value to validate
            field_name: Name of the field (for error messages)
            min_length: Minimum length
            max_length: Maximum length
            pattern: Optional regex pattern

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")

        if len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters")

        if len(value) > max_length:
            raise ValidationError(f"{field_name} must be at most {max_length} characters")

        if pattern and not re.match(pattern, value):
            raise ValidationError(f"{field_name} format is invalid")

        return True

    @staticmethod
    def validate_number(value: Any, field_name: str, min_value: Optional[float] = None,
                        max_value: Optional[float] = None) -> bool:
        """
        Validate numeric field

        Args:
            value: Numeric value to validate
            field_name: Name of the field
            min_value: Minimum value
            max_value: Maximum value

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number")

        if min_value is not None and value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}")

        if max_value is not None and value > max_value:
            raise ValidationError(f"{field_name} must be at most {max_value}")

        return True

    @staticmethod
    def validate_date(date_str: str, field_name: str) -> bool:
        """
        Validate date string (ISO 8601 format)

        Args:
            date_str: Date string to validate
            field_name: Name of the field

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(date_str, str):
            raise ValidationError(f"{field_name} must be a string")

        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except ValueError:
            raise ValidationError(f"{field_name} must be in ISO 8601 format")

    @staticmethod
    def validate_business_data(data: Dict[str, Any]) -> bool:
        """
        Validate business creation/update data

        Args:
            data: Business data dictionary

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ['name', 'type', 'registration_number']

        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")

        # Validate name
        ComprehensiveValidators.validate_string(
            data['name'], 'name', min_length=2, max_length=200
        )

        # Validate type
        valid_types = ['corporation', 'llc', 'partnership', 'sole_proprietorship', 'nonprofit']
        if data['type'] not in valid_types:
            raise ValidationError(f"Invalid business type. Must be one of: {', '.join(valid_types)}")

        # Validate registration number
        ComprehensiveValidators.validate_string(
            data['registration_number'], 'registration_number',
            min_length=5, max_length=50
        )

        # Validate optional fields
        if 'address' in data:
            ComprehensiveValidators.validate_string(
                data['address'], 'address', max_length=500
            )

        if 'contact_info' in data:
            contact = data['contact_info']
            if 'email' in contact:
                ComprehensiveValidators.validate_email(contact['email'])
            if 'phone' in contact:
                ComprehensiveValidators.validate_phone(contact['phone'])

        return True

    @staticmethod
    def validate_asset_data(data: Dict[str, Any]) -> bool:
        """
        Validate asset creation/update data

        Args:
            data: Asset data dictionary

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ['business_id', 'name', 'type', 'value']

        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")

        # Validate business_id
        ComprehensiveValidators.validate_number(
            data['business_id'], 'business_id', min_value=1
        )

        # Validate name
        ComprehensiveValidators.validate_string(
            data['name'], 'name', min_length=2, max_length=200
        )

        # Validate type
        valid_types = ['equipment', 'property', 'vehicle', 'intellectual_property', 'other']
        if data['type'] not in valid_types:
            raise ValidationError(f"Invalid asset type. Must be one of: {', '.join(valid_types)}")

        # Validate value
        ComprehensiveValidators.validate_number(
            data['value'], 'value', min_value=0
        )

        # Validate optional fields
        if 'acquisition_date' in data:
            ComprehensiveValidators.validate_date(data['acquisition_date'], 'acquisition_date')

        if 'ownership_percentage' in data:
            ComprehensiveValidators.validate_number(
                data['ownership_percentage'], 'ownership_percentage',
                min_value=0, max_value=100
            )

        if 'description' in data:
            ComprehensiveValidators.validate_string(
                data['description'], 'description', max_length=1000
            )

        return True

    @staticmethod
    def validate_telemetry_data(data: Dict[str, Any]) -> bool:
        """
        Validate telemetry event data

        Args:
            data: Telemetry data dictionary

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ['ver', 'name', 'time']

        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")

        # Validate version
        ComprehensiveValidators.validate_string(data['ver'], 'ver', max_length=10)

        # Validate name
        ComprehensiveValidators.validate_string(data['name'], 'name', max_length=500)

        # Validate time
        ComprehensiveValidators.validate_date(data['time'], 'time')

        # Validate optional data field
        if 'data' in data and not isinstance(data['data'], dict):
            raise ValidationError("data field must be a dictionary")

        return True

    @staticmethod
    def validate_user_registration(data: Dict[str, Any]) -> bool:
        """
        Validate user registration data

        Args:
            data: User registration data

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If validation fails
        """
        required_fields = ['username', 'password', 'email']

        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")

        # Validate username
        ComprehensiveValidators.validate_string(
            data['username'], 'username',
            min_length=3, max_length=50,
            pattern=r'^[a-zA-Z0-9_-]+$'
        )

        # Validate password
        password = data['password']
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters")
        if len(password) > 128:
            raise ValidationError("Password must be at most 128 characters")
        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', password):
            raise ValidationError("Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', password):
            raise ValidationError("Password must contain at least one digit")

        # Validate email
        ComprehensiveValidators.validate_email(data['email'])

        return True

    @staticmethod
    def sanitize_input(value: str) -> str:
        """
        Sanitize user input to prevent injection attacks

        Args:
            value: Input string to sanitize

        Returns:
            str: Sanitized string
        """
        if not isinstance(value, str):
            return value

        # Remove script tags
        import re
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)

        # Escape potentially dangerous characters
        sanitized = sanitized.replace('<', '<').replace('>', '>')
        sanitized = sanitized.replace('"', '"').replace("'", '&#x27;')
        sanitized = sanitized.replace('&', '&amp;')

        return sanitized

    @staticmethod
    def validate_batch_size(size: int, max_size: int = 1000) -> bool:
        """
        Validate batch operation size

        Args:
            size: Batch size
            max_size: Maximum allowed size

        Returns:
            bool: True if valid

        Raises:
            ValidationError: If size is invalid
        """
        if not isinstance(size, int):
            raise ValidationError("Batch size must be an integer")

        if size < 1:
            raise ValidationError("Batch size must be at least 1")

        if size > max_size:
            raise ValidationError(f"Batch size must be at most {max_size}")

        return True

# Convenience functions for common validations
def validate_business(data: Dict[str, Any]) -> bool:
    """Validate business data"""
    return ComprehensiveValidators.validate_business_data(data)

def validate_asset(data: Dict[str, Any]) -> bool:
    """Validate asset data"""
    return ComprehensiveValidators.validate_asset_data(data)

def validate_telemetry(data: Dict[str, Any]) -> bool:
    """Validate telemetry data"""
    return ComprehensiveValidators.validate_telemetry_data(data)

def validate_user(data: Dict[str, Any]) -> bool:
    """Validate user registration data"""
    return ComprehensiveValidators.validate_user_registration(data)
