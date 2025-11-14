import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class InputValidator:
    """Utility class for input validation"""

    @staticmethod
    def validate_telemetry_data(data: Dict[str, Any]) -> bool:
        """Validate telemetry data structure"""
        required_fields = ['ver', 'name', 'time', 'data']
        if not all(field in data for field in required_fields):
            raise ValidationError("Missing required fields: ver, name, time, data")

        # Validate version
        if not isinstance(data['ver'], str) or not data['ver'].strip():
            raise ValidationError("Version must be a non-empty string")

        # Validate name
        if not isinstance(data['name'], str) or not data['name'].strip():
            raise ValidationError("Name must be a non-empty string")

        # Validate time
        if not isinstance(data['time'], str):
            raise ValidationError("Time must be a string")
        try:
            datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
        except ValueError:
            raise ValidationError("Invalid time format")

        # Validate data field
        if not isinstance(data['data'], dict):
            raise ValidationError("Data must be a dictionary")

        return True

    @staticmethod
    def validate_batch_data(data: Dict[str, Any]) -> bool:
        """Validate batch telemetry data"""
        if 'telemetry_data' not in data:
            raise ValidationError("Missing telemetry_data field")

        if not isinstance(data['telemetry_data'], list):
            raise ValidationError("telemetry_data must be a list")

        if len(data['telemetry_data']) == 0:
            raise ValidationError("telemetry_data cannot be empty")

        if len(data['telemetry_data']) > 1000:
            raise ValidationError("Maximum batch size is 1000")

        for item in data['telemetry_data']:
            if not isinstance(item, dict):
                raise ValidationError("Each item in telemetry_data must be a dictionary")
            InputValidator.validate_telemetry_data(item)

        return True

    @staticmethod
    def validate_metrics_params(hours: int) -> bool:
        """Validate metrics query parameters"""
        if not isinstance(hours, int):
            raise ValidationError("Hours must be an integer")

        if hours <= 0 or hours > 720:
            raise ValidationError("Hours must be between 1 and 720")

        return True

    @staticmethod
    def validate_export_params(operation: Optional[str], limit: int, export_format: str) -> bool:
        """Validate export query parameters"""
        if operation and not isinstance(operation, str):
            raise ValidationError("Operation must be a string")

        if not isinstance(limit, int):
            raise ValidationError("Limit must be an integer")

        if limit <= 0 or limit > 10000:
            raise ValidationError("Limit must be between 1 and 10000")

        if export_format not in ['json', 'csv']:
            raise ValidationError("Format must be json or csv")

        return True

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            raise ValidationError("Input must be a string")

        # Remove potentially harmful characters
        sanitized = re.sub(r'[^\w\s\-_.]', '', input_str)
        sanitized = sanitized.strip()

        if len(sanitized) > max_length:
            raise ValidationError(f"String too long (max {max_length} characters)")

        return sanitized

    @staticmethod
    def validate_json_structure(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate JSON structure against a schema"""
        for key, expected_type in schema.items():
            if key not in data:
                raise ValidationError(f"Missing required field: {key}")

            if not isinstance(data[key], expected_type):
                raise ValidationError(f"Field {key} must be of type {expected_type.__name__}")

        return True
