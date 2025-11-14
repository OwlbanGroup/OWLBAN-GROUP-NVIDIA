"""
Enhanced input validation using Pydantic models
"""
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from .schemas import (
    TelemetryEvent, TelemetryBatchRequest, ExportRequest,
    CloudExportRequest, DataConversionRequest, GitHubIssueRequest,
    validate_telemetry_data, validate_batch_data, validate_export_request,
    validate_cloud_export_request, validate_conversion_request,
    validate_github_issue_request
)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class InputValidator:
    """Utility class for input validation using Pydantic models"""

    @staticmethod
    def validate_telemetry_data(data: Dict[str, Any]) -> bool:
        """Validate telemetry data structure using Pydantic"""
        try:
            validate_telemetry_data(data)
            return True
        except Exception as e:
            raise ValidationError(f"Telemetry validation failed: {str(e)}")

    @staticmethod
    def validate_batch_data(data: Dict[str, Any]) -> bool:
        """Validate batch telemetry data using Pydantic"""
        try:
            validate_batch_data(data)
            return True
        except Exception as e:
            raise ValidationError(f"Batch validation failed: {str(e)}")

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
        """Validate export query parameters using Pydantic"""
        try:
            export_req = ExportRequest(
                operation=operation,
                limit=limit,
                format=export_format
            )
            return True
        except Exception as e:
            raise ValidationError(f"Export validation failed: {str(e)}")

    @staticmethod
    def validate_cloud_export_request(data: Dict[str, Any]) -> bool:
        """Validate cloud export request using Pydantic"""
        try:
            validate_cloud_export_request(data)
            return True
        except Exception as e:
            raise ValidationError(f"Cloud export validation failed: {str(e)}")

    @staticmethod
    def validate_conversion_request(data: Dict[str, Any]) -> bool:
        """Validate data conversion request using Pydantic"""
        try:
            validate_conversion_request(data)
            return True
        except Exception as e:
            raise ValidationError(f"Conversion validation failed: {str(e)}")

    @staticmethod
    def validate_github_issue_request(data: Dict[str, Any]) -> bool:
        """Validate GitHub issue request using Pydantic"""
        try:
            validate_github_issue_request(data)
            return True
        except Exception as e:
            raise ValidationError(f"GitHub issue validation failed: {str(e)}")

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
