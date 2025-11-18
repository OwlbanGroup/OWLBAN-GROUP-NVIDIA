"""
Shared schema definitions for API responses
"""
from typing import Any, Optional, Dict
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response format"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "data": {"key": "value"}
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response format"""
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "An error occurred",
                "error_code": "ERR_001",
                "details": {"field": "error details"}
            }
        }
