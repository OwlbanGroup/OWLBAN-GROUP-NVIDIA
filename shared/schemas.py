"""
Shared schema definitions for API responses
"""
from typing import Any, Optional, Dict, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class APIResponse(BaseModel):
    """Standard API response format"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "data": {"key": "value"}
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response format"""
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "message": "An error occurred",
                "error_code": "ERR_001",
                "details": {"field": "error details"}
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "shared"
    version: str = Field(default="1.0.0", description="Deployed version")
    environment: str = Field(default="development", description="Deployment environment")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checks: Optional[Dict[str, Any]] = None


class TokenData(BaseModel):
    """JWT token payload data"""
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    exp: Optional[int] = None
    iat: Optional[int] = None
