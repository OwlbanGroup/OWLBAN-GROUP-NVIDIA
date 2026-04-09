"""
Pydantic V2 schemas for API request/response validation
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from pydantic import Field as PydanticField


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    """Schema for individual telemetry events"""
    ver: str = PydanticField(..., description="Telemetry version", json_schema_extra={'examples': ["4.0"]})
    name: str = PydanticField(..., description="Event name", json_schema_extra={'examples': ["Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation"]})
    time: str = PydanticField(..., description="Event timestamp in ISO format", json_schema_extra={'examples': ["2025-09-22T19:42:10.2549325Z"]})
    data: Dict[str, Any] = PydanticField(..., description="Event data payload")
    ext: Dict[str, Any] = PydanticField(default_factory=dict, description="Extended metadata")

    @field_validator('time')
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        try:
            # Try to parse the timestamp
            if v.endswith('Z'):
                datetime.fromisoformat(v[:-1])
            else:
                datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format. Use ISO format.')

    @field_validator('ver')
    @classmethod
    def validate_version(cls, v):
        """Validate version format"""
        if not isinstance(v, str):
            raise ValueError('Version must be a string')
        return v


class TelemetryBatchRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for batch telemetry processing requests"""
    telemetry_data: List[TelemetryEvent] = PydanticField(..., min_length=1, max_length=1000,
                                                    description="List of telemetry events to process")

    @field_validator('telemetry_data')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size limits"""
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 events')
        return v


class MetricsRequest(BaseModel):
    """Schema for metrics requests"""
    hours: int = PydanticField(default=24, ge=1, le=720, description="Hours to look back (1-720)")


class ExportRequest(BaseModel):
    """Schema for data export requests"""
    operation: Optional[str] = PydanticField(None, description="Filter by operation name")
    limit: int = PydanticField(default=1000, ge=1, le=10000, description="Maximum records to export")
    format: str = PydanticField(default="json", pattern="^(json|csv)$", description="Export format")


class AnomalyDetectionRequest(BaseModel):
    """Schema for anomaly detection requests"""
    telemetry_data: List[TelemetryEvent] = PydanticField(..., min_length=1, max_length=500,
                                                    description="Telemetry data for anomaly detection")


class CloudExportRequest(BaseModel):
    """Schema for cloud storage export requests"""
    operation: Optional[str] = PydanticField(None, description="Filter by operation")
    limit: int = PydanticField(default=1000, ge=1, le=10000, description="Records to export")
    format: str = PydanticField(default="json", description="Export format")
    providers: List[str] = PydanticField(default_factory=lambda: ["aws", "gcs", "azure"],
                                description="Cloud providers to export to")
    filename_prefix: str = PydanticField(default="telemetry_export", description="Export filename prefix")

    @field_validator('providers')
    @classmethod
    def validate_providers(cls, v):
        """Validate cloud providers"""
        valid_providers = {"aws", "gcs", "azure"}
        invalid_providers = set(v) - valid_providers
        if invalid_providers:
            raise ValueError(f"Invalid providers: {invalid_providers}. Valid: {valid_providers}")
        return v


class DataConversionRequest(BaseModel):
    """Schema for data format conversion requests"""
    data: List[Dict[str, Any]] = PydanticField(..., min_length=1, description="Data to convert")
    from_format: str = PydanticField(..., pattern="^(json|csv|xml|yaml)$", description="Source format")
    to_format: str = PydanticField(..., pattern="^(json|csv|xml|yaml|excel|parquet)$", description="Target format")
    options: Dict[str, Any] = PydanticField(default_factory=dict, description="Conversion options")


class GitHubIssueRequest(BaseModel):
    """Schema for GitHub issue creation requests"""
    title: str = PydanticField(..., min_length=1, max_length=256, description="Issue title")
    body: str = PydanticField(default="", description="Issue description")
    assignees: List[str] = PydanticField(default_factory=list, description="GitHub usernames to assign")


class BusinessType(str, Enum):
    """Business entity types"""
    CORPORATION = "corporation"
    LLC = "llc"
    PARTNERSHIP = "partnership"
    SOLE_PROPRIETORSHIP = "sole_proprietorship"
    NONPROFIT = "nonprofit"
    OTHER = "other"


class AssetType(str, Enum):
    """Asset types"""
    STOCKS = "stocks"
    BONDS = "bonds"
    REAL_ESTATE = "real_estate"
    CASH = "cash"
    EQUIPMENT = "equipment"
    VEHICLES = "vehicles"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    INVESTMENTS = "investments"
    OTHER = "other"


class BusinessCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for creating a new business"""
    name: str = PydanticField(..., min_length=1, max_length=255, description="Business name")
    type: BusinessType = PydanticField(..., description="Business entity type")
    registration_number: Optional[str] = PydanticField(None, max_length=100, description="Business registration number")
    address: Optional[str] = PydanticField(None, description="Business address")
    contact_info: Optional[Dict[str, Any]] = PydanticField(default_factory=dict, description="Contact information (email, phone, etc.)")


class BusinessUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for updating a business"""
    name: Optional[str] = PydanticField(None, min_length=1, max_length=255, description="Business name")
    type: Optional[BusinessType] = PydanticField(None, description="Business entity type")
    registration_number: Optional[str] = PydanticField(None, max_length=100, description="Business registration number")
    address: Optional[str] = PydanticField(None, description="Business address")
    contact_info: Optional[Dict[str, Any]] = PydanticField(None, description="Contact information (email, phone, etc.)")


class BusinessResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for business response"""
    id: int = PydanticField(..., description="Business ID")
    name: str = PydanticField(..., description="Business name")
    type: BusinessType = PydanticField(..., description="Business entity type")
    registration_number: Optional[str] = PydanticField(None, description="Business registration number")
    address: Optional[str] = PydanticField(None, description="Business address")
    contact_info: Optional[Dict[str, Any]] = PydanticField(None, description="Contact information")
    created_at: Union[str, datetime] = PydanticField(..., description="Creation timestamp")
    updated_at: Union[str, datetime] = PydanticField(..., description="Last update timestamp")

    @field_validator('contact_info', mode='before')
    @classmethod
    def parse_contact_info(cls, v):
        """Parse contact_info from JSON string to dict"""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return v

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def format_datetime(cls, v):
        """Format datetime objects to ISO strings"""
        if isinstance(v, datetime):
            return v.isoformat()
        return v


class AssetCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for creating a new asset"""
    business_id: int = PydanticField(..., description="ID of the owning business")
    name: str = PydanticField(..., min_length=1, max_length=255, description="Asset name")
    type: AssetType = PydanticField(..., description="Asset type")
    value: float = PydanticField(..., gt=0, description="Asset value")
    acquisition_date: str = PydanticField(..., description="Asset acquisition date (ISO format)")
    current_value: Optional[float] = PydanticField(None, gt=0, description="Current asset value")
    ownership_percentage: float = PydanticField(default=100.0, ge=0, le=100, description="Ownership percentage (0-100%)")
    description: Optional[str] = PydanticField(None, description="Asset description")

    @field_validator('acquisition_date')
    @classmethod
    def validate_acquisition_date(cls, v):
        """Validate acquisition date format"""
        try:
            from datetime import datetime
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid acquisition date format. Use ISO format.')


class AssetUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for updating an asset"""
    name: Optional[str] = PydanticField(None, min_length=1, max_length=255, description="Asset name")
    type: Optional[AssetType] = PydanticField(None, description="Asset type")
    value: Optional[float] = PydanticField(None, gt=0, description="Asset value")
    acquisition_date: Optional[str] = PydanticField(None, description="Asset acquisition date (ISO format)")
    current_value: Optional[float] = PydanticField(None, gt=0, description="Current asset value")
    ownership_percentage: Optional[float] = PydanticField(None, ge=0, le=100, description="Ownership percentage (0-100%)")
    description: Optional[str] = PydanticField(None, description="Asset description")

    @field_validator('acquisition_date')
    @classmethod
    def validate_acquisition_date(cls, v):
        """Validate acquisition date format"""
        if v is None:
            return v
        try:
            from datetime import datetime
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid acquisition date format. Use ISO format.')


class AssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    """Schema for asset response"""
    id: int = PydanticField(..., description="Asset ID")
    business_id: int = PydanticField(..., description="ID of the owning business")
    name: str = PydanticField(..., description="Asset name")
    type: AssetType = PydanticField(..., description="Asset type")
    value: float = PydanticField(..., description="Asset value")
    acquisition_date: Optional[str] = PydanticField(None, description="Asset acquisition date")
    current_value: Optional[float] = PydanticField(None, description="Current asset value")
    ownership_percentage: float = PydanticField(..., description="Ownership percentage")
    description: Optional[str] = PydanticField(None, description="Asset description")
    created_at: Optional[str] = PydanticField(None, description="Creation timestamp")
    updated_at: Optional[str] = PydanticField(None, description="Last update timestamp")

    @field_validator('acquisition_date', 'created_at', 'updated_at', mode='before')
    @conflict with existing file
