"""
Pydantic schemas for API request/response validation
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class TelemetryEvent(BaseModel):
    """Schema for individual telemetry events"""
    ver: str = Field(..., description="Telemetry version", json_schema_extra={"example": "4.0"})
    name: str = Field(..., description="Event name", json_schema_extra={"example": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation"})
    time: str = Field(..., description="Event timestamp in ISO format", json_schema_extra={"example": "2025-09-22T19:42:10.2549325Z"})
    data: Dict[str, Any] = Field(..., description="Event data payload")
    ext: Dict[str, Any] = Field(default_factory=dict, description="Extended metadata")

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
    """Schema for batch telemetry processing requests"""
    telemetry_data: List[TelemetryEvent] = Field(..., min_length=1, max_length=1000,
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
    hours: int = Field(default=24, ge=1, le=720, description="Hours to look back (1-720)")


class ExportRequest(BaseModel):
    """Schema for data export requests"""
    operation: Optional[str] = Field(None, description="Filter by operation name")
    limit: int = Field(default=1000, ge=1, le=10000, description="Maximum records to export")
    format: str = Field(default="json", pattern="^(json|csv)$", description="Export format")


class AnomalyDetectionRequest(BaseModel):
    """Schema for anomaly detection requests"""
    telemetry_data: List[TelemetryEvent] = Field(..., min_length=1, max_length=500,
                                                    description="Telemetry data for anomaly detection")


class CloudExportRequest(BaseModel):
    """Schema for cloud storage export requests"""
    operation: Optional[str] = Field(None, description="Filter by operation")
    limit: int = Field(default=1000, ge=1, le=10000, description="Records to export")
    format: str = Field(default="json", description="Export format")
    providers: List[str] = Field(default_factory=lambda: ["aws", "gcs", "azure"],
                                description="Cloud providers to export to")
    filename_prefix: str = Field(default="telemetry_export", description="Export filename prefix")

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
    data: List[Dict[str, Any]] = Field(..., min_length=1, description="Data to convert")
    from_format: str = Field(..., pattern="^(json|csv|xml|yaml)$", description="Source format")
    to_format: str = Field(..., pattern="^(json|csv|xml|yaml|excel|parquet)$", description="Target format")
    options: Dict[str, Any] = Field(default_factory=dict, description="Conversion options")


class GitHubIssueRequest(BaseModel):
    """Schema for GitHub issue creation requests"""
    title: str = Field(..., min_length=1, max_length=256, description="Issue title")
    body: str = Field(default="", description="Issue description")
    assignees: List[str] = Field(default_factory=list, description="GitHub usernames to assign")


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
    """Schema for creating a new business"""
    name: str = Field(..., min_length=1, max_length=255, description="Business name")
    type: BusinessType = Field(..., description="Business entity type")
    registration_number: Optional[str] = Field(None, max_length=100, description="Business registration number")
    address: Optional[str] = Field(None, description="Business address")
    contact_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contact information (email, phone, etc.)")


class BusinessUpdate(BaseModel):
    """Schema for updating a business"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Business name")
    type: Optional[BusinessType] = Field(None, description="Business entity type")
    registration_number: Optional[str] = Field(None, max_length=100, description="Business registration number")
    address: Optional[str] = Field(None, description="Business address")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact information (email, phone, etc.)")


class BusinessResponse(BaseModel):
    """Schema for business response"""
    id: int = Field(..., description="Business ID")
    name: str = Field(..., description="Business name")
    type: BusinessType = Field(..., description="Business entity type")
    registration_number: Optional[str] = Field(None, description="Business registration number")
    address: Optional[str] = Field(None, description="Business address")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact information")
    created_at: Union[str, datetime] = Field(..., description="Creation timestamp")
    updated_at: Union[str, datetime] = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

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
    """Schema for creating a new asset"""
    business_id: int = Field(..., description="ID of the owning business")
    name: str = Field(..., min_length=1, max_length=255, description="Asset name")
    type: AssetType = Field(..., description="Asset type")
    value: float = Field(..., gt=0, description="Asset value")
    acquisition_date: str = Field(..., description="Asset acquisition date (ISO format)")
    current_value: Optional[float] = Field(None, gt=0, description="Current asset value")
    ownership_percentage: float = Field(default=100.0, ge=0, le=100, description="Ownership percentage (0-100%)")
    description: Optional[str] = Field(None, description="Asset description")

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
    """Schema for updating an asset"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Asset name")
    type: Optional[AssetType] = Field(None, description="Asset type")
    value: Optional[float] = Field(None, gt=0, description="Asset value")
    acquisition_date: Optional[str] = Field(None, description="Asset acquisition date (ISO format)")
    current_value: Optional[float] = Field(None, gt=0, description="Current asset value")
    ownership_percentage: Optional[float] = Field(None, ge=0, le=100, description="Ownership percentage (0-100%)")
    description: Optional[str] = Field(None, description="Asset description")

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
    """Schema for asset response"""
    id: int = Field(..., description="Asset ID")
    business_id: int = Field(..., description="ID of the owning business")
    name: str = Field(..., description="Asset name")
    type: AssetType = Field(..., description="Asset type")
    value: float = Field(..., description="Asset value")
    acquisition_date: Optional[str] = Field(None, description="Asset acquisition date")
    current_value: Optional[float] = Field(None, description="Current asset value")
    ownership_percentage: float = Field(..., description="Ownership percentage")
    description: Optional[str] = Field(None, description="Asset description")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

    @field_validator('acquisition_date', 'created_at', 'updated_at', mode='before')
    @classmethod
    def format_datetime(cls, v):
        """Format datetime objects to ISO strings"""
        if isinstance(v, datetime):
            return v.isoformat()
        return v




class OrganizationType(str, Enum):
    """Organization entity types"""
    CORPORATION = "corporation"
    LLC = "llc"
    PARTNERSHIP = "partnership"
    NONPROFIT = "nonprofit"
    TEAM = "team"
    OTHER = "other"


class OrganizationCreate(BaseModel):
    """Schema for creating a new organization"""
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    type: OrganizationType = Field(..., description="Organization entity type")
    registration_number: Optional[str] = Field(None, max_length=100, description="Organization registration number")
    address: Optional[str] = Field(None, description="Organization address")
    contact_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contact information (email, phone, etc.)")
    subscription_type: str = Field(default="team", description="Subscription type (team, business, etc.)")


class OrganizationUpdate(BaseModel):
    """Schema for updating an organization"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Organization name")
    type: Optional[OrganizationType] = Field(None, description="Organization entity type")
    registration_number: Optional[str] = Field(None, max_length=100, description="Organization registration number")
    address: Optional[str] = Field(None, description="Organization address")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact information (email, phone, etc.)")
    subscription_type: Optional[str] = Field(None, description="Subscription type (team, business, etc.)")


class OrganizationResponse(BaseModel):
    """Schema for organization response"""
    id: int = Field(..., description="Organization ID")
    name: str = Field(..., description="Organization name")
    type: OrganizationType = Field(..., description="Organization entity type")
    registration_number: Optional[str] = Field(None, description="Organization registration number")
    address: Optional[str] = Field(None, description="Organization address")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact information")
    owner_id: str = Field(..., description="Owner Docker ID")
    subscription_type: str = Field(..., description="Subscription type")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class OrganizationMemberCreate(BaseModel):
    """Schema for adding an organization member"""
    user_id: str = Field(..., description="Docker ID of the user to add")
    role: str = Field(default="member", description="Role of the member (owner, admin, member)")


class OrganizationMemberUpdate(BaseModel):
    """Schema for updating an organization member"""
    role: str = Field(..., description="New role for the member (owner, admin, member)")


class OrganizationMemberResponse(BaseModel):
    """Schema for organization member response"""
    id: int = Field(..., description="Member ID")
    organization_id: int = Field(..., description="Organization ID")
    user_id: str = Field(..., description="Docker ID")
    role: str = Field(..., description="Member role")
    joined_at: str = Field(..., description="Join timestamp")

    model_config = ConfigDict(from_attributes=True)


class ConvertUserToOrganizationRequest(BaseModel):
    """Schema for converting a user account to an organization"""
    organization_name: str = Field(..., min_length=1, max_length=255, description="Name for the new organization")
    organization_type: OrganizationType = Field(default=OrganizationType.TEAM, description="Type of organization")
    registration_number: Optional[str] = Field(None, max_length=100, description="Organization registration number")
    address: Optional[str] = Field(None, description="Organization address")
    contact_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contact information")
    subscription_type: str = Field(default="team", description="Subscription type")


class APIResponse(BaseModel):
    """Base API response schema"""
    status: str = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                            description="Response timestamp")


class TelemetryResponse(APIResponse):
    """Response for telemetry processing"""
    event_id: Optional[str] = Field(None, description="Processed event ID")


class BatchResponse(APIResponse):
    """Response for batch processing"""
    statistics: Dict[str, Any] = Field(..., description="Processing statistics")


class MetricsResponse(APIResponse):
    """Response for metrics queries"""
    metrics: Dict[str, Any] = Field(..., description="Metrics data")


class ExportResponse(APIResponse):
    """Response for data export"""
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Exported events")
    count: int = Field(default=0, description="Number of exported records")


class AnomalyResponse(APIResponse):
    """Response for anomaly detection"""
    anomaly_results: List[Dict[str, Any]] = Field(..., description="Anomaly detection results")


class CloudExportResponse(APIResponse):
    """Response for cloud export"""
    export_results: Dict[str, Any] = Field(..., description="Export results by provider")
    exported_records: int = Field(..., description="Number of records exported")


class GitHubSearchResponse(APIResponse):
    """Response for GitHub repository search"""
    repositories: List[Dict[str, Any]] = Field(..., description="Found repositories")
    count: int = Field(default=0, description="Number of repositories found")


class GitHubIssuesResponse(APIResponse):
    """Response for GitHub issues listing"""
    issues: List[Dict[str, Any]] = Field(..., description="Repository issues")
    count: int = Field(default=0, description="Number of issues")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


class WebSocketStatusResponse(APIResponse):
    """WebSocket status response"""
    active_connections: int = Field(..., description="Active WebSocket connections")
    unique_clients: int = Field(..., description="Unique client connections")


class FormatsResponse(APIResponse):
    """Supported formats response"""
    import_formats: List[str] = Field(..., description="Supported import formats")
    export_formats: List[str] = Field(..., description="Supported export formats")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    status: str = Field(default="error", description="Error status")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                            description="Error timestamp")


# Validation utilities
def validate_telemetry_data(data: Dict[str, Any]) -> TelemetryEvent:
    """Validate and parse telemetry data"""
    return TelemetryEvent(**data)


def validate_batch_data(data: Dict[str, Any]) -> TelemetryBatchRequest:
    """Validate batch telemetry data"""
    return TelemetryBatchRequest(**data)


def validate_export_request(data: Dict[str, Any]) -> ExportRequest:
    """Validate export request"""
    return ExportRequest(**data)


def validate_cloud_export_request(data: Dict[str, Any]) -> CloudExportRequest:
    """Validate cloud export request"""
    return CloudExportRequest(**data)


def validate_conversion_request(data: Dict[str, Any]) -> DataConversionRequest:
    """Validate data conversion request"""
    return DataConversionRequest(**data)


def validate_github_issue_request(data: Dict[str, Any]) -> GitHubIssueRequest:
    """Validate GitHub issue request"""
    return GitHubIssueRequest(**data)
