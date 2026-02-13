"""
FastAPI application for JPMorgan Financial APIs - Async Edition
"""
import asyncio
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any, List

import numpy as np
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# JP Morgan Financial Dashboard Extensions
import psycopg2
from psycopg2.extras import RealDictCursor
import schedule
import time
from threading import Thread
import requests
from decimal import Decimal

# Import sync scheduler for integrated data synchronization
try:
    from sync_scheduler import JPMorganSyncScheduler, create_scheduler
except ImportError:
    # Fallback if sync_scheduler is not available
    create_scheduler = None

# Import PFM blueprint (Note: PFM is a Flask blueprint, not FastAPI - needs conversion)
# For now, we'll handle this after the app is created
pfm_bp = None
PFM_BLUEPRINT_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Load version information
def get_version():
    """Get the current version from VERSION file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return '1.0.0'

# Configuration
try:
    from config import config
except ImportError:
    # Fallback: define a minimal config object for local development
    class Config:
        SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret')
        TOKEN_CLIENT_ID = os.environ.get('TOKEN_CLIENT_ID', 'dummy_client_id')
        TOKEN_CLIENT_SECRET = os.environ.get('TOKEN_CLIENT_SECRET', 'dummy_client_secret')
        TOKEN_URL = os.environ.get('TOKEN_URL', 'https://dummy.token.url')
        TOKEN_SCOPE = os.environ.get('TOKEN_SCOPE', 'dummy_scope')
        REDIS_URL = os.environ.get('REDIS_URL', None)
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///telemetry.db')
        @staticmethod
        def get_all_settings():
            return {
                'SECRET_KEY': Config.SECRET_KEY,
                'TOKEN_CLIENT_ID': Config.TOKEN_CLIENT_ID,
                'TOKEN_CLIENT_SECRET': Config.TOKEN_CLIENT_SECRET,
                'TOKEN_URL': Config.TOKEN_URL,
                'TOKEN_SCOPE': Config.TOKEN_SCOPE,
                'REDIS_URL': Config.REDIS_URL,
                'LOG_LEVEL': Config.LOG_LEVEL,
                'DATABASE_URL': Config.DATABASE_URL
            }
    config = Config()  # type: ignore

# Ensure 'src' directory is in sys.path before importing modules
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.telemetry_handler import telemetry_handler  # type: ignore
except ImportError as e:
    raise ImportError("Could not import 'src.telemetry_handler'. Make sure 'src/telemetry_handler.py' exists and is not empty.") from e

from src.logger import telemetry_logger  # type: ignore
from src.token_manager import TokenManager  # type: ignore
from src.validation import InputValidator, ValidationError  # type: ignore

# Initialize cloud storage with error handling
try:
    from src.cloud_storage import setup_cloud_storage  # type: ignore
    setup_cloud_storage(config.get_all_settings())
except ImportError:
    # Cloud storage not available, continue without it
    pass

from src.data_format_converter import DataFormatConverter  # type: ignore
from src.ml_model import ml_model  # type: ignore
from src.database_fixed import async_db_manager, AsyncDatabaseManager  # type: ignore
from src.schemas import BusinessCreate, BusinessUpdate, BusinessResponse, AssetCreate, AssetUpdate, AssetResponse, OrganizationCreate, OrganizationUpdate, OrganizationResponse, OrganizationMemberCreate, OrganizationMemberUpdate, OrganizationMemberResponse, ConvertUserToOrganizationRequest  # type: ignore

# Initialize AI service with error handling
try:
    from src.ai_service import ai_service  # type: ignore
except ImportError:
    ai_service = None

# Initialize auth0 with error handling
try:
    from src.auth0_auth import setup_auth0_routes, auth0_required  # type: ignore
except ImportError:
    setup_auth0_routes = None
    auth0_required = None

# Initialize payments service with error handling
try:
    from src.payments_service import payments_service  # type: ignore
except ImportError:
    payments_service = None

# Initialize ML model
anomaly_detector = ml_model()

# Initialize sync scheduler
sync_scheduler = None
try:
    sync_scheduler = create_scheduler()
    telemetry_logger.get_logger().info("Sync scheduler initialized successfully")
except Exception as e:
    telemetry_logger.get_logger().error(f"Failed to initialize sync scheduler: {e}")
    sync_scheduler = None

# Prometheus metrics (FastAPI async version)
REQUEST_COUNT_ASYNC = Counter('http_requests_total_async', 'Total HTTP requests (async)', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY_ASYNC = Histogram('http_request_duration_seconds_async', 'HTTP request duration (async)', ['method', 'endpoint'])
ACTIVE_CONNECTIONS_ASYNC = Gauge('active_connections_async', 'Number of active connections (async)')
ERROR_COUNT_ASYNC = Counter('errors_total_async', 'Total errors (async)', ['type', 'endpoint'])
TELEMETRY_EVENTS_PROCESSED_ASYNC = Counter('telemetry_events_processed_total_async', 'Total telemetry events processed (async)', ['status'])
BATCH_SIZE_ASYNC = Histogram('telemetry_batch_size_async', 'Size of telemetry batches processed (async)')

# Initialize FastAPI app
app = FastAPI(
    title='JPMorgan Telemetry API - Async',
    version=get_version(),
    description='Enterprise-grade API for processing Microsoft Windows Store '
                'telemetry data with ML anomaly detection, cloud storage integration, '
                'and GitHub MCP connectivity. Built with FastAPI for high performance.',
    docs_url='/swagger/',
    redoc_url='/redoc/',
    openapi_url='/openapi.json'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Initialize token manager
token_manager = TokenManager(
    client_id=config.TOKEN_CLIENT_ID,
    client_secret=config.TOKEN_CLIENT_SECRET,
    token_url=config.TOKEN_URL,
    scope=config.TOKEN_SCOPE
)

# Initialize Redis cache
redis_client = None
if config.REDIS_URL:
    try:
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    except Exception as e:
        telemetry_logger.get_logger().warning(f"Failed to connect to Redis at {config.REDIS_URL}: {str(e)}. Using in-memory cache.")

# In-memory user store for demonstration (replace with DB in production)
users = {}

# Add test user if in testing mode
if os.environ.get('TESTING') == '1':
    users['testuser'] = {
        'password': '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fM9q1wqK',  # testpass
        'created_at': datetime.now(timezone.utc).isoformat(),
        'token': 'test_token',
        'token_created_at': datetime.now(timezone.utc).isoformat()
    }
    users['davidleeper'] = {
        'password': '$2b$12$password123.hash.here',  # password123
        'created_at': datetime.now(timezone.utc).isoformat(),
        'token': 'david_token',
        'token_created_at': datetime.now(timezone.utc).isoformat()
    }

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2024-01-01T00:00:00Z")
    version: str = Field(..., example="1.0.0")

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)

class UserLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)

class UserResponse(BaseModel):
    status: str = Field(..., example="success")
    message: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None
    created_at: Optional[str] = None
    token_created_at: Optional[str] = None

class TelemetryEvent(BaseModel):
    name: str
    ver: str
    time: str
    iKey: str
    flags: Optional[Dict[str, Any]] = None
    cV: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class TelemetryBatchRequest(BaseModel):
    telemetry_data: List[Dict[str, Any]] = Field(..., description="List of telemetry events to process")

class TelemetryResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Telemetry data processed successfully")
    timestamp: str = Field(..., example="2024-01-01T00:00:00Z")

class MetricsResponse(BaseModel):
    status: str = Field(..., example="success")
    metrics: Dict[str, Any]
    timestamp: str = Field(..., example="2024-01-01T00:00:00Z")

# Dependency injection
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Get current user from token"""
    if not credentials:
        return None

    token = credentials.credentials
    # Validate token against in-memory store
    for username, user_data in users.items():
        if user_data.get('token') == token:
            return username
    return None

async def require_auth(current_user: Optional[str] = Depends(get_current_user)):
    """Require authentication"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

# Middleware for request logging and metrics
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging and metrics middleware"""
    start_time = time.time()

    # Increment active connections
    ACTIVE_CONNECTIONS_ASYNC.inc()

    try:
        response = await call_next(request)

        # Record metrics
        REQUEST_COUNT_ASYNC.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()

        REQUEST_LATENCY_ASYNC.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)

        return response
    except Exception as e:
        ERROR_COUNT_ASYNC.labels(
            type=type(e).__name__,
            endpoint=request.url.path
        ).inc()
        raise
    finally:
        ACTIVE_CONNECTIONS_ASYNC.dec()

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint"""
    telemetry_logger.get_logger().info("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=get_version()
    )

@app.post("/user/register", response_model=UserResponse)
@limiter.limit("5/minute")
async def register_user(request: Request, user_data: UserRegisterRequest):
    """
    Register a new user with username and password
    """
    try:
        from passlib.hash import bcrypt

        if user_data.username in users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )

        hashed_password = bcrypt.hash(user_data.password)
        users[user_data.username] = {
            'password': hashed_password,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        return UserResponse(
            status="success",
            message="User created successfully"
        )
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'register_user'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/user/login", response_model=UserResponse)
@limiter.limit("10/minute")
async def login_user(request: Request, login_data: UserLoginRequest):
    """
    Login user and return a token
    """
    try:
        from passlib.hash import bcrypt

        user = users.get(login_data.username)
        if not user or not bcrypt.verify(login_data.password, user['password']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Generate a simple token (in production use JWT or OAuth)
        token = secrets.token_hex(16)
        users[login_data.username]['token'] = token
        users[login_data.username]['token_created_at'] = datetime.now(timezone.utc).isoformat()

        return UserResponse(
            status="success",
            token=token
        )
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'login_user'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/user/profile", response_model=UserResponse)
@limiter.limit("10/minute")
async def user_profile(request: Request, current_user: str = Depends(require_auth)):
    """
    Get user profile information (requires user token)
    """
    try:
        user_data = users.get(current_user)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return UserResponse(
            status="success",
            username=current_user,
            created_at=user_data['created_at'],
            token_created_at=user_data.get('token_created_at')
        )
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'user_profile'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/telemetry", response_model=TelemetryResponse)
@limiter.limit("5/minute")
async def receive_telemetry(
    telemetry_data: Dict[str, Any],
    request: Request,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Receive and process telemetry data
    """
    try:
        # Validate the data
        try:
            InputValidator.validate_telemetry_data(telemetry_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Validation error: {str(e)}'
            )

        # Process the event (using async database manager)
        success = await process_telemetry_event_async(telemetry_data)

        if success:
            TELEMETRY_EVENTS_PROCESSED_ASYNC.labels(status="success").inc()
            return TelemetryResponse(
                status="success",
                message="Telemetry data processed successfully",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            TELEMETRY_EVENTS_PROCESSED_ASYNC.labels(status="failed").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process telemetry data"
            )

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid JSON format"
        )
    except Exception as e:
        if 'JSON' in str(e) or 'json' in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid JSON format"
            )
        else:
            telemetry_logger.log_error(e, {'context': 'telemetry_endpoint'})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

@app.post("/telemetry/batch", response_model=TelemetryResponse)
@limiter.limit("3/minute")
async def receive_telemetry_batch(
    batch_request: TelemetryBatchRequest,
    request: Request,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Receive and process batch telemetry data
    """
    try:
        # Validate batch data
        try:
            InputValidator.validate_batch_data({"telemetry_data": batch_request.telemetry_data})
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Validation error: {str(e)}'
            )

        # Process batch asynchronously
        stats = await process_telemetry_batch_async(batch_request.telemetry_data)

        BATCH_SIZE_ASYNC.observe(len(batch_request.telemetry_data))

        return TelemetryResponse(
            status="success",
            message=f'Batch processed: {stats["successful"]}/{stats["total"]} events successful',
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON format"
        )
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'telemetry_batch_endpoint'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/telemetry/metrics", response_model=MetricsResponse)
@limiter.limit("5/minute")
async def get_telemetry_metrics(request: Request, hours: int = 24):
    """
    Get telemetry metrics and statistics
    """
    try:
        if hours <= 0 or hours > 720:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 720"
            )

        # Get metrics from async database
        metrics = await get_telemetry_metrics_async(hours)

        return MetricsResponse(
            status="success",
            metrics=metrics,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'metrics_endpoint'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Business Management Endpoints
@app.get("/businesses", response_model=Dict[str, Any])
async def list_businesses(current_user: str = Depends(require_auth)):
    """
    List all businesses
    """
    try:
        businesses = await async_db_manager.get_all_businesses()
        return {
            'status': 'success',
            'businesses': [BusinessResponse.from_orm(business).dict() for business in businesses],
            'count': len(businesses),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_businesses'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/businesses", response_model=Dict[str, Any])
async def create_business(
    business_data: BusinessCreate,
    current_user: str = Depends(require_auth)
):
    """
    Create a new business
    """
    try:
        business = await async_db_manager.create_business(business_data.dict())
        return {
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_business'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/businesses/{business_id}", response_model=Dict[str, Any])
async def get_business(
    business_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Get business details by ID
    """
    try:
        business = await async_db_manager.get_business_by_id(business_id)
        if not business:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Business not found"
            )
        return {
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.put("/businesses/{business_id}", response_model=Dict[str, Any])
async def update_business(
    business_id: int,
    business_data: BusinessUpdate,
    current_user: str = Depends(require_auth)
):
    """
    Update business details
    """
    try:
        business = await async_db_manager.update_business(business_id, business_data.dict(exclude_unset=True))
        if not business:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Business not found"
            )
        return {
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_business'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.delete("/businesses/{business_id}", response_model=Dict[str, Any])
async def delete_business(
    business_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Delete a business
    """
    try:
        success = await async_db_manager.delete_business(business_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Business not found"
            )
        return {
            'status': 'success',
            'message': 'Business deleted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_business'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Asset Management Endpoints
@app.get("/assets", response_model=Dict[str, Any])
async def list_assets(current_user: str = Depends(require_auth)):
    """
    List all assets
    """
    try:
        assets = await async_db_manager.get_all_assets()
        return {
            'status': 'success',
            'assets': [AssetResponse.from_orm(asset).dict() for asset in assets],
            'count': len(assets),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_assets'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/assets", response_model=Dict[str, Any])
async def create_asset(
    asset_data: AssetCreate,
    current_user: str = Depends(require_auth)
):
    """
    Create a new asset
    """
    try:
        asset = await async_db_manager.create_asset(asset_data.dict())
        return {
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_asset'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/assets/{asset_id}", response_model=Dict[str, Any])
async def get_asset(
    asset_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Get asset details by ID
    """
    try:
        asset = await async_db_manager.get_asset_by_id(asset_id)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Asset not found"
            )
        return {
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_asset'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.put("/assets/{asset_id}", response_model=Dict[str, Any])
async def update_asset(
    asset_id: int,
    asset_data: AssetUpdate,
    current_user: str = Depends(require_auth)
):
    """
    Update asset details
    """
    try:
        asset = await async_db_manager.update_asset(asset_id, asset_data.dict(exclude_unset=True))
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Asset not found"
            )
        return {
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_asset'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.delete("/assets/{asset_id}", response_model=Dict[str, Any])
async def delete_asset(
    asset_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Delete an asset
    """
    try:
        success = await async_db_manager.delete_asset(asset_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Asset not found"
            )
        return {
            'status': 'success',
            'message': 'Asset deleted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_asset'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# ML Endpoints
@app.post("/ml/anomalies", response_model=Dict[str, Any])
@limiter.limit("2 per minute")
async def detect_anomalies(
    request: Request,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Detect anomalies in telemetry data using ML

    Expected JSON payload:
    {
        "telemetry_data": [
            { ... telemetry event 1 ... },
            { ... telemetry event 2 ... },
            ...
        ]
    }
    """
    try:
        request_data = await request.json()

        if not request_data or 'telemetry_data' not in request_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No telemetry data provided"
            )

        telemetry_data_list = request_data['telemetry_data']

        if not isinstance(telemetry_data_list, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="telemetry_data must be a list"
            )

        # Detect anomalies using the anomaly detector
        anomaly_results = anomaly_detector.detect_anomalies(telemetry_data_list)

        return {
            'status': 'success',
            'anomaly_results': anomaly_results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'anomalies_endpoint'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/ml/train", response_model=Dict[str, Any])
@limiter.limit("1 per minute")
async def train_ml_model(
    request: Request,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Train the ML model with telemetry data

    Expected JSON payload:
    {
        "training_data": [
            [feature1, feature2, ...],
            [feature1, feature2, ...],
            ...
        ]
    }
    """
    try:
        request_data = await request.json()

        if not request_data or 'training_data' not in request_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No training data provided"
            )

        training_data = request_data['training_data']

        if not isinstance(training_data, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="training_data must be a list"
            )

        # Train the model
        success = anomaly_detector.train_model(training_data)

        if success:
            return {
                'status': 'success',
                'message': 'ML model trained successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to train ML model"
            )

    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'train_ml_endpoint'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Telemetry Export Endpoint
@app.get("/telemetry/export", response_model=Dict[str, Any])
@limiter.limit("2 per minute")
async def export_telemetry(
    request: Request,
    operation: Optional[str] = None,
    limit: int = 1000,
    format: str = "json",
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Export telemetry events

    Query parameters:
    - operation: Filter by operation (optional)
    - limit: Maximum number of events (default: 1000)
    - format: Export format (json, csv) (default: json)
    """
    try:
        if limit <= 0 or limit > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 10000"
            )

        if format.lower() not in ['json', 'csv']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Format must be json or csv"
            )

        # Get events from telemetry handler (mock for now)
        events = []  # This would integrate with actual telemetry handler

        if format.lower() == 'csv':
            # Convert to CSV format
            if not events:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No events found"
                )

            import csv
            import io

            output = io.StringIO()
            if events:
                fieldnames = events[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(events)

            return Response(
                content=output.getvalue(),
                media_type='text/csv',
                headers={"Content-Disposition": "attachment; filename=telemetry_export.csv"}
            )

        return {
            'status': 'success',
            'events': events,
            'count': len(events),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'export_endpoint'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Dashboard Endpoint
@app.get("/dashboard")
async def dashboard():
    """Serve the web dashboard"""
    try:
        dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return Response(
                content=f.read(),
                media_type='text/html'
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard file not found"
        )

# Data Conversion Endpoint
@app.post("/data/convert")
@limiter.limit("5 per minute")
async def convert_data_format(
    request: Request
):
    """
    Convert data between different formats

    Expected JSON payload:
    {
        "data": [...],  // Data to convert
        "from_format": "json",
        "to_format": "csv",
        "options": {...}  // Optional conversion options
    }
    """
    try:
        request_data = await request.json()

        if not request_data or 'data' not in request_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for conversion"
            )

        data = request_data['data']
        from_format = request_data.get('from_format', 'json').lower()
        to_format = request_data.get('to_format', 'json').lower()
        options = request_data.get('options', {})

        if not isinstance(data, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data must be a list of records"
            )

        # Basic format validation
        supported_formats = ['json', 'csv', 'xml', 'yaml']
        if from_format not in supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported import format. Supported formats: {supported_formats}"
            )

        if to_format not in supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported export format. Supported formats: {supported_formats}"
            )

        # For now, return the data as-is (would integrate with DataFormatConverter)
        result = data
        content_type = 'application/json'

        if to_format == 'csv':
            # Simple CSV conversion
            if data:
                import csv
                import io
                output = io.StringIO()
                fieldnames = data[0].keys() if isinstance(data[0], dict) else ['value']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                result = output.getvalue()
                content_type = 'text/csv'

        return Response(
            content=result if isinstance(result, str) else json.dumps(result),
            media_type=content_type
        )

    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'data_conversion_endpoint'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Organization Management Endpoints
@app.get("/organizations", response_model=Dict[str, Any])
async def list_organizations(current_user: str = Depends(require_auth)):
    """
    List all organizations
    """
    try:
        organizations = await async_db_manager.get_all_organizations()
        return {
            'status': 'success',
            'organizations': [OrganizationResponse.from_orm(org).dict() for org in organizations],
            'count': len(organizations),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_organizations'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/organizations", response_model=Dict[str, Any])
async def create_organization(
    organization_data: OrganizationCreate,
    current_user: str = Depends(require_auth)
):
    """
    Create a new organization
    """
    try:
        # Check if user already owns an organization
        existing_org = await async_db_manager.get_organization_by_owner_id(current_user)
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already owns an organization"
            )

        org_data = organization_data.dict()
        org_data['owner_id'] = current_user

        organization = await async_db_manager.create_organization(org_data)

        # Add creator as owner member
        await async_db_manager.add_organization_member({
            'organization_id': organization.id,
            'user_id': current_user,
            'role': 'owner'
        })

        return {
            'status': 'success',
            'organization': OrganizationResponse.from_orm(organization).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_organization'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/organizations/{organization_id}", response_model=Dict[str, Any])
async def get_organization(
    organization_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Get organization details by ID
    """
    try:
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check if user is a member or owner
        user_role = await async_db_manager.get_member_role(organization_id, current_user)
        if user_role is None and organization.owner_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        return {
            'status': 'success',
            'organization': OrganizationResponse.from_orm(organization).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_organization'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.put("/organizations/{organization_id}", response_model=Dict[str, Any])
async def update_organization(
    organization_id: int,
    organization_data: OrganizationUpdate,
    current_user: str = Depends(require_auth)
):
    """
    Update organization details
    """
    try:
        # Check if user is owner
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        if organization.owner_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owner can update organization"
            )

        updated_org = await async_db_manager.update_organization(organization_id, organization_data.dict(exclude_unset=True))
        if not updated_org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        return {
            'status': 'success',
            'organization': OrganizationResponse.from_orm(updated_org).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_organization'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.delete("/organizations/{organization_id}", response_model=Dict[str, Any])
async def delete_organization(
    organization_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Delete an organization
    """
    try:
        # Check if user is owner
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        if organization.owner_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owner can delete organization"
            )

        success = await async_db_manager.delete_organization(organization_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        return {
            'status': 'success',
            'message': 'Organization deleted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_organization'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/organizations/{organization_id}/members", response_model=Dict[str, Any])
async def get_organization_members(
    organization_id: int,
    current_user: str = Depends(require_auth)
):
    """
    Get organization members
    """
    try:
        # Check if user is a member
        user_role = await async_db_manager.get_member_role(organization_id, current_user)
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if user_role is None and (not organization or organization.owner_id != current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        members = await async_db_manager.get_organization_members(organization_id)
        return {
            'status': 'success',
            'members': [OrganizationMemberResponse.from_orm(member).dict() for member in members],
            'count': len(members),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_organization_members'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/organizations/{organization_id}/members", response_model=Dict[str, Any])
async def add_organization_member(
    organization_id: int,
    member_data: OrganizationMemberCreate,
    current_user: str = Depends(require_auth)
):
    """
    Add a member to an organization
    """
    try:
        # Check if user is owner or admin
        user_role = await async_db_manager.get_member_role(organization_id, current_user)
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if (user_role not in ['owner', 'admin']) and (not organization or organization.owner_id != current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owner or admin can add members"
            )

        # Check if user is already a member
        existing_role = await async_db_manager.get_member_role(organization_id, member_data.user_id)
        if existing_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member of this organization"
            )

        member = await async_db_manager.add_organization_member({
            'organization_id': organization_id,
            'user_id': member_data.user_id,
            'role': member_data.role
        })

        return {
            'status': 'success',
            'member': OrganizationMemberResponse.from_orm(member).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'add_organization_member'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.put("/organizations/{organization_id}/members/{user_id}", response_model=Dict[str, Any])
async def update_member_role(
    organization_id: int,
    user_id: str,
    member_data: OrganizationMemberUpdate,
    current_user: str = Depends(require_auth)
):
    """
    Update a member's role in an organization
    """
    try:
        # Check if user is owner or admin
        user_role = await async_db_manager.get_member_role(organization_id, current_user)
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if (user_role not in ['owner', 'admin']) and (not organization or organization.owner_id != current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owner or admin can update member roles"
            )

        success = await async_db_manager.update_member_role(organization_id, user_id, member_data.role)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found"
            )

        return {
            'status': 'success',
            'message': 'Member role updated successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_member_role'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.delete("/organizations/{organization_id}/members/{user_id}", response_model=Dict[str, Any])
async def remove_organization_member(
    organization_id: int,
    user_id: str,
    current_user: str = Depends(require_auth)
):
    """
    Remove a member from an organization
    """
    try:
        # Check if user is owner or admin, or removing themselves
        user_role = await async_db_manager.get_member_role(organization_id, current_user)
        organization = await async_db_manager.get_organization_by_id(organization_id)
        if (user_role not in ['owner', 'admin']) and (not organization or organization.owner_id != current_user) and current_user != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owner or admin can remove members"
            )

        # Cannot remove the owner
        if organization and organization.owner_id == user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove organization owner"
            )

        success = await async_db_manager.remove_organization_member(organization_id, user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found"
            )

        return {
            'status': 'success',
            'message': 'Member removed successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'remove_organization_member'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/user/convert-to-organization", response_model=Dict[str, Any])
async def convert_user_to_organization(
    conversion_data: ConvertUserToOrganizationRequest,
    current_user: str = Depends(require_auth)
):
    """
    Convert a user account to an organization
    """
    try:
        # Check if user already owns an organization
        existing_org = await async_db_manager.get_organization_by_owner_id(current_user)
        if existing_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already owns an organization"
            )

        organization = await async_db_manager.convert_user_to_organization(
            current_user,
            conversion_data.dict()
        )

        return {
            'status': 'success',
            'organization': OrganizationResponse.from_orm(organization).dict(),
            'message': 'User account successfully converted to organization',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'convert_user_to_organization'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def index():
    """Root endpoint for API information"""
    return {
        'message': 'Welcome to JPMorgan Financial APIs - Async Edition',
        'version': get_version(),
        'description': 'Enterprise-grade API for telemetry processing, ML anomaly detection, cloud integration, business asset management, and AI-powered financial insights. Built with FastAPI for high performance.',
        'endpoints': [
            '/health - Health check',
            '/auth/login - Auth0 login URL',
            '/auth/callback - Auth0 callback',
            '/auth/userinfo - Current user info (Auth0)',
            '/auth/logout - Auth0 logout',
            '/user/register - User registration (legacy)',
            '/user/login - User login (legacy)',
            '/user/profile - User profile (requires token)',
            '/telemetry - Process telemetry events',
            '/telemetry/batch - Batch telemetry processing',
            '/telemetry/metrics - Telemetry metrics',
            '/telemetry/export - Export telemetry data',
            '/ml/anomalies - ML anomaly detection',
            '/ml/train - Train ML model',
            '/data/convert - Data format conversion',
            '/businesses - Business management (CRUD) - Auth0 required',
            '/assets - Asset management (CRUD)',
            '/businesses/{id}/assets - Business-asset relationships',
            '/ai/analyze - AI-powered financial data analysis',
            '/ai/risk-assess - AI transaction risk assessment',
            '/ai/query - Natural language financial queries',
            '/ai/status - AI service status',
            '/dashboard - Web dashboard',
            '/welcome/create-workspace - Create workspace page (Auth0 required)',
            '/api/workspaces - Create workspace API (Auth0 required)',
            '/api/github/orgs - Get GitHub organizations',
            '/api/github/repos - Get GitHub repositories'
        ],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        media_type="text/plain; charset=utf-8",
        content=generate_latest()
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": "error",
            "path": str(request.url),
            "method": request.method
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "status": "error",
            "path": str(request.url),
            "method": request.method
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    telemetry_logger.log_error(exc, {
        'context': 'fastapi_error_handler',
        'path': str(request.url),
        'method': request.method
    })
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status": "error",
            "path": str(request.url),
            "method": request.method
        }
    )

# Async utility functions
async def process_telemetry_event_async(telemetry_data: Dict[str, Any]) -> bool:
    """Process a single telemetry event asynchronously"""
    try:
        # This would integrate with the async telemetry handler
        # For now, return success
        return True
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'process_telemetry_event_async'})
        return False

async def process_telemetry_batch_async(telemetry_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a batch of telemetry events asynchronously"""
    try:
        # This would integrate with the async telemetry handler
        # For now, return mock stats
        return {
            'total': len(telemetry_data_list),
            'successful': len(telemetry_data_list),
            'failed': 0
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'process_telemetry_batch_async'})
        return {
            'total': len(telemetry_data_list),
            'successful': 0,
            'failed': len(telemetry_data_list)
        }

async def get_telemetry_metrics_async(hours: int) -> Dict[str, Any]:
    """Get telemetry metrics asynchronously"""
    try:
        # This would integrate with the async telemetry handler
        # For now, return mock metrics
        return {
            'total_events': 0,
            'operation_counts': {},
            'device_counts': {},
            'time_period_hours': hours,
            'avg_events_per_hour': 0
        }
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_telemetry_metrics_async'})
        return {}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    telemetry_logger.get_logger().info("Starting FastAPI Telemetry API Server")

    # Initialize database
    try:
        await async_db_manager.initialize_database()
        telemetry_logger.get_logger().info("Database initialized successfully")
    except Exception as e:
        telemetry_logger.get_logger().error(f"Failed to initialize database: {e}")

    # Print configuration
    telemetry_logger.get_logger().info(f"Configuration: {config.get_all_settings()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    telemetry_logger.get_logger().info("Shutting down FastAPI Telemetry API Server")

    # Close database connections
    try:
        await async_db_manager.close()
    except Exception as e:
        telemetry_logger.get_logger().error(f"Error closing database: {e}")

    # Close Redis connection
    if redis_client:
        try:
            await redis_client.close()
        except Exception as e:
            telemetry_logger.get_logger().error(f"Error closing Redis: {e}")
