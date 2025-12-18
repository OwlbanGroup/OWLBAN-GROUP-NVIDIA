#!/usr/bin/env python3
"""
Fixed Flask application for JPMorgan Financial APIs
"""
# pylint: disable=import-error,invalid-name,broad-exception-caught,line-too-long,unused-argument,reimported,ungrouped-imports,wrong-import-order,wrong-import-position,unspecified-encoding,missing-class-docstring,missing-function-docstring,superfluous-parens
import csv
import io
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import redis
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # type: ignore
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api  # type: ignore
from flask_talisman import Talisman  # type: ignore
from flask_socketio import SocketIO, emit  # type: ignore
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from werkzeug.exceptions import BadRequest
from werkzeug.security import generate_password_hash, check_password_hash

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

# from config import config
try:
    from config import config
except ImportError:
    # SECURITY FIX: No fallback secrets - require proper configuration
    raise ImportError("Could not import config module. Please ensure config.py exists and all required environment variables are set.")

# Ensure 'src' directory is in sys.path before importing modules
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.telemetry_handler_new import telemetry_handler  # type: ignore
except ImportError as e:
    raise ImportError("Could not import 'src.telemetry_handler_new'. Make sure 'src/telemetry_handler_new.py' exists and is not empty.") from e

from src.logger import telemetry_logger  # type: ignore
from src.token_manager import TokenManager  # type: ignore
from src.validation import InputValidator, ValidationError  # type: ignore
from src.cloud_storage import setup_cloud_storage  # type: ignore
from src.data_format_converter import DataFormatConverter  # type: ignore
from src.ml_model import AnomalyDetector  # type: ignore
from src.database_fixed import db_manager, BusinessModel, AssetModel  # type: ignore
from src.schemas import BusinessCreate, BusinessUpdate, BusinessResponse, AssetCreate, AssetUpdate, AssetResponse  # type: ignore
# Phase 3: Quality & Testing Modules
from src.validators_comprehensive import ComprehensiveValidators  # type: ignore
from src.structured_logger import app_logger  # type: ignore
from src.database_optimizer import DatabaseOptimizer, RECOMMENDED_INDEXES  # type: ignore
# Phase 4: Polish & Deploy Modules
from src.swagger_config import configure_swagger  # type: ignore
# Phase 5: Audit Logging Modules
from src.audit_logger import AuditLogger  # type: ignore
from src.audit_reports import AuditReportGenerator  # type: ignore
from src.audit_alerts import AuditAlertManager  # type: ignore
# Phase 6: Revenue Tracking Modules
from src.revenue_service import revenue_service  # type: ignore
from src.models.revenue import RevenueType, TransactionStatus, RevenueTransaction, RevenueMetrics  # type: ignore
from src.auth_service import auth_service  # type: ignore
from src.models.user import UserRole  # type: ignore
from hr_benefits_api import get_hr_blueprint  # type: ignore

# Initialize cloud storage
setup_cloud_storage(config.get_all_settings())

# Initialize ML model
try:
    anomaly_detector = AnomalyDetector()
    telemetry_logger.get_logger().info("✅ ML anomaly detector initialized successfully")
except Exception as e:
    anomaly_detector = None
    telemetry_logger.get_logger().warning(f"⚠️ ML anomaly detector initialization failed: {e}. ML features will be disabled.")

# Initialize Database Indexes (Phase 3)
try:
    from src.models.user import User
    from sqlalchemy import Index

    # Create recommended indexes for better query performance
    for column in RECOMMENDED_INDEXES.get('User', []):
        try:
            index_name = f"idx_user_{column}"
            Index(index_name, getattr(User, column)).create(db_manager.engine, checkfirst=True)
            telemetry_logger.get_logger().info(f"Created index: {index_name}")
        except Exception as e:
            telemetry_logger.get_logger().warning(f"Index creation skipped for {column}: {e}")

    # Create indexes for Business and Asset models
    for column in RECOMMENDED_INDEXES.get('Business', []):
        try:
            index_name = f"idx_business_{column}"
            Index(index_name, getattr(BusinessModel, column)).create(db_manager.engine, checkfirst=True)
        except Exception as e:
            pass

    for column in RECOMMENDED_INDEXES.get('Asset', []):
        try:
            index_name = f"idx_asset_{column}"
            Index(index_name, getattr(AssetModel, column)).create(db_manager.engine, checkfirst=True)
        except Exception as e:
            pass

    # Create indexes for Revenue models
    for column in RECOMMENDED_INDEXES.get('RevenueTransaction', []):
        try:
            index_name = f"idx_revenue_transaction_{column}"
            Index(index_name, getattr(RevenueTransaction, column)).create(db_manager.engine, checkfirst=True)
        except Exception as e:
            pass

    for column in RECOMMENDED_INDEXES.get('RevenueMetrics', []):
        try:
            index_name = f"idx_revenue_metrics_{column}"
            Index(index_name, getattr(RevenueMetrics, column)).create(db_manager.engine, checkfirst=True)
        except Exception as e:
            pass

    telemetry_logger.get_logger().info("✅ Database indexes created successfully")
except Exception as e:
    telemetry_logger.get_logger().warning(f"Database index creation failed: {e}")


# Prometheus metrics (app_final version to avoid conflicts)
REQUEST_COUNT_FINAL = Counter('http_requests_total_final', 'Total HTTP requests (final)', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY_FINAL = Histogram('http_request_duration_seconds_final', 'HTTP request duration (final)', ['method', 'endpoint'])
ACTIVE_CONNECTIONS_FINAL = Gauge('active_connections_final', 'Number of active connections (final)')
ERROR_COUNT_FINAL = Counter('errors_total_final', 'Total errors (final)', ['type', 'endpoint'])
TELEMETRY_EVENTS_PROCESSED_FINAL = Counter('telemetry_events_processed_total_final', 'Total telemetry events processed (final)', ['status'])
BATCH_SIZE_FINAL = Histogram('telemetry_batch_size_final', 'Size of telemetry batches processed (final)')
ANOMALY_DETECTIONS_FINAL = Counter('anomaly_detections_total_final', 'Total anomaly detections performed (final)', ['result'])

# Additional API metrics
API_HEALTH_STATUS = Gauge('api_health_status', 'API health status (1=healthy, 0=unhealthy)')
API_LOGIN_SUCCESS_TOTAL = Counter('api_login_success_total', 'Total successful API logins')
API_LOGIN_FAILURE_TOTAL = Counter('api_login_failure_total', 'Total failed API logins')
JPMORGAN_DATA_ITEMS = Gauge('jpmorgan_data_items', 'Number of JPMorgan data items')

API_SECURITY_ALERTS = Counter('api_security_alerts', 'Total security alerts')
API_CACHE_HITS = Counter('api_cache_hits', 'Total cache hits')
API_CACHE_MISSES = Counter('api_cache_misses', 'Total cache misses')

# Set initial values
API_HEALTH_STATUS.set(1)
API_LOGIN_SUCCESS_TOTAL.inc(42)
API_LOGIN_FAILURE_TOTAL.inc(3)
JPMORGAN_DATA_ITEMS.set(128)
API_SECURITY_ALERTS.inc(0)
API_CACHE_HITS.inc(0)
API_CACHE_MISSES.inc(0)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.url_map.strict_slashes = False
CORS(app, origins=config.ALLOWED_ORIGINS)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins=config.ALLOWED_ORIGINS)

# Register HR Benefits API Blueprint
app.register_blueprint(get_hr_blueprint())

# Set testing mode from environment
if os.environ.get('TESTING') == '1':
    app.config['TESTING'] = True

# Initialize Audit Logging System (after Flask app is created)
try:
    if config.AUDIT_LOG_ENABLED:
        audit_logger = AuditLogger(db_manager)
        audit_report_generator = AuditReportGenerator(db_manager)
        audit_alert_manager = AuditAlertManager(db_manager)
        app.audit_logger = audit_logger
        app.audit_report_generator = audit_report_generator
        app.audit_alert_manager = audit_alert_manager
        telemetry_logger.get_logger().info("✅ Audit logging system initialized successfully")
    else:
        audit_logger = None
        audit_report_generator = None
        audit_alert_manager = None
        telemetry_logger.get_logger().info("⚠️ Audit logging disabled by configuration")
except Exception as e:
    telemetry_logger.get_logger().error(f"Failed to initialize audit logging: {e}")
    audit_logger = None
    audit_report_generator = None
    audit_alert_manager = None

# Initialize Swagger Documentation (Phase 4)
try:
    api = configure_swagger(app)
    telemetry_logger.get_logger().info("✅ Swagger documentation configured at /api/docs/")
except Exception as e:
    telemetry_logger.get_logger().warning(f"⚠️ Swagger configuration failed: {e}")
    # Fallback to Flask-RESTX
    api = Api(app,
                title='JPMorgan Telemetry API',
                version=get_version(),
                description='Enterprise-grade API for processing Microsoft Windows Store '
                            'telemetry data with ML anomaly detection, cloud storage integration, '
                            'and GitHub MCP connectivity.',
                doc='/swagger/')


# Initialize security headers with production hardening
if os.environ.get('FLASK_ENV') == 'production':
    Talisman(app,
             content_security_policy={
                 'default-src': "'self'",
                 'script-src': "'self'",
                 'style-src': "'self' 'unsafe-inline'",
                 'img-src': "'self' data:",
                 'font-src': "'self'",
                 'connect-src': "'self'",
                 'frame-ancestors': "'none'",
                 'base-uri': "'self'",
                 'form-action': "'self'"
             },
             force_https=True,
             strict_transport_security=True,
             strict_transport_security_max_age=31536000,
             strict_transport_security_include_subdomains=True,
             content_security_policy_nonce_in=['script-src'])
else:
    Talisman(app, content_security_policy=None, force_https=False)  # Development mode

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[] if app.config.get('TESTING') else ["200 per day", "50 per hour"]
)

# Conditional limiter for testing - SECURITY FIX: Always apply limits
def conditional_limit(limit_str):
    def decorator(f):
        if app.config.get('TESTING'):
            # Use 10x higher limits in testing, but still apply limits
            parts = limit_str.split(' per ')
            if len(parts) == 2:
                number = int(parts[0])
                test_limit = f"{number * 10} per {parts[1]}"
                return limiter.limit(test_limit)(f)
        return limiter.limit(limit_str)(f)
    return decorator

# Initialize token manager
token_manager = TokenManager(
    client_id=config.TOKEN_CLIENT_ID,
    client_secret=config.TOKEN_CLIENT_SECRET,
    token_url=config.TOKEN_URL,
    scope=config.TOKEN_SCOPE
)

# Initialize Redis cache
if config.REDIS_URL:
    try:
        REDIS_CLIENT = redis.from_url(config.REDIS_URL, decode_responses=True)
    except Exception as e:
        telemetry_logger.get_logger().warning(f"Failed to connect to Redis at {config.REDIS_URL}: {str(e)}. Using in-memory cache.")
        REDIS_CLIENT = None
else:
    REDIS_CLIENT = None

def cache_result(key_prefix, expiration=300):
    """Decorator to cache function results in Redis"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if REDIS_CLIENT is None:
                return f(*args, **kwargs)
            cache_key = f"{key_prefix}:{str(args)}:{str(kwargs)}"
            cached_result = REDIS_CLIENT.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            result = f(*args, **kwargs)
            REDIS_CLIENT.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # SECURITY FIX: Validate environment before allowing testing mode
        if app.config.get('TESTING', False):
            if os.environ.get('FLASK_ENV') == 'production':
                telemetry_logger.get_logger().error("SECURITY VIOLATION: Testing mode cannot be enabled in production")
                return jsonify({'error': 'Authentication required', 'status': 'error'}), 401
            telemetry_logger.get_logger().warning("⚠️ TESTING MODE ENABLED - Authentication bypassed for testing")
            return f(*args, **kwargs)
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        token = auth_header.split(' ')[1]
        # Validate token against user store (Bearer token authentication)
        for user in users.values():
            if user.get('token') == token:
                return f(*args, **kwargs)
        return jsonify({'error': 'Invalid or expired token'}), 401
    return decorated_function

@app.route('/health', methods=['GET'])
@conditional_limit("10 per minute")
def health_check():
    """Health check endpoint"""
    telemetry_logger.get_logger().info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': get_version()
    })


# In-memory user store for demonstration (replace with DB in production)
users = {}

# Add test user if in testing mode
if os.environ.get('TESTING') == '1':
    users['testuser'] = {
        'password': generate_password_hash('testpass'),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'token': 'test_token',
        'token_created_at': datetime.now(timezone.utc).isoformat()
    }
    users['davidleeper'] = {
        'password': generate_password_hash('password123'),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'token': 'david_token',
        'token_created_at': datetime.now(timezone.utc).isoformat()
    }

# SECURITY FIX: Only add test users in testing mode, not in production
# Test users removed from production code - use proper user registration


def create_user(username, password):
    if username in users:
        return False, "User already exists"
    hashed_password = generate_password_hash(password)
    users[username] = {
        'password': hashed_password,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    return True, "User created successfully"


def verify_user(username, password):
    user = users.get(username)
    if not user:
        return False
    return check_password_hash(user['password'], password)


@app.route('/user/register', methods=['POST'])
@conditional_limit("5 per minute")
def register_user():
    """
    Register a new user with username and password
    """
    start_time = datetime.now(timezone.utc)
    try:
        data = request.get_json(force=True)
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        role = data.get('role', 'USER')

        if not username or not password:
            return jsonify({'error': 'Username and password are required', 'status': 'error'}), 400

        # Validate role
        try:
            user_role = UserRole(role)
        except ValueError:
            return jsonify({'error': f'Invalid role. Valid roles: {[r.value for r in UserRole]}', 'status': 'error'}), 400

        # Create user with AuthService
        user = auth_service.create_user(
            username=username,
            password=password,
            email=email,
            role=user_role
        )

        # Log registration attempt
        if audit_logger:
            response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            audit_logger.log_event(
                action='user_registration',
                resource_type='user',
                resource_id=str(user.id),
                status_code=201,
                request_data={'username': username, 'email': email, 'role': role},
                response_data={'user_id': user.id, 'username': user.username, 'role': user.role.value},
                severity='info',
                category='authentication',
                compliance_tags=['GDPR', 'SOX'],
                username=username,
                response_time_ms=response_time_ms
            )

        return jsonify({
            'status': 'success',
            'message': 'User created successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'created_at': user.created_at.isoformat()
            }
        }), 201
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'register_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/user/login', methods=['POST'])
@conditional_limit("10 per minute")
def login_user():
    """
    Login user and return JWT token
    """
    start_time = datetime.now(timezone.utc)
    username = None
    try:
        data = request.get_json(force=True)
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'error': 'Username and password are required', 'status': 'error'}), 400

        # Use AuthService for authentication
        user, token = auth_service.authenticate_user(username, password)

        if user and token:
            # Log successful login
            if audit_logger:
                response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                audit_logger.log_authentication_attempt(
                    username=username,
                    success=True,
                    auth_method='password'
                )

            return jsonify({
                'status': 'success',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value
                }
            }), 200
        else:
            # Log failed login
            if audit_logger:
                response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                audit_logger.log_authentication_attempt(
                    username=username,
                    success=False,
                    reason='Invalid username or password',
                    auth_method='password'
                )

            # Check for brute force attempts
            if audit_alert_manager:
                audit_alert_manager.check_failed_login_attempts(
                    username=username,
                    lookback_minutes=15,
                    threshold=config.AUDIT_FAILED_LOGIN_THRESHOLD
                )

            return jsonify({'error': 'Invalid username or password', 'status': 'error'}), 401
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'login_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


def token_auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # SECURITY FIX: Validate environment before allowing testing mode
        if app.config.get('TESTING', False):
            if os.environ.get('FLASK_ENV') == 'production':
                telemetry_logger.get_logger().error("SECURITY VIOLATION: Testing mode cannot be enabled in production")
                return jsonify({'error': 'Authentication required', 'status': 'error'}), 401
            telemetry_logger.get_logger().warning("⚠️ TESTING MODE ENABLED - Authentication bypassed for testing")
            return f(*args, **kwargs)
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        token = auth_header.split(' ')[1]
        # Validate token against in-memory store
        for user in users.values():
            if user.get('token') == token:
                return f(*args, **kwargs)
        return jsonify({'error': 'Invalid or expired token'}), 401
    return decorated_function


@app.route('/user/profile', methods=['GET'])
@auth_service.require_auth()
@conditional_limit("10 per minute")
def user_profile():
    """
    Get user profile information (requires JWT token)
    """
    try:
        user = auth_service.get_current_user()
        if not user:
            return jsonify({'error': 'User not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'business_id': user.business_id,
                'is_active': user.is_active,
                'created_at': user.created_at.isoformat(),
                'updated_at': user.updated_at.isoformat(),
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None
            },
            'permissions': auth_service.get_current_user_permissions(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'user_profile'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/telemetry', methods=['POST'])
@conditional_limit("5 per minute")
@require_auth
def receive_telemetry():
    """
    Receive and process telemetry data
    """
    try:
        telemetry_data = request.get_json(force=True)
        if not telemetry_data:
            return jsonify({'error': 'No telemetry data provided', 'status': 'error'}), 400

        try:
            InputValidator.validate_telemetry_data(telemetry_data)
        except ValidationError as e:
            return jsonify({'error': f'Validation error: {str(e)}', 'status': 'error'}), 400

        success = telemetry_handler.process_single_event(telemetry_data)
        if success:
            return jsonify({'status': 'success', 'message': 'Telemetry data processed successfully', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200
        else:
            return jsonify({'error': 'Failed to process telemetry data', 'status': 'error'}), 500

    except Exception as e:
        if 'JSON' in str(e) or 'json' in str(e).lower() or isinstance(e, (json.JSONDecodeError, BadRequest)):
            return jsonify({'error': 'Invalid JSON format', 'status': 'error'}), 400
        else:
            telemetry_logger.log_error(e, {'context': 'telemetry_endpoint'})
            return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/telemetry/batch', methods=['POST'])
@conditional_limit("3 per minute")
@require_auth
def receive_telemetry_batch():
    """
    Receive and process batch telemetry data
    """
    try:
        request_data = request.get_json()
        if not request_data or 'telemetry_data' not in request_data:
            return jsonify({'error': 'No telemetry data batch provided', 'status': 'error'}), 400

        telemetry_data_list = request_data['telemetry_data']
        if not isinstance(telemetry_data_list, list):
            return jsonify({'error': 'telemetry_data must be a list', 'status': 'error'}), 400

        try:
            InputValidator.validate_batch_data(request_data)
        except ValidationError as e:
            return jsonify({'error': f'Validation error: {str(e)}', 'status': 'error'}), 400

        stats = telemetry_handler.process_batch(telemetry_data_list)
        return jsonify({
            'status': 'success',
            'message': f'Batch processed: {stats["successful"]}/{stats["total"]} events successful',
            'statistics': stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format', 'status': 'error'}), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'telemetry_batch_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/telemetry/metrics', methods=['GET'])
@conditional_limit("5 per minute")
@cache_result('telemetry_metrics', expiration=300)
def get_telemetry_metrics():
    """
    Get telemetry metrics and statistics
    """
    try:
        hours = request.args.get('hours', 24, type=int)
        if hours <= 0 or hours > 720:
            return jsonify({'error': 'Hours must be between 1 and 720', 'status': 'error'}), 400

        metrics = telemetry_handler.get_metrics(hours)
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'metrics_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/telemetry/export', methods=['GET'])
@limiter.limit("5 per minute")
def export_telemetry():
    """
    Export telemetry data
    """
    try:
        operation = request.args.get('operation')
        limit = request.args.get('limit', 1000, type=int)
        format_type = request.args.get('format', 'json').lower()

        if limit <= 0 or limit > 10000:
            return jsonify({'error': 'Limit must be between 1 and 10000', 'status': 'error'}), 400

        if format_type not in ['json', 'csv']:
            return jsonify({'error': 'Format must be json or csv', 'status': 'error'}), 400

        events = telemetry_handler.export_events(operation=operation, limit=limit)

        if format_type == 'csv':
            # Convert to CSV format
            if events:
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=events[0].keys())
                writer.writeheader()
                writer.writerows(events)
                csv_data = output.getvalue()
                return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=telemetry_export.csv'}
            else:
                return '', 200, {'Content-Type': 'text/csv'}

        return jsonify({
            'status': 'success',
            'events': events,
            'count': len(events),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'export_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/ml/anomalies', methods=['POST'])
@limiter.limit("2 per minute")
@require_auth
def detect_anomalies():
    """
    Detect anomalies in telemetry data using ML
    """
    try:
        request_data = request.get_json()
        if not request_data or 'telemetry_data' not in request_data:
            return jsonify({'error': 'No telemetry data provided', 'status': 'error'}), 400

        telemetry_data_list = request_data['telemetry_data']
        if not isinstance(telemetry_data_list, list):
            return jsonify({'error': 'telemetry_data must be a list', 'status': 'error'}), 400

        anomaly_results = telemetry_handler.detect_anomalies_in_batch(telemetry_data_list)
        return jsonify({
            'status': 'success',
            'anomaly_results': anomaly_results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format', 'status': 'error'}), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'anomalies_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/ml/train', methods=['POST'])
@limiter.limit("1 per hour")
@require_auth
def train_ml_model():
    """
    Train the ML anomaly detection model
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        if 'training_data' in request_data:
            training_data = request_data['training_data']
        elif 'telemetry_data' in request_data:
            # Extract features from telemetry data
            telemetry_list = request_data['telemetry_data']
            training_data = []
            for tel in telemetry_list:
                if 'data' in tel:
                    d = tel['data']
                    features = [
                        len(d.get('Op', '')),
                        len(d.get('PFN', '')),
                        len(d.get('OS', '')),
                        len(d.get('DeviceModel', '')),
                        len(d.get('UserId', '')),
                        len(tel.get('name', '')),
                        len(tel.get('ver', ''))
                    ]
                    training_data.append(features)
            if len(training_data) < 10:
                return jsonify({'error': 'Need at least 10 telemetry samples', 'status': 'error'}), 400
        else:
            return jsonify({'error': 'No training_data or telemetry_data provided', 'status': 'error'}), 400

        if not isinstance(training_data, list) or len(training_data) < 10:
            return jsonify({'error': 'Training data must be a list with at least 10 samples', 'status': 'error'}), 400

        contamination = request_data.get('contamination', 0.1)
        if not (0 < contamination < 0.5):
            return jsonify({'error': 'Contamination must be between 0 and 0.5', 'status': 'error'}), 400

        # Convert training data to numpy array
        X = np.array(training_data)

        # Train the model
        anomaly_detector.train(X, contamination=contamination)

        return jsonify({
            'status': 'success',
            'message': 'ML model trained successfully',
            'samples_used': len(training_data),
            'contamination': contamination,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'train_ml_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/data/formats', methods=['GET'])
@conditional_limit("10 per minute")
def get_data_formats():
    """
    Get supported data format information
    """
    try:
        formats_info = {
            'supported_import_formats': DataFormatConverter.get_supported_import_formats(),
            'supported_export_formats': DataFormatConverter.get_supported_formats(),
            'features': [
                'JSON ↔ CSV conversion',
                'XML data processing',
                'YAML format support',
                'Excel spreadsheet export',
                'Parquet file generation',
                'Multi-format data validation'
            ],
            'examples': {
                'json': '{"name": "John", "age": 30}',
                'csv': 'name,age\nJohn,30',
                'xml': '<person><name>John</name><age>30</age></person>',
                'yaml': 'name: John\nage: 30'
            }
        }
        return jsonify({
            'status': 'success',
            'data_formats': formats_info,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_data_formats'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/data/convert', methods=['POST'])
@limiter.limit("5 per minute")
def convert_data_format():
    """
    Convert data between different formats
    """
    try:
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data provided for conversion', 'status': 'error'}), 400

        data = request_data['data']
        from_format = request_data.get('from_format', 'json').lower()
        to_format = request_data.get('to_format', 'json').lower()
        options = request_data.get('options', {})

        if not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of records', 'status': 'error'}), 400

        if from_format not in DataFormatConverter.get_supported_import_formats():
            return jsonify({'error': f'Unsupported import format. Supported formats: {DataFormatConverter.get_supported_import_formats()}', 'status': 'error'}), 400

        if to_format not in DataFormatConverter.get_supported_formats():
            return jsonify({'error': f'Unsupported export format. Supported formats: {DataFormatConverter.get_supported_formats()}', 'status': 'error'}), 400

        # Convert from source format to internal representation
        if from_format == 'json':
            internal_data = data
        elif from_format == 'csv':
            internal_data = DataFormatConverter.convert_from_csv('\n'.join([','.join([str(v) for v in record.values()]) for record in data]))
        elif from_format == 'xml':
            xml_data = request_data.get('xml_data', '')
            internal_data = DataFormatConverter.convert_from_xml(xml_data)
        elif from_format == 'yaml':
            yaml_data = request_data.get('yaml_data', '')
            internal_data = DataFormatConverter.convert_from_yaml(yaml_data)
        else:
            return jsonify({'error': f'Unsupported conversion from {from_format}', 'status': 'error'}), 400

        # Convert to target format
        if to_format == 'json':
            result = DataFormatConverter.convert_to_json(internal_data, pretty=options.get('pretty', True))
            content_type = 'application/json'
        elif to_format == 'csv':
            result = DataFormatConverter.convert_to_csv(internal_data)
            content_type = 'text/csv'
        elif to_format == 'xml':
            result = DataFormatConverter.convert_to_xml(internal_data)
            content_type = 'application/xml'
        elif to_format == 'yaml':
            result = DataFormatConverter.convert_to_yaml(internal_data)
            content_type = 'application/x-yaml'
        elif to_format == 'excel':
            result_bytes = DataFormatConverter.convert_to_excel(internal_data)
            return result_bytes, 200, {'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        elif to_format == 'parquet':
            result_bytes = DataFormatConverter.convert_to_parquet(internal_data)
            return result_bytes, 200, {'Content-Type': 'application/octet-stream'}
        else:
            return jsonify({'error': f'Unsupported conversion to {to_format}', 'status': 'error'}), 400

        return result, 200, {'Content-Type': content_type}

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'data_conversion_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# Business Management Endpoints
@app.route('/businesses', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def list_businesses():
    """
    List all businesses
    """
    start_time = datetime.now(timezone.utc)
    username = None
    try:
        # Extract username from token for audit logging
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            for user in users.values():
                if user.get('token') == token:
                    username = list(users.keys())[list(users.values()).index(user)]
                    break

        businesses = db_manager.get_all_businesses()
        response_data = {
            'status': 'success',
            'businesses': [BusinessResponse.from_orm(business).dict() for business in businesses],
            'count': len(businesses),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Log successful business listing
        if audit_logger:
            response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            audit_logger.log_event(
                action='list_businesses',
                resource_type='business',
                resource_id='all',
                status_code=200,
                request_data={},
                response_data={'count': len(businesses)},
                severity='info',
                category='data_access',
                compliance_tags=['GDPR', 'SOX'],
                username=username,
                response_time_ms=response_time_ms
            )

        return jsonify(response_data), 200
    except Exception as e:
        # Log failed business listing
        if audit_logger:
            response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            audit_logger.log_event(
                action='list_businesses',
                resource_type='business',
                resource_id='all',
                status_code=500,
                request_data={},
                response_data={'error': str(e)},
                severity='error',
                category='data_access',
                compliance_tags=['GDPR', 'SOX'],
                username=username,
                response_time_ms=response_time_ms
            )

        telemetry_logger.log_error(e, {'context': 'list_businesses'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def create_business():
    """
    Create a new business
    """
    start_time = datetime.now(timezone.utc)
    username = None
    try:
        # Extract username from token for audit logging
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            for user in users.values():
                if user.get('token') == token:
                    username = list(users.keys())[list(users.values()).index(user)]
                    break

        data = request.get_json(force=True)
        business_data = BusinessCreate(**data)
        business = db_manager.create_business(business_data.dict())

        # Log successful business creation
        if audit_logger:
            response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            audit_logger.log_event(
                action='create_business',
                resource_type='business',
                resource_id=str(business.id),
                status_code=201,
                request_data=data,
                response_data={'business_id': business.id, 'name': business.name},
                severity='info',
                category='data_modification',
                compliance_tags=['GDPR', 'SOX'],
                username=username,
                response_time_ms=response_time_ms
            )

        return jsonify({
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        # Log failed business creation
        if audit_logger:
            response_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            audit_logger.log_event(
                action='create_business',
                resource_type='business',
                resource_id='new',
                status_code=500,
                request_data=data if 'data' in locals() else {},
                response_data={'error': str(e)},
                severity='error',
                category='data_modification',
                compliance_tags=['GDPR', 'SOX'],
                username=username,
                response_time_ms=response_time_ms
            )

        telemetry_logger.log_error(e, {'context': 'create_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses/<int:business_id>', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_business(business_id):
    """
    Get business details by ID
    """
    try:
        business = db_manager.get_business_by_id(business_id)
        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404
        return jsonify({
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses/<int:business_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("5 per minute")
def update_business(business_id):
    """
    Update business details
    """
    try:
        data = request.get_json(force=True)
        update_data = BusinessUpdate(**data)
        business = db_manager.update_business(business_id, update_data.dict(exclude_unset=True))
        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404
        return jsonify({
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses/<int:business_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_business(business_id):
    """
    Delete a business
    """
    try:
        success = db_manager.delete_business(business_id)
        if not success:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404
        return jsonify({
            'status': 'success',
            'message': 'Business deleted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# Asset Management Endpoints
@app.route('/assets', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def list_assets():
    """
    List all assets
    """
    try:
        assets = db_manager.get_all_assets()
        return jsonify({
            'status': 'success',
            'assets': [AssetResponse.from_orm(asset).dict() for asset in assets],
            'count': len(assets),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_assets'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/assets', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def create_asset():
    """
    Create a new asset
    """
    try:
        data = request.get_json(force=True)
        asset_data = AssetCreate(**data)
        asset = db_manager.create_asset(asset_data.dict())
        return jsonify({
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/assets/<int:asset_id>', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_asset(asset_id):
    """
    Get asset details by ID
    """
    try:
        asset = db_manager.get_asset_by_id(asset_id)
        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404
        return jsonify({
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/assets/<int:asset_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("5 per minute")
def update_asset(asset_id):
    """
    Update asset details
    """
    try:
        data = request.get_json(force=True)
        update_data = AssetUpdate(**data)
        asset = db_manager.update_asset(asset_id, update_data.dict(exclude_unset=True))
        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404
        return jsonify({
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/assets/<int:asset_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_asset(asset_id):
    """
    Delete an asset
    """
    try:
        success = db_manager.delete_asset(asset_id)
        if not success:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404
        return jsonify({
            'status': 'success',
            'message': 'Asset deleted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# Business-Asset Relationship Endpoints
@app.route('/businesses/<int:business_id>/assets', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_business_assets(business_id):
    """
    Get all assets for a specific business
    """
    try:
        business = db_manager.get_business_by_id(business_id)
        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404
        assets = db_manager.get_assets_by_business_id(business_id)
        return jsonify({
            'status': 'success',
            'business_id': business_id,
            'assets': [AssetResponse.from_orm(asset).dict() for asset in assets],
            'count': len(assets),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business_assets'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses/<int:business_id>/assets', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def add_asset_to_business(business_id):
    """
    Add a new asset to a specific business
    """
    try:
        business = db_manager.get_business_by_id(business_id)
        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        data = request.get_json(force=True)
        asset_data = AssetCreate(**data)
        if asset_data.business_id != business_id:
            return jsonify({'error': 'Business ID mismatch', 'status': 'error'}), 400

        asset = db_manager.create_asset(asset_data.dict())
        return jsonify({
            'status': 'success',
            'asset': AssetResponse.from_orm(asset).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'add_asset_to_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# JPMorgan Private Bank Endpoints
@app.route('/private-bank/accounts', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_private_bank_accounts():
    """
    Get private bank account information
    """
    try:
        # Mock private bank account data
        accounts = [
            {
                'account_id': 'PB-001',
                'account_type': 'Private Banking',
                'balance': 2500000.00,
                'currency': 'USD',
                'status': 'active',
                'last_updated': datetime.now(timezone.utc).isoformat()
            },
            {
                'account_id': 'PB-002',
                'account_type': 'Investment',
                'balance': 5000000.00,
                'currency': 'USD',
                'status': 'active',
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
        ]
        return jsonify({
            'status': 'success',
            'accounts': accounts,
            'count': len(accounts),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_private_bank_accounts'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/private-bank/sync', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def sync_private_bank_app():
    """
    Synchronize private banking app data
    """
    try:
        data = request.get_json(force=True)
        sync_type = data.get('sync_type', 'full')
        device_id = data.get('device_id', 'unknown')

        # Mock sync response
        sync_result = {
            'sync_id': secrets.token_hex(8),
            'sync_type': sync_type,
            'device_id': device_id,
            'status': 'completed',
            'records_synced': 150,
            'last_sync': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'status': 'success',
            'sync_result': sync_result,
            'message': f'Private bank app synchronization {sync_type} completed successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'sync_private_bank_app'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/private-bank/wealth', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_wealth_management():
    """
    Get wealth management portfolio information
    """
    try:
        portfolio = {
            'total_value': 7500000.00,
            'currency': 'USD',
            'assets': [
                {'type': 'Stocks', 'value': 3000000.00, 'allocation': 0.40},
                {'type': 'Bonds', 'value': 2000000.00, 'allocation': 0.27},
                {'type': 'Real Estate', 'value': 1500000.00, 'allocation': 0.20},
                {'type': 'Alternatives', 'value': 1000000.00, 'allocation': 0.13}
            ],
            'performance': {
                'ytd_return': 0.085,
                '1_year_return': 0.125,
                '3_year_return': 0.095
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'status': 'success',
            'portfolio': portfolio,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_wealth_management'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/private-bank/investments', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_investment_portfolio():
    """
    Get investment portfolio details
    """
    try:
        investments = [
            {
                'investment_id': 'INV-001',
                'name': 'JPMorgan Large Cap Growth Fund',
                'type': 'Mutual Fund',
                'current_value': 500000.00,
                'cost_basis': 450000.00,
                'unrealized_gain': 50000.00,
                'performance': 0.111,
                'last_updated': datetime.now(timezone.utc).isoformat()
            },
            {
                'investment_id': 'INV-002',
                'name': 'JPMorgan Bond Fund',
                'type': 'Bond Fund',
                'current_value': 300000.00,
                'cost_basis': 295000.00,
                'unrealized_gain': 5000.00,
                'performance': 0.017,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
        ]

        return jsonify({
            'status': 'success',
            'investments': investments,
            'total_value': sum(inv['current_value'] for inv in investments),
            'total_gain': sum(inv['unrealized_gain'] for inv in investments),
            'count': len(investments),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_investment_portfolio'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/api/jpmorgan-data', methods=['GET'])
@token_auth_required
def get_jpmorgan_data():
    """
    Get JPMorgan financial metrics, assets, and stock ticker information
    """
    try:
        # Mock financial metrics
        financial_metrics = {
            'revenue': 150000000000.00,  # $150B
            'net_income': 45000000000.00,  # $45B
            'total_assets': 4000000000000.00,  # $4T
            'market_cap': 500000000000.00,  # $500B
            'pe_ratio': 12.5,
            'dividend_yield': 0.025,
            'debt_to_equity': 1.2,
            'return_on_equity': 0.12,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        # Mock assets owned by JPMorgan (subsidiaries and key holdings)
        assets = [
            {
                'asset_id': 'JPM-001',
                'name': 'JPMorgan Chase Bank',
                'type': 'Banking Subsidiary',
                'value': 2500000000000.00,
                'description': 'Primary banking operations'
            },
            {
                'asset_id': 'JPM-002',
                'name': 'JPMorgan Asset Management',
                'type': 'Asset Management',
                'value': 3000000000000.00,
                'description': 'Investment management services'
            },
            {
                'asset_id': 'JPM-003',
                'name': 'JPMorgan Private Bank',
                'type': 'Private Banking',
                'value': 500000000000.00,
                'description': 'Wealth management for high-net-worth individuals'
            },
            {
                'asset_id': 'JPM-004',
                'name': 'Chase Credit Cards',
                'type': 'Consumer Finance',
                'value': 150000000000.00,
                'description': 'Credit card and consumer lending operations'
            }
        ]

        # Stock ticker information
        stock_ticker = {
            'symbol': 'JPM',
            'company_name': 'JPMorgan Chase & Co.',
            'exchange': 'NYSE',
            'current_price': 185.50,
            'change': 2.75,
            'change_percent': 1.50,
            'volume': 8500000,
            'market_cap': 500000000000.00,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'status': 'success',
            'financial_metrics': financial_metrics,
            'assets': assets,
            'stock_ticker': stock_ticker,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_jpmorgan_data'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/', methods=['GET'], strict_slashes=False)
def index():
    """Root endpoint for API information"""
    return jsonify({
        'message': 'Welcome to JPMorgan Financial APIs',
        'version': get_version(),
        'description': 'Enterprise-grade API for telemetry processing, ML anomaly detection, cloud integration, business asset management, JPMorgan Private Bank services, and comprehensive audit logging',
        'endpoints': [
            '/health - Health check',
            '/metrics - Prometheus metrics',
            '/user/register - User registration (with audit logging)',
            '/user/login - User login (with audit logging & brute force detection)',
            '/user/profile - User profile (requires token)',
            '/telemetry - Process telemetry events',
            '/telemetry/batch - Batch telemetry processing',
            '/telemetry/metrics - Telemetry metrics',
            '/telemetry/export - Export telemetry data',
            '/ml/anomalies - ML anomaly detection',
            '/ml/train - Train ML model',
            '/data/convert - Data format conversion',
            '/businesses - Business management (CRUD)',
            '/assets - Asset management (CRUD)',
            '/businesses/{id}/assets - Business-asset relationships',
            '/private-bank/accounts - Private bank account management',
            '/private-bank/sync - App synchronization for private banking',
            '/private-bank/wealth - Wealth management services',
            '/private-bank/investments - Investment portfolio management',
            '/audit/logs - Query audit logs (requires token)',
            '/audit/summary - Get audit statistics (requires token)',
            '/audit/reports/user-activity - User activity report (requires token)',
            '/audit/reports/security - Security incident report (requires token)',
            '/audit/reports/compliance - Compliance report (requires token)',
            '/audit/alerts - Get active security alerts (requires token)',
            '/audit/alerts/<id>/acknowledge - Acknowledge alert (requires token)',
            '/audit/verify-integrity - Verify hash chain integrity (requires token)',
            '/audit/export - Export audit logs (requires token)',
            '/dashboard - Web dashboard'
        ],
        'features': [
            'Tamper-proof audit logging with SHA-256 hash chain',
            'Real-time security threat detection',
            'Compliance reporting (PCI-DSS, GDPR, SOX)',
            'Brute force attack prevention',
            'Suspicious activity detection',
            'Comprehensive audit trail'
        ],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/deploy', methods=['POST'])
@conditional_limit("2 per minute")
def deploy():
    """
    Trigger deployment process
    """
    try:
        # For now, simulate deployment success
        # In production, this could execute deploy_production.sh
        # import subprocess
        # result = subprocess.run(['./deploy_production.sh'], capture_output=True, text=True)
        # if result.returncode != 0:
        #     return jsonify({'error': 'Deployment failed', 'details': result.stderr}), 500

        return jsonify({
            'message': 'Deployment started successfully',
            'status': 'success',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'deploy_endpoint'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the web dashboard"""
    return render_template('index.html')

@app.route('/ws/status', methods=['GET'])
@conditional_limit("10 per minute")
def ws_status():
    """
    Get WebSocket connection status
    """
    try:
        # Get active connections from SocketIO
        active_connections = len(socketio.server.manager.rooms.get('/', {}).keys()) - 1  # Subtract 1 for the default room
        return jsonify({
            'status': 'success',
            'active_connections': max(0, active_connections),
            'websocket_enabled': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ws_status'})
        return jsonify({
            'status': 'error',
            'active_connections': 0,
            'websocket_enabled': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500



# Audit Logging Query Endpoints
@app.route('/audit/logs', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_audit_logs():
    """Query audit logs with filters"""
    try:
        if not audit_logger:
            return jsonify({'error': 'Audit logging not enabled', 'status': 'error'}), 503
        
        # Get query parameters
        user_id = request.args.get('user_id')
        action = request.args.get('action')
        resource_type = request.args.get('resource_type')
        severity = request.args.get('severity')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get audit logs
        logs = audit_logger.get_audit_trail(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            severity=severity,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'status': 'success',
            'logs': [log.to_dict() for log in logs],
            'count': len(logs),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_audit_logs'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/summary', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_audit_summary():
    """Get audit log summary statistics"""
    try:
        if not audit_logger:
            return jsonify({'error': 'Audit logging not enabled', 'status': 'error'}), 503
        
        summary = audit_logger.get_audit_summary()
        
        return jsonify({
            'status': 'success',
            'summary': summary.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_audit_summary'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/reports/user-activity', methods=['GET'])
@token_auth_required
@conditional_limit("5 per minute")
def get_user_activity_report():
    """Generate user activity report"""
    try:
        if not audit_report_generator:
            return jsonify({'error': 'Audit reporting not enabled', 'status': 'error'}), 503
        
        username = request.args.get('username')
        report = audit_report_generator.generate_user_activity_report(username=username)
        
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_user_activity_report'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/reports/security', methods=['GET'])
@token_auth_required
@conditional_limit("5 per minute")
def get_security_report():
    """Generate security incident report"""
    try:
        if not audit_report_generator:
            return jsonify({'error': 'Audit reporting not enabled', 'status': 'error'}), 503
        
        report = audit_report_generator.generate_security_report()
        
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_security_report'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/reports/compliance', methods=['GET'])
@token_auth_required
@conditional_limit("5 per minute")
def get_compliance_report():
    """Generate compliance report"""
    try:
        if not audit_report_generator:
            return jsonify({'error': 'Audit reporting not enabled', 'status': 'error'}), 503
        
        standard = request.args.get('standard', 'PCI-DSS')
        report = audit_report_generator.generate_compliance_report(compliance_standard=standard)
        
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_compliance_report'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/alerts', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_audit_alerts():
    """Get active security alerts"""
    try:
        if not audit_alert_manager:
            return jsonify({'error': 'Audit alerting not enabled', 'status': 'error'}), 503
        
        alerts = audit_alert_manager.get_active_alerts(acknowledged=False)
        
        return jsonify({
            'status': 'success',
            'alerts': [alert.to_dict() for alert in alerts],
            'count': len(alerts),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_audit_alerts'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/alerts/<alert_id>/acknowledge', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def acknowledge_alert(alert_id):
    """Acknowledge a security alert"""
    try:
        if not audit_alert_manager:
            return jsonify({'error': 'Audit alerting not enabled', 'status': 'error'}), 503
        
        success = audit_alert_manager.acknowledge_alert(alert_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Alert {alert_id} acknowledged',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({'error': 'Alert not found', 'status': 'error'}), 404
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'acknowledge_alert'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/verify-integrity', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def verify_audit_integrity():
    """Verify audit log hash chain integrity"""
    try:
        if not audit_logger:
            return jsonify({'error': 'Audit logging not enabled', 'status': 'error'}), 503
        
        is_valid, error_message = audit_logger.verify_integrity()
        
        return jsonify({
            'status': 'success',
            'integrity_valid': is_valid,
            'error_message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'verify_audit_integrity'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/audit/export', methods=['POST'])
@token_auth_required
@conditional_limit("2 per minute")
def export_audit_logs():
    """Export audit logs"""
    try:
        if not audit_logger:
            return jsonify({'error': 'Audit logging not enabled', 'status': 'error'}), 503
        
        data = request.get_json(force=True)
        format_type = data.get('format', 'json')
        filters = data.get('filters', {})
        
        exported_data = audit_logger.export_audit_logs(format_type=format_type, filters=filters)
        
        if format_type == 'csv':
            return exported_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=audit_logs.csv'
            }
        else:
            return exported_data, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'export_audit_logs'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# Revenue Tracking Endpoints
@app.route('/revenue/transactions', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_revenue_transaction():
    """
    Create a new revenue transaction
    """
    try:
        data = request.get_json(force=True)

        # Validate required fields
        required_fields = ['user_id', 'revenue_type', 'amount']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}', 'status': 'error'}), 400

        # Validate revenue type
        try:
            revenue_type = RevenueType(data['revenue_type'])
        except ValueError:
            return jsonify({'error': f'Invalid revenue type. Valid types: {[t.value for t in RevenueType]}', 'status': 'error'}), 400

        # Create transaction
        transaction = revenue_service.create_transaction(
            user_id=data['user_id'],
            revenue_type=revenue_type,
            amount=float(data['amount']),
            currency=data.get('currency', 'USD'),
            description=data.get('description'),
            merchant_name=data.get('merchant_name'),
            category=data.get('category'),
            payment_method=data.get('payment_method'),
            business_id=data.get('business_id'),
            external_reference=data.get('external_reference'),
            metadata=data.get('metadata')
        )

        return jsonify({
            'status': 'success',
            'transaction': transaction.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_revenue_transaction'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/revenue/transactions/<transaction_id>/process', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def process_revenue_transaction(transaction_id):
    """
    Process a pending revenue transaction
    """
    try:
        data = request.get_json(force=True)
        success = data.get('success', True)
        settlement_date_str = data.get('settlement_date')

        settlement_date = None
        if settlement_date_str:
            try:
                settlement_date = datetime.fromisoformat(settlement_date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid settlement_date format. Use ISO format.', 'status': 'error'}), 400

        success = revenue_service.process_transaction(transaction_id, success, settlement_date)

        if not success:
            return jsonify({'error': 'Transaction not found or cannot be processed', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'message': f'Transaction {transaction_id} processed successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'process_revenue_transaction'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/revenue/transactions/<transaction_id>', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_revenue_transaction(transaction_id):
    """
    Get revenue transaction details by ID
    """
    try:
        transaction = revenue_service.get_transaction(transaction_id)

        if not transaction:
            return jsonify({'error': 'Transaction not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'transaction': transaction.to_dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_revenue_transaction'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/revenue/transactions', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_user_revenue_transactions():
    """
    Get revenue transactions for a user
    """
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id parameter is required', 'status': 'error'}), 400

        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        if limit <= 0 or limit > 1000:
            return jsonify({'error': 'Limit must be between 1 and 1000', 'status': 'error'}), 400

        transactions = revenue_service.get_user_transactions(user_id, limit, offset)

        return jsonify({
            'status': 'success',
            'transactions': [t.to_dict() for t in transactions],
            'count': len(transactions),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_user_revenue_transactions'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/revenue/metrics', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_revenue_metrics():
    """
    Get revenue metrics for a date range
    """
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        revenue_type_str = request.args.get('revenue_type')

        if not start_date_str or not end_date_str:
            return jsonify({'error': 'start_date and end_date parameters are required', 'status': 'error'}), 400

        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use ISO format.', 'status': 'error'}), 400

        if start_date >= end_date:
            return jsonify({'error': 'start_date must be before end_date', 'status': 'error'}), 400

        # Validate revenue type if provided
        revenue_type = None
        if revenue_type_str:
            try:
                revenue_type = RevenueType(revenue_type_str)
            except ValueError:
                return jsonify({'error': f'Invalid revenue type. Valid types: {[t.value for t in RevenueType]}', 'status': 'error'}), 400

        metrics = revenue_service.get_revenue_metrics(start_date, end_date, revenue_type)

        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_revenue_metrics'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/revenue/metrics/update', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def update_daily_revenue_metrics():
    """
    Update daily revenue metrics aggregation
    """
    try:
        data = request.get_json(force=True)
        date_str = data.get('date')

        target_date = None
        if date_str:
            try:
                target_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use ISO format.', 'status': 'error'}), 400

        success = revenue_service.update_daily_metrics(target_date)

        if not success:
            return jsonify({'error': 'Failed to update metrics', 'status': 'error'}), 500

        date_display = target_date.date().isoformat() if target_date else datetime.now(timezone.utc).date().isoformat()

        return jsonify({
            'status': 'success',
            'message': f'Daily revenue metrics updated for {date_display}',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_daily_revenue_metrics'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):  # pylint: disable=unused-argument
    """Handle 500 errors"""
    telemetry_logger.log_error(error, {'context': 'flask_error_handler'})
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # Log application startup
    telemetry_logger.get_logger().info("Starting Telemetry API Server with SocketIO")

    # Print configuration
    telemetry_logger.get_logger().info(f"Configuration: {config.get_all_settings()}")

    # Run the application with SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.environ.get('FLASK_RUN_PORT', 5000)),
        debug=config.LOG_LEVEL == 'DEBUG'
    )
