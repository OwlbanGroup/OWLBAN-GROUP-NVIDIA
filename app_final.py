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
from typing import Optional

from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS  # type: ignore
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api  # type: ignore
from flask import Blueprint
from flask_talisman import Talisman  # type: ignore
from prometheus_client import Counter, Histogram, Gauge
from werkzeug.exceptions import BadRequest
from werkzeug.security import generate_password_hash, check_password_hash

# JP Morgan Financial Dashboard Extensions
import psycopg2
from psycopg2.extras import RealDictCursor
import schedule
import time
from threading import Thread
import requests
from decimal import Decimal

# Import sync scheduler for integrated data synchronization
from sync_scheduler import JPMorganSyncScheduler, create_scheduler

# Ensure project root and 'src' directory are in sys.path before importing blueprints
# This allows importing from blueprints.* modules and their src.* dependencies
# both when run directly and via app_final.py
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

src_path = os.path.join(_project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import PFM blueprint (will be registered after app creation)
pfm_bp: Optional[Blueprint] = None
PFM_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.pfm import pfm_bp
    PFM_BLUEPRINT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PFM blueprint: {e}")

# Import Payments Blueprint
payments_bp: Optional[Blueprint] = None
PAYMENTS_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.payments import payments_bp
    PAYMENTS_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import Payroll Blueprint
payroll_bp: Optional[Blueprint] = None
PAYROLL_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.payroll import payroll_bp
    PAYROLL_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import User Blueprint
user_bp: Optional[Blueprint] = None
USER_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.user import user_bp
    USER_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import Asset Blueprint
asset_bp: Optional[Blueprint] = None
ASSET_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.asset import asset_bp
    ASSET_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import Business Blueprint
business_bp: Optional[Blueprint] = None
BUSINESS_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.business import business_bp
    BUSINESS_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import ML Blueprint
ml_bp: Optional[Blueprint] = None
ML_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.ml import ml_bp
    ML_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import Data Blueprint
data_bp: Optional[Blueprint] = None
DATA_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.data import data_bp
    DATA_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import AI Blueprint
ai_bp: Optional[Blueprint] = None
AI_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.ai import ai_bp
    AI_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

# Import Internal Operations Blueprint
internal_ops_bp: Optional[Blueprint] = None
INTERNAL_OPS_BLUEPRINT_AVAILABLE = False
try:
    from blueprints.internal_ops import internal_ops_bp
    INTERNAL_OPS_BLUEPRINT_AVAILABLE = True
except ImportError:
    pass

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
    # Fallback: define a minimal config object for local development
    class Config:
        SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret')
        TOKEN_CLIENT_ID = os.environ.get('TOKEN_CLIENT_ID', 'dummy_client_id')
        TOKEN_CLIENT_SECRET = os.environ.get('TOKEN_CLIENT_SECRET', 'dummy_client_secret')
        TOKEN_URL = os.environ.get('TOKEN_URL', 'https://dummy.token.url')
        TOKEN_SCOPE = os.environ.get('TOKEN_SCOPE', 'dummy_scope')
        REDIS_URL = os.environ.get('REDIS_URL', None)
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        @staticmethod
        def get_all_settings():
            return {
                'SECRET_KEY': Config.SECRET_KEY,
                'TOKEN_CLIENT_ID': Config.TOKEN_CLIENT_ID,
                'TOKEN_CLIENT_SECRET': Config.TOKEN_CLIENT_SECRET,
                'TOKEN_URL': Config.TOKEN_URL,
                'TOKEN_SCOPE': Config.TOKEN_SCOPE,
                'REDIS_URL': Config.REDIS_URL,
                'LOG_LEVEL': Config.LOG_LEVEL
            }
    config = Config()  # type: ignore

try:
    from src.telemetry_handler_new import telemetry_handler  # type: ignore
except ImportError as e:
    raise ImportError("Could not import 'src.telemetry_handler_new'. Make sure 'src/telemetry_handler_new.py' exists and is not empty.") from e

from src.logger import telemetry_logger  # type: ignore
from src.token_manager import TokenManager  # type: ignore
from src.validation import InputValidator, ValidationError  # type: ignore
from src.cloud_storage import setup_cloud_storage  # type: ignore
from src.data_format_converter import DataFormatConverter  # type: ignore
from src.data_conversion_handler import convert_data_format_logic  # type: ignore
from src.ml_model import AnomalyDetector  # type: ignore
from src.database_fixed import db_manager, DBBusinessModel, DBAssetModel  # type: ignore
from src.schemas import BusinessCreate, BusinessUpdate, BusinessResponse, AssetCreate, AssetUpdate, AssetResponse  # type: ignore
from src.ai_service import ai_service  # type: ignore
from src.auth0_auth import setup_auth0_routes, auth0_required  # type: ignore
try:
    from src.payments_service import payments_service  # type: ignore
except ImportError:
    payments_service = None
    print("Warning: payments_service not available (STRIPE config missing)")

from src.sync_service import sync_service  # type: ignore

# Initialize cloud storage
setup_cloud_storage(config.get_all_settings())

# Initialize ML model
anomaly_detector = AnomalyDetector()

# Initialize sync scheduler
sync_scheduler = None
try:
    sync_scheduler = create_scheduler()
    telemetry_logger.get_logger().info("Sync scheduler initialized successfully")
except Exception as e:
    telemetry_logger.get_logger().debug(f"Non-fatal startup - sync scheduler unavailable (creds missing?): {e}")
    sync_scheduler = None


# Prometheus metrics (app_final version to avoid conflicts)
REQUEST_COUNT_FINAL = Counter('http_requests_total_final', 'Total HTTP requests (final)', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY_FINAL = Histogram('http_request_duration_seconds_final', 'HTTP request duration (final)', ['method', 'endpoint'])
ACTIVE_CONNECTIONS_FINAL = Gauge('active_connections_final', 'Number of active connections (final)')
ERROR_COUNT_FINAL = Counter('errors_total_final', 'Total errors (final)', ['type', 'endpoint'])
TELEMETRY_EVENTS_PROCESSED_FINAL = Counter('telemetry_events_processed_total_final', 'Total telemetry events processed (final)', ['status'])
BATCH_SIZE_FINAL = Histogram('telemetry_batch_size_final', 'Size of telemetry batches processed (final)')
ANOMALY_DETECTIONS_FINAL = Counter('anomaly_detections_total_final', 'Total anomaly detections performed (final)', ['result'])

# Initialize Flask app
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
CORS(app)

# Set testing mode from environment
if os.environ.get('TESTING') == '1':
    app.config['TESTING'] = True

# Initialize Flask-RESTX API for documentation
api = Api(app,
          title='JPMorgan Telemetry API',
          version=get_version(),
          description='Enterprise-grade API for processing Microsoft Windows Store '
                      'telemetry data with ML anomaly detection, cloud storage integration, '
                      'and GitHub MCP connectivity.',
          doc='/swagger/')

# Initialize security headers
Talisman(
    app,
    content_security_policy={
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline'",
        'style-src': "'self' 'unsafe-inline'",
        'img-src': "'self' data: https:",
        'font-src': "'self'",
        'connect-src': "'self' http://localhost:9090 http://localhost:3000",
        'frame-ancestors': "'none'",
    },
    force_https=False,  # Local dev
    strict_transport_security=True,
    strict_transport_security_max_age=31536000,  # 1 year
    strict_transport_security_include_subdomains=True,
    frame_options='DENY',
    content_security_policy_nonce_in=['script-src', 'style-src'],
    referrer_policy='strict-origin-when-cross-origin'
)  # Production security headers

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[] if app.config.get('TESTING') else ["200 per day", "50 per hour"]
)

# Conditional limiter for testing
def conditional_limit(limit_str):
    def decorator(f):
        if app.config.get('TESTING'):
            return f
        return limiter.limit(limit_str)(f)
    return decorator

# Initialize token manager
token_manager = TokenManager(
    client_id=config.TOKEN_CLIENT_ID,
    client_secret=config.TOKEN_CLIENT_SECRET,
    token_url=config.TOKEN_URL,
    scope=config.TOKEN_SCOPE
)

# Setup Auth0 routes
# setup_auth0_routes(app)  # Commented out to fix import issue

# Initialize Redis cache
if config.REDIS_URL:
    try:
        REDIS_CLIENT = redis.from_url(config.REDIS_URL, decode_responses=True)
    except Exception as e:
        telemetry_logger.get_logger().debug(f"Non-fatal startup - Redis unavailable, using in-memory: {config.REDIS_URL}: {str(e)}")
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
        # Skip authentication in testing mode
        if app.config.get('TESTING', False):
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
    try:
        data = request.get_json(force=True)
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'error': 'Username and password are required', 'status': 'error'}), 400

        success, message = create_user(username, password)
        if success:
            return jsonify({'status': 'success', 'message': message}), 201
        else:
            return jsonify({'error': message, 'status': 'error'}), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'register_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/user/login', methods=['POST'])
@conditional_limit("10 per minute")
def login_user():
    """
    Login user and return a token
    """
    try:
        data = request.get_json(force=True)
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'error': 'Username and password are required', 'status': 'error'}), 400

        if verify_user(username, password):
            # Generate a simple token (in production use JWT or OAuth)
            token = secrets.token_hex(16)
            # Store token in Redis or in-memory for validation (here in-memory for demo)
            users[username]['token'] = token
            users[username]['token_created_at'] = datetime.now(timezone.utc).isoformat()
            return jsonify({'status': 'success', 'token': token}), 200
        else:
            return jsonify({'error': 'Invalid username or password', 'status': 'error'}), 401
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'login_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


def token_auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if app.config.get('TESTING', False):
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
@token_auth_required
@conditional_limit("10 per minute")
def user_profile():
    """
    Get user profile information (requires user token)
    """
    try:
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1]
        # Find user by token
        for username, user_data in users.items():
            if user_data.get('token') == token:
                return jsonify({
                    'status': 'success',
                    'username': username,
                    'created_at': user_data['created_at'],
                    'token_created_at': user_data.get('token_created_at'),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }), 200
        return jsonify({'error': 'User not found', 'status': 'error'}), 404
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

@app.route('/data/convert', methods=['POST'])
@limiter.limit("5 per minute")
def convert_data_format():
    """
    Convert data between different formats
    """
    try:
        request_data = request.get_json()
        return convert_data_format_logic(request_data)
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
    # Parse pagination params
    page = int(request.args.get('page', 1))
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = (page - 1) * limit
    
    try:
        businesses = db_manager.get_all_businesses()
        total = len(businesses)
        paginated = businesses[offset:offset + limit]
        return jsonify({
                'status': 'success',
                'businesses': [BusinessResponse.from_orm(business).dict() for business in paginated],
                'count': len(paginated),
                'total': total,
                'page': page,
                'limit': limit,
                'pages': (total + limit - 1) // limit,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_businesses'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def create_business():
    """
    Create a new business
    """
    try:
        data = request.get_json(force=True)
        business_data = BusinessCreate(**data)
        business = db_manager.create_business(business_data.dict())
        return jsonify({
            'status': 'success',
            'business': BusinessResponse.from_orm(business).dict(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
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
    # Parse pagination params
    page = int(request.args.get('page', 1))
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = (page - 1) * limit
    
    try:
        assets = db_manager.get_all_assets()
        total = len(assets)
        paginated = assets[offset:offset + limit]
        return jsonify({
                'status': 'success',
                'assets': [AssetResponse.from_orm(asset).dict() for asset in paginated],
                'count': len(paginated),
                'total': total,
                'page': page,
                'limit': limit,
                'pages': (total + limit - 1) // limit,
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
    # Parse pagination params
    page = int(request.args.get('page', 1))
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = (page - 1) * limit
    
    try:
        business = db_manager.get_business_by_id(business_id)
        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404
        assets = db_manager.get_assets_by_business_id(business_id)
        total = len(assets)
        paginated = assets[offset:offset + limit]
        return jsonify({
            'status': 'success',
            'business_id': business_id,
            'assets': [AssetResponse.from_orm(asset).dict() for asset in paginated],
            'count': len(paginated),
            'total': total,
            'page': page,
            'limit': limit,
            'pages': (total + limit - 1) // limit,
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


@app.route('/', methods=['GET'])
def index():
    """Root endpoint for API information"""
    return jsonify({
        'message': 'Welcome to JPMorgan Financial APIs',
        'version': get_version(),
        'description': 'Enterprise-grade API for telemetry processing, ML anomaly detection, cloud integration, business asset management, and AI-powered financial insights',
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
    }), 200

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the web dashboard"""
    return render_template('index.html')

@app.route('/api/dashboard/summary', methods=['GET'])
@conditional_limit("30 per minute")
def dashboard_summary():
    """Return live summary metrics for dashboard widgets"""
    try:
        # Telemetry metrics
        telemetry_metrics = telemetry_handler.get_metrics(24)
        events_processed = telemetry_metrics.get('events_processed', 0) if isinstance(telemetry_metrics, dict) else 0
        anomalies_detected = telemetry_metrics.get('anomalies_detected', 0) if isinstance(telemetry_metrics, dict) else 0

        # Business / assets metrics
        businesses = db_manager.get_all_businesses()
        assets = db_manager.get_all_assets()

        active_users = 0
        for user in users.values():
            if user.get('token'):
                active_users += 1

        return jsonify({
            'status': 'success',
            'data': {
                'telemetry_events': events_processed,
                'anomalies_detected': anomalies_detected,
                'total_businesses': len(businesses),
                'total_assets': len(assets),
                'active_users': active_users
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'dashboard_summary'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/dashboard/trends', methods=['GET'])
@conditional_limit("30 per minute")
def dashboard_trends():
    """Return lightweight trend series for dashboard chart"""
    try:
        points = request.args.get('points', 12, type=int)
        if points <= 0 or points > 60:
            points = 12

        now = datetime.now(timezone.utc)
        labels = []
        values = []
        base_value = 5

        # Use telemetry metrics if available to influence trend level
        telemetry_metrics = telemetry_handler.get_metrics(1)
        if isinstance(telemetry_metrics, dict):
            base_value = max(1, int(telemetry_metrics.get('events_processed', 0) or 1))

        for i in range(points):
            labels.append((now.replace(microsecond=0)).isoformat())
            values.append(base_value + i)

        return jsonify({
            'status': 'success',
            'data': {
                'labels': labels,
                'requests': values
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'dashboard_trends'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/welcome/create-workspace', methods=['GET'])
@auth0_required
def create_workspace_page():
    """Serve the create workspace page"""
    return render_template('create_workspace.html')

@app.route('/api/workspaces', methods=['POST'])
@auth0_required
@conditional_limit("5 per minute")
def create_workspace():
    """
    Create a new workspace
    """
    try:
        data = request.get_json(force=True)
        workspace_data = {
            'name': data.get('name'),
            'url': data.get('url'),
            'description': data.get('description', ''),
            'region': data.get('region'),
            'created_by': g.user_id,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        # Validate required fields
        if not workspace_data['name'] or not workspace_data['url'] or not workspace_data['region']:
            return jsonify({'error': 'Name, URL, and region are required', 'status': 'error'}), 400

        # Here you would typically save to database
        # For now, we'll just return success
        workspace_data['id'] = secrets.token_hex(8)  # Generate workspace ID

        return jsonify({
            'status': 'success',
            'workspace': workspace_data,
            'message': f'Workspace "{workspace_data["name"]}" created successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_workspace'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# AI-Powered Endpoints
@app.route('/ai/analyze', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def ai_analyze_data():
    """
    AI-powered financial data analysis
    """
    try:
        data = request.get_json(force=True)
        financial_data = data.get('data', {})
        question = data.get('question', 'Analyze this financial data')
        context = data.get('context', 'General financial analysis')

        if not financial_data:
            return jsonify({'error': 'Financial data is required', 'status': 'error'}), 400

        result = ai_service.analyze_financial_data(financial_data, question, context)
        return jsonify(result), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ai_analyze_data'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/ai/risk-assess', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def ai_risk_assessment():
    """
    AI-powered transaction risk assessment
    """
    try:
        data = request.get_json(force=True)
        transaction_data = data.get('transaction_data', {})
        historical_patterns = data.get('historical_patterns', [])
        market_conditions = data.get('market_conditions', {})

        if not transaction_data:
            return jsonify({'error': 'Transaction data is required', 'status': 'error'}), 400

        result = ai_service.assess_transaction_risk(transaction_data, historical_patterns, market_conditions)
        return jsonify(result), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ai_risk_assessment'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/ai/query', methods=['POST'])
@token_auth_required
@conditional_limit("20 per minute")
def ai_natural_language_query():
    """
    Natural language financial queries using AI
    """
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        data_schema = data.get('data_schema', {})
        available_data = data.get('available_data', {})

        if not query:
            return jsonify({'error': 'Query is required', 'status': 'error'}), 400

        result = ai_service.process_natural_language_query(query, data_schema, available_data)
        return jsonify(result), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ai_natural_language_query'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/ai/status', methods=['GET'])
@conditional_limit("30 per minute")
def ai_service_status():
    """
    Get AI service status and configuration
    """
    try:
        status = ai_service.get_service_status()
        return jsonify({
            'status': 'success',
            'ai_service': status,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ai_service_status'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# Data Synchronization Endpoints
@app.route('/sync/start', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def start_sync_scheduler():
    """
    Start the data synchronization scheduler
    """
    try:
        global sync_scheduler
        if sync_scheduler is None:
            sync_scheduler = create_scheduler()
            telemetry_logger.get_logger().info("Sync scheduler created for manual start")

        if sync_scheduler.running:
            return jsonify({
                'status': 'error',
                'message': 'Sync scheduler is already running',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 400

        sync_scheduler.start_scheduler()
        return jsonify({
            'status': 'success',
            'message': 'Sync scheduler started successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'start_sync_scheduler'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/sync/stop', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def stop_sync_scheduler():
    """
    Stop the data synchronization scheduler
    """
    try:
        global sync_scheduler
        if sync_scheduler is None or not sync_scheduler.running:
            return jsonify({
                'status': 'error',
                'message': 'Sync scheduler is not running',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 400

        sync_scheduler.stop_scheduler()
        return jsonify({
            'status': 'success',
            'message': 'Sync scheduler stopped successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'stop_sync_scheduler'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/sync/status', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_sync_status():
    """
    Get the current status of the data synchronization scheduler
    """
    try:
        global sync_scheduler
        if sync_scheduler is None:
            return jsonify({
                'status': 'success',
                'scheduler_status': 'not_initialized',
                'jobs': {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200

        job_status = sync_scheduler.get_job_status()
        return jsonify({
            'status': 'success',
            'scheduler_status': 'running' if sync_scheduler.running else 'stopped',
            'jobs': job_status,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_sync_status'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/sync/run/<job_id>', methods=['POST'])
@token_auth_required
@conditional_limit("2 per minute")
def run_sync_job(job_id):
    """
    Run a specific synchronization job immediately
    """
    try:
        global sync_scheduler
        if sync_scheduler is None:
            return jsonify({
                'status': 'error',
                'message': 'Sync scheduler is not initialized',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 400

        result = sync_scheduler.run_job_now(job_id)
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'result': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'run_sync_job', 'job_id': job_id})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/sync/logs', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_sync_logs():
    """
    Get synchronization logs and history
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        if limit <= 0 or limit > 500:
            return jsonify({'error': 'Limit must be between 1 and 500', 'status': 'error'}), 400

        # Query sync logs from database
        with psycopg2.connect(sync_scheduler.db_manager.connection_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, sync_type, status, records_processed, records_failed,
                           error_message, started_at, completed_at, duration_seconds
                    FROM sync_logs
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (limit,))
                logs = cursor.fetchall()

        return jsonify({
            'status': 'success',
            'logs': [dict(log) for log in logs],
            'count': len(logs),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_sync_logs'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# Stripe Payment Endpoints
@app.route('/stripe/payment-intent', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_stripe_payment_intent():
    """
    Create a Stripe payment intent
    """
    try:
        data = request.get_json(force=True)
        amount = data.get('amount')
        currency = data.get('currency', config.STRIPE_CURRENCY)
        description = data.get('description', '')
        metadata = data.get('metadata', {})

        if not amount or amount <= 0:
            return jsonify({'error': 'Valid amount is required', 'status': 'error'}), 400

        result = payments_service.create_stripe_payment_intent(
            amount=amount,
            currency=currency,
            description=description,
            metadata=metadata
        )

        if result['status'] == 'success':
            return jsonify({
                'status': 'success',
                'payment_intent_id': result['payment_intent_id'],
                'client_secret': result['client_secret'],
                'amount': result['amount'],
                'currency': result['currency'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 201
        else:
            return jsonify({'error': result['error'], 'status': 'error'}), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_stripe_payment_intent'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/stripe/payment-intent/<payment_intent_id>/confirm', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def confirm_stripe_payment(payment_intent_id):
    """
    Confirm a Stripe payment intent
    """
    try:
        result = payments_service.confirm_stripe_payment(payment_intent_id)

        if result['status'] == 'success':
            return jsonify({
                'status': 'success',
                'payment_intent_id': result['payment_intent_id'],
                'amount': result['amount'],
                'currency': result['currency'],
                'payment_status': result['status'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        elif result['status'] == 'pending':
            return jsonify({
                'status': 'pending',
                'payment_intent_id': result['payment_intent_id'],
                'payment_status': result['status'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 202
        else:
            return jsonify({'error': result['error'], 'status': 'error'}), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'confirm_stripe_payment', 'payment_intent_id': payment_intent_id})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/stripe/refund', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def create_stripe_refund():
    """
    Create a Stripe refund
    """
    try:
        data = request.get_json(force=True)
        payment_intent_id = data.get('payment_intent_id')
        amount = data.get('amount')  # Optional, full refund if not specified
        reason = data.get('reason', 'requested_by_customer')

        if not payment_intent_id:
            return jsonify({'error': 'Payment intent ID is required', 'status': 'error'}), 400

        result = payments_service.create_stripe_refund(
            payment_intent_id=payment_intent_id,
            amount=amount,
            reason=reason
        )

        if result['status'] == 'success':
            return jsonify({
                'status': 'success',
                'refund_id': result['refund_id'],
                'amount': result['amount'],
                'currency': result['currency'],
                'refund_status': result['status'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 201
        else:
            return jsonify({'error': result['error'], 'status': 'error'}), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_stripe_refund'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/stripe/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handle Stripe webhook events
    """
    try:
        payload = request.get_data(as_text=True)
        sig_header = request.headers.get('stripe-signature')

        if not sig_header:
            return jsonify({'error': 'Missing Stripe signature', 'status': 'error'}), 400

        result = payments_service.process_stripe_webhook(payload, sig_header)

        if result['status'] == 'success':
            return jsonify({
                'status': 'success',
                'event_type': result['event_type'],
                'event_id': result['event_id'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({'error': result['error'], 'status': 'error'}), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'stripe_webhook'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/github/orgs', methods=['GET'])
def get_github_orgs():
    """
    Retrieve all GitHub organizations associated with the authenticated user's account
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header', 'status': 'error'}), 401

        api_key = auth_header.split(' ')[1]

        # Make request to Blackbox API
        response = requests.get('https://cloud.blackbox.ai/api/github/orgs', headers={'Authorization': f'Bearer {api_key}'})

        if response.status_code == 200:
            organizations = response.json()
            return jsonify(organizations), 200
        elif response.status_code == 401:
            return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing API key', 'status': 401}), 401
        elif response.status_code == 404:
            return jsonify({'error': 'Not Found', 'message': 'GitHub token not found or expired', 'status': 404}), 404
        elif response.status_code == 502:
            return jsonify({'error': 'Bad Gateway', 'message': 'GitHub API error occurred', 'status': 502}), 502
        else:
            return jsonify({'error': 'Internal server error', 'status': response.status_code}), 500

    except requests.RequestException as e:
        telemetry_logger.log_error(e, {'context': 'get_github_orgs'})
        return jsonify({'error': 'Failed to connect to Blackbox API', 'status': 'error'}), 500
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_github_orgs'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/github/repos', methods=['GET'])
def get_github_repos():
    """
    Retrieve repositories for a specific GitHub user or organization
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header', 'status': 'error'}), 401

        owner = request.args.get('owner')
        if not owner:
            return jsonify({'error': 'Missing required parameter: owner', 'status': 'error'}), 400

        api_key = auth_header.split(' ')[1]

        # Make request to Blackbox API
        response = requests.get(f'https://cloud.blackbox.ai/api/github/repos?owner={owner}', headers={'Authorization': f'Bearer {api_key}'})

        if response.status_code == 200:
            repositories = response.json()
            return jsonify(repositories), 200
        elif response.status_code == 400:
            return jsonify({'error': 'Bad Request', 'message': 'Missing required parameter: owner', 'status': 400}), 400
        elif response.status_code == 401:
            return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing API key', 'status': 401}), 401
        elif response.status_code == 404:
            return jsonify({'error': 'Not Found', 'message': 'GitHub token not found or expired', 'status': 404}), 404
        elif response.status_code == 502:
            return jsonify({'error': 'Bad Gateway', 'message': 'GitHub API error occurred', 'status': 502}), 502
        else:
            return jsonify({'error': 'Internal server error', 'status': response.status_code}), 500

    except requests.RequestException as e:
        telemetry_logger.log_error(e, {'context': 'get_github_repos'})
        return jsonify({'error': 'Failed to connect to Blackbox API', 'status': 'error'}), 500
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_github_repos'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# Webhook Endpoints for Real-time Data Updates
@app.route('/webhooks/jpmorgan/transactions', methods=['POST'])
@require_auth
@conditional_limit("100 per minute")
def jpmorgan_transaction_webhook():
    """
    Webhook endpoint for real-time JPMorgan transaction updates
    """
    try:
        webhook_data = request.get_json(force=True)
        if not webhook_data:
            return jsonify({'error': 'No webhook data provided', 'status': 'error'}), 400

        telemetry_logger.get_logger().info(f"Received JPMorgan transaction webhook: {webhook_data}")

        # Process webhook data immediately
        # This would trigger enrichment and AI analysis for real-time data
        transaction_data = webhook_data.get('transaction', {})

        # Store webhook event
        with psycopg2.connect(sync_scheduler.db_manager.connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO webhook_events (source, event_type, event_data, received_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                """, ('jpmorgan', 'transaction', json.dumps(webhook_data)))
                conn.commit()

        # Trigger real-time processing (enrichment + AI analysis)
        # This would be implemented based on the webhook payload

        return jsonify({
            'status': 'success',
            'message': 'Transaction webhook processed successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'jpmorgan_transaction_webhook'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/webhooks/jpmorgan/accounts', methods=['POST'])
@require_auth
@conditional_limit("50 per minute")
def jpmorgan_account_webhook():
    """
    Webhook endpoint for real-time JPMorgan account updates
    """
    try:
        webhook_data = request.get_json(force=True)
        if not webhook_data:
            return jsonify({'error': 'No webhook data provided', 'status': 'error'}), 400

        telemetry_logger.get_logger().info(f"Received JPMorgan account webhook: {webhook_data}")

        # Store webhook event
        with psycopg2.connect(sync_scheduler.db_manager.connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO webhook_events (source, event_type, event_data, received_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                """, ('jpmorgan', 'account', json.dumps(webhook_data)))
                conn.commit()

        return jsonify({
            'status': 'success',
            'message': 'Account webhook processed successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'jpmorgan_account_webhook'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/business/intelligence/<user_id>', methods=['GET'])
@require_auth
@conditional_limit("10 per minute")
def get_business_intelligence(user_id):
    """
    Get comprehensive business intelligence for a user
    """
    try:
        days = int(request.args.get('days', 30))

        result = sync_service.get_business_intelligence(user_id, days)

        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'business_intelligence', 'user_id': user_id})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/business/forecast/<user_id>', methods=['GET'])
@require_auth
@conditional_limit("5 per minute")
def forecast_revenue(user_id):
    """
    Get revenue forecast for a user
    """
    try:
        forecast_days = int(request.args.get('days', 30))

        result = sync_service.forecast_revenue(user_id, forecast_days)

        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'revenue_forecast', 'user_id': user_id})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/sync/payment/<payment_id>', methods=['POST'])
@require_auth
@conditional_limit("20 per minute")
def sync_payment_to_revenue(payment_id):
    """
    Sync a payment to create corresponding revenue transaction
    """
    try:
        from src.models.revenue import RevenueType

        revenue_type_str = request.json.get('revenue_type', 'purchase')
        revenue_type = RevenueType(revenue_type_str.upper())

        result = sync_service.sync_payment_to_revenue(payment_id, revenue_type)

        if result['status'] == 'success':
            return jsonify(result), 201
        else:
            return jsonify(result), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'sync_payment', 'payment_id': payment_id})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/ai/analyze', methods=['POST'])
@require_auth
@conditional_limit("10 per minute")
def api_ai_analyze_data():
    """
    AI-powered financial data analysis (API version)
    """
    try:
        data = request.json.get('data', {})
        question = request.json.get('question', 'Analyze this financial data')

        result = ai_service.analyze_financial_data(data, question)

        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'api_ai_analyze'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/ai/risk-assess', methods=['POST'])
@require_auth
@conditional_limit("10 per minute")
def ai_risk_assess():
    """
    AI-powered transaction risk assessment
    """
    try:
        transaction_data = request.json.get('transaction_data', {})
        historical_patterns = request.json.get('historical_patterns', [])
        market_conditions = request.json.get('market_conditions', {})

        result = ai_service.assess_transaction_risk(
            transaction_data, historical_patterns, market_conditions
        )

        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ai_risk_assess'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/ai/query', methods=['POST'])
@require_auth
@conditional_limit("20 per minute")
def api_ai_natural_language_query():
    """
    AI-powered natural language query processing
    """
    try:
        query = request.json.get('query', '')
        data_schema = request.json.get('data_schema', {})
        available_data = request.json.get('available_data', {})

        result = ai_service.process_natural_language_query(query, data_schema, available_data)

        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'ai_query'})
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

# Register PFM blueprint after app creation
if PFM_BLUEPRINT_AVAILABLE and pfm_bp:
    app.register_blueprint(pfm_bp, url_prefix='/pfm')
    telemetry_logger.get_logger().info("PFM blueprint registered successfully")
else:
    telemetry_logger.get_logger().debug("PFM blueprint not available - skipped")

# Register Payments blueprint
if PAYMENTS_BLUEPRINT_AVAILABLE and payments_bp:
    app.register_blueprint(payments_bp, url_prefix='/payments')
    telemetry_logger.get_logger().info("Payments blueprint registered successfully")
else:
    telemetry_logger.get_logger().debug("Payments blueprint not available - skipped")

# Register Payroll blueprint
if PAYROLL_BLUEPRINT_AVAILABLE and payroll_bp:
    app.register_blueprint(payroll_bp, url_prefix='/payroll')
    telemetry_logger.get_logger().info("Payroll blueprint registered successfully")
else:
    telemetry_logger.get_logger().debug("Payroll blueprint not available - skipped")

# Register User blueprint
if USER_BLUEPRINT_AVAILABLE and user_bp:
    app.register_blueprint(user_bp, url_prefix='/user')
    telemetry_logger.get_logger().info("User blueprint registered successfully")
else:
    telemetry_logger.get_logger().debug("User blueprint not available - skipped")

# Register Asset blueprint
if ASSET_BLUEPRINT_AVAILABLE and asset_bp:
    app.register_blueprint(asset_bp, url_prefix='/asset')
    telemetry_logger.get_logger().info("Asset blueprint registered successfully")
else:
    telemetry_logger.get_logger().debug("Asset blueprint not available - skipped")

# Register Business blueprint
if BUSINESS_BLUEPRINT_AVAILABLE and business_bp:
    app.register_blueprint(business_bp, url_prefix='/business')
    telemetry_logger.get_logger().info("Business blueprint registered successfully")
else:
    telemetry_logger.get_logger().warning("Business blueprint not available")

# Register ML blueprint
if ML_BLUEPRINT_AVAILABLE and ml_bp:
    app.register_blueprint(ml_bp, url_prefix='/ml')
    telemetry_logger.get_logger().info("ML blueprint registered successfully")
else:
    telemetry_logger.get_logger().warning("ML blueprint not available")

# Register Data blueprint
if DATA_BLUEPRINT_AVAILABLE and data_bp:
    app.register_blueprint(data_bp, url_prefix='/data')
    telemetry_logger.get_logger().info("Data blueprint registered successfully")
else:
    telemetry_logger.get_logger().warning("Data blueprint not available")

# Register AI blueprint
if AI_BLUEPRINT_AVAILABLE and ai_bp:
    app.register_blueprint(ai_bp, url_prefix='/ai')
    telemetry_logger.get_logger().info("AI blueprint registered successfully")
else:
    telemetry_logger.get_logger().warning("AI blueprint not available")

# Register Internal Operations blueprint
if INTERNAL_OPS_BLUEPRINT_AVAILABLE and internal_ops_bp:
    app.register_blueprint(internal_ops_bp, url_prefix='/internal-ops')
    telemetry_logger.get_logger().info("Internal Operations blueprint registered successfully")
else:
    telemetry_logger.get_logger().warning("Internal Operations blueprint not available")

if __name__ == '__main__':
    # Log application startup
    telemetry_logger.get_logger().info("Starting Telemetry API Server")

    # Print configuration
    telemetry_logger.get_logger().info(f"Configuration: {config.get_all_settings()}")

    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('FLASK_RUN_PORT', 5000)),
        debug=config.LOG_LEVEL == 'DEBUG'
    )
