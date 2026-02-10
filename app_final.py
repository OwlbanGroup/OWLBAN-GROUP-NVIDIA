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

# Initialize cloud storage
setup_cloud_storage(config.get_all_settings())

# Initialize ML model
anomaly_detector = AnomalyDetector()

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
Talisman(app, content_security_policy=None, force_https=False)  # Configure CSP and HTTPS for production

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
setup_auth0_routes(app)

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
@auth0_required
@conditional_limit("10 per minute")
def list_businesses():
    """
    List all businesses
    """
    try:
        businesses = db_manager.get_all_businesses()
        return jsonify({
            'status': 'success',
            'businesses': [BusinessResponse.from_orm(business).dict() for business in businesses],
            'count': len(businesses),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_businesses'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/businesses', methods=['POST'])
@auth0_required
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


@app.route('/', methods=['GET'])
def index():
    """Root endpoint for API information"""
    return jsonify({
        'message': 'Welcome to JPMorgan Financial APIs',
        'version': get_version(),
        'description': 'Enterprise-grade API for telemetry processing, ML anomaly detection, cloud integration, and business asset management',
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
            '/dashboard - Web dashboard'
        ],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the web dashboard"""
    return render_template('index.html')

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
    telemetry_logger.get_logger().info("Starting Telemetry API Server")

    # Print configuration
    telemetry_logger.get_logger().info(f"Configuration: {config.get_all_settings()}")

    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('FLASK_RUN_PORT', 5000)),
        debug=config.LOG_LEVEL == 'DEBUG'
    )
