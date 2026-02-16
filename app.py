"""Main Flask application for handling telemetry data

This module is intentionally large while features are consolidated here.
Some top-level `except Exception` handlers are used to return 500 responses
from Flask endpoints; they are intentional and documented below.
"""

# pylint: disable=broad-exception-caught,line-too-long,too-many-lines

# Standard library
import json
from datetime import datetime, timezone
import os
import asyncio
import random
import csv
import io
from functools import wraps
import sys

# Ensure project root is on sys.path so local `src` package resolves for linters and runtime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Third-party
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from flask_restx import Api
import redis
from prometheus_client import Counter, Histogram, generate_latest, Gauge
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Local application imports (after environment loaded)
from config import config  # pylint: disable=wrong-import-position
from src.telemetry_handler import telemetry_handler
from src.logger import telemetry_logger
from src.token_manager import TokenManager
from src.validation import InputValidator, ValidationError
from src.websocket_manager import websocket_manager
from src.cloud_storage import cloud_storage_manager, setup_cloud_storage
from src.data_format_converter import DataFormatConverter
from src.mcp_integration import mcp_client
from src.security_middleware import sanitize_request_data, audit_request
from src.user_manager import user_manager
from src.data_conversion_handler import convert_data_format_logic

# Module-level Redis client placeholder (set later if configured)
REDIS_CLIENT = None

# Load version information
def get_version():
    """Get the current version from VERSION file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return '1.0.0'


# Build a safe configuration dictionary from `config` for libraries that expect a mapping
def _config_to_dict(cfg):
    try:
        if hasattr(cfg, 'get_all_settings') and callable(cfg.get_all_settings):
            return cfg.get_all_settings()
    except Exception:
        pass
    keys = [
        'SECRET_KEY', 'TOKEN_CLIENT_ID', 'TOKEN_CLIENT_SECRET', 'TOKEN_URL',
        'TOKEN_SCOPE', 'REDIS_URL', 'LOG_LEVEL'
    ]
    return {k: getattr(cfg, k, None) for k in keys}

# Initialize cloud storage with a safe settings dict
_settings = _config_to_dict(config)
setup_cloud_storage(_settings)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
ERROR_COUNT = Counter('errors_total', 'Total errors', ['type', 'endpoint'])
TELEMETRY_EVENTS_PROCESSED = Counter('telemetry_events_processed_total', 'Total telemetry events processed', ['status'])
BATCH_SIZE = Histogram('telemetry_batch_size', 'Size of telemetry batches processed')
ANOMALY_DETECTIONS = Counter('anomaly_detections_total', 'Total anomaly detections performed', ['result'])

# Initialize Flask app
app = Flask(__name__)
app.secret_key = _settings.get('SECRET_KEY')
CORS(app)

# Register security middleware
app.before_request(sanitize_request_data)
app.before_request(audit_request)

# Initialize Flask-RESTX API for documentation
api = Api(app,
            title='JPMorgan Telemetry API',
            version=get_version(),
            description='Enterprise-grade API for processing Microsoft Windows Store telemetry data with ML anomaly detection, cloud storage integration, and GitHub MCP connectivity.',
            doc='/swagger/')

# Initialize security headers with enhanced configuration
Talisman(app,
        content_security_policy={
            'default-src': "'self'",
            'script-src': "'self' 'unsafe-inline'",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data:",
            'font-src': "'self'",
            'connect-src': "'self'",
            'media-src': "'self'",
            'object-src': "'none'",
            'frame-ancestors': "'none'",
            'base-uri': "'self'",
            'form-action': "'self'"
        },
        content_security_policy_nonce_in=['script-src', 'style-src'],
        force_https=False,  # Disable HTTPS enforcement for local testing
        strict_transport_security=True,
        strict_transport_security_max_age=31536000,  # 1 year
        strict_transport_security_include_subdomains=True,
        frame_options='DENY',
        referrer_policy='strict-origin-when-cross-origin'
)

# Initialize rate limiter with Redis-based distributed limiting for production
# Use Redis storage for distributed rate limiting across multiple instances
redis_url = _settings.get('REDIS_URL')
if redis_url:
    try:
        # Test Redis connection for rate limiting
        test_redis = redis.from_url(redis_url)
        test_redis.ping()
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri=redis_url,
            default_limits=["200 per day", "50 per hour"]
        )
        telemetry_logger.get_logger().info("Rate limiter initialized with Redis storage: %s", redis_url)
    except Exception as e:
        telemetry_logger.get_logger().warning("Failed to initialize Redis rate limiter: %s. Using in-memory storage.", str(e))
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
else:
    # Fallback to in-memory storage if Redis not available
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    telemetry_logger.get_logger().warning("No REDIS_URL configured. Using in-memory rate limiting (not recommended for production).")

# Rate limit tiers
TIER_LIMITS = {
    'free': ["100 per day", "20 per hour"],
    'premium': ["1000 per day", "200 per hour"],
    'enterprise': ["5000 per day", "1000 per hour"]
}

# Initialize token manager
token_manager = TokenManager(
    client_id=_settings.get('TOKEN_CLIENT_ID'),
    client_secret=_settings.get('TOKEN_CLIENT_SECRET'),
    token_url=_settings.get('TOKEN_URL'),
    scope=_settings.get('TOKEN_SCOPE')
)

# Initialize Redis cache
if _settings.get('REDIS_URL'):
    try:
        REDIS_CLIENT = redis.from_url(_settings.get('REDIS_URL'), decode_responses=True)
    except Exception as e:
        telemetry_logger.get_logger().warning("Failed to connect to Redis at %s: %s. Using in-memory cache.", _settings.get('REDIS_URL'), str(e))
        REDIS_CLIENT = None
else:
    REDIS_CLIENT = None

def cache_result(key_prefix, expiration=300):
    """Decorator to cache function results in Redis with enhanced features"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if REDIS_CLIENT is None:
                return f(*args, **kwargs)

            # Create a more robust cache key
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            cache_key = f"{key_prefix}:{hash(args_str + kwargs_str)}"

            try:
                cached_result = REDIS_CLIENT.get(cache_key)
                if cached_result:
                    # Update access time for LRU-style cache management
                    REDIS_CLIENT.expire(cache_key, expiration)
                    return json.loads(cached_result)
            except Exception as e:
                telemetry_logger.get_logger().warning("Redis cache read error: %s", str(e))

            result = f(*args, **kwargs)

            try:
                REDIS_CLIENT.setex(cache_key, expiration, json.dumps(result, default=str))
            except Exception as e:
                telemetry_logger.get_logger().warning("Redis cache write error: %s", str(e))

            return result
        return wrapper
    return decorator


def cache_database_query(expiration=600):
    """Decorator specifically for caching database query results"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if REDIS_CLIENT is None:
                return f(*args, **kwargs)

            # Create cache key based on function name and parameters
            func_name = f.__name__
            args_hash = hash(json.dumps(args, sort_keys=True, default=str) +
                            json.dumps(kwargs, sort_keys=True, default=str))
            cache_key = f"db_query:{func_name}:{args_hash}"

            try:
                cached_result = REDIS_CLIENT.get(cache_key)
                if cached_result:
                    REDIS_CLIENT.expire(cache_key, expiration)
                    return json.loads(cached_result)
            except Exception as e:
                telemetry_logger.get_logger().warning(f"Database cache read error: {e}")

            result = f(*args, **kwargs)

            try:
                REDIS_CLIENT.setex(cache_key, expiration, json.dumps(result, default=str))
            except Exception as e:
                telemetry_logger.get_logger().warning("Database cache write error: %s", str(e))

            return result
        return wrapper
    return decorator


def invalidate_cache_pattern(pattern):
    """Invalidate cache keys matching a pattern"""
    if REDIS_CLIENT is None:
        return

    try:
        keys = REDIS_CLIENT.keys(pattern)
        if keys:
            REDIS_CLIENT.delete(*keys)
            telemetry_logger.get_logger().info("Invalidated %d cache keys matching %s", len(keys), pattern)
    except Exception as e:
        telemetry_logger.get_logger().warning("Cache invalidation error: %s", str(e))

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
        if not token_manager.validate_token(token):
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated_function


@app.route('/health', methods=['GET'])
@limiter.limit("10 per minute")
def health_check():
    """Health check endpoint"""
    telemetry_logger.get_logger().info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': get_version()
    })

@app.route('/telemetry', methods=['POST'])
@limiter.limit("5 per minute")
@require_auth
def receive_telemetry():
    """
    Receive and process telemetry data

    Expected JSON payload:
    {
        "ver": "4.0",
        "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
        "time": "2025-09-22T19:42:10.2549325Z",
        "data": {
            "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
            "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
            ...
        },
        "ext": { ... }
    }
    """
    try:
        # Get telemetry data from request
        telemetry_data = request.get_json()

        if not telemetry_data:
            return jsonify({
                'error': 'No telemetry data provided',
                'status': 'error'
            }), 400

        # Validate telemetry data
        try:
            InputValidator.validate_telemetry_data(telemetry_data)
        except ValidationError as e:
            return jsonify({
                'error': f'Validation error: {str(e)}',
                'status': 'error'
            }), 400

        # Process the telemetry data
        success = telemetry_handler.process_single_event(telemetry_data)

        if success:
            return jsonify({
                'status': 'success',
                'message': 'Telemetry data processed successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to process telemetry data',
                'status': 'error'
            }), 500

    except BadRequest:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'telemetry_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/telemetry/batch', methods=['POST'])
@limiter.limit("3 per minute")
@require_auth
def receive_telemetry_batch():
    """
    Receive and process batch telemetry data

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
        # Get batch data from request
        request_data = request.get_json()

        if not request_data or 'telemetry_data' not in request_data:
            return jsonify({
                'error': 'No telemetry data batch provided',
                'status': 'error'
            }), 400

        telemetry_data_list = request_data['telemetry_data']

        if not isinstance(telemetry_data_list, list):
            return jsonify({
                'error': 'telemetry_data must be a list',
                'status': 'error'
            }), 400

        # Validate batch data
        try:
            InputValidator.validate_batch_data(request_data)
        except ValidationError as e:
            return jsonify({
                'error': f'Validation error: {str(e)}',
                'status': 'error'
            }), 400

        # Process the batch
        stats = telemetry_handler.process_batch(telemetry_data_list)

        return jsonify({
            'status': 'success',
            'message': f'Batch processed: {stats["successful"]}/{stats["total"]} events successful',
            'statistics': stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'telemetry_batch_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/telemetry/metrics', methods=['GET'])
@limiter.limit("5 per minute")
def get_telemetry_metrics():
    """
    Get telemetry metrics and statistics

    Query parameters:
    - hours: Number of hours to look back (default: 24)
    """
    try:
        hours = request.args.get('hours', 24, type=int)

        if hours <= 0 or hours > 720:  # Max 30 days
            return jsonify({
                'error': 'Hours must be between 1 and 720',
                'status': 'error'
            }), 400

        metrics = telemetry_handler.get_metrics(hours)

        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'metrics_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/ml/anomalies', methods=['POST'])
@limiter.limit("2 per minute")
@require_auth
def detect_anomalies():
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
        request_data = request.get_json()

        if not request_data or 'telemetry_data' not in request_data:
            return jsonify({
                'error': 'No telemetry data provided',
                'status': 'error'
            }), 400

        telemetry_data_list = request_data['telemetry_data']

        if not isinstance(telemetry_data_list, list):
            return jsonify({
                'error': 'telemetry_data must be a list',
                'status': 'error'
            }), 400

        # Detect anomalies
        anomaly_results = telemetry_handler.detect_anomalies_in_batch(telemetry_data_list)

        return jsonify({
            'status': 'success',
            'anomaly_results': anomaly_results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'anomalies_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/ml/train', methods=['POST'])
@limiter.limit("1 per minute")
@require_auth
def train_ml_model():
    """
    Train the ML model with telemetry data

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
        request_data = request.get_json()

        if not request_data or 'telemetry_data' not in request_data:
            return jsonify({
                'error': 'No telemetry data provided',
                'status': 'error'
            }), 400

        telemetry_data_list = request_data['telemetry_data']

        if not isinstance(telemetry_data_list, list):
            return jsonify({
                'error': 'telemetry_data must be a list',
                'status': 'error'
            }), 400

        # Train the model
        success = telemetry_handler.train_anomaly_model(telemetry_data_list)

        if success:
            return jsonify({
                'status': 'success',
                'message': 'ML model trained successfully',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to train ML model',
                'status': 'error'
            }), 500

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'train_ml_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/telemetry/export', methods=['GET'])
@limiter.limit("2 per minute")
@require_auth
def export_telemetry():
    """
    Export telemetry events

    Query parameters:
    - operation: Filter by operation (optional)
    - limit: Maximum number of events (default: 1000)
    - format: Export format (json, csv) (default: json)
    """
    try:
        operation = request.args.get('operation', None)
        limit = request.args.get('limit', 1000, type=int)
        export_format = request.args.get('format', 'json').lower()

        if limit <= 0 or limit > 10000:
            return jsonify({
                'error': 'Limit must be between 1 and 10000',
                'status': 'error'
            }), 400

        if export_format not in ['json', 'csv']:
            return jsonify({
                'error': 'Format must be json or csv',
                'status': 'error'
            }), 400

        events = telemetry_handler.export_events(operation, limit)

        if export_format == 'csv':
            # Convert to CSV format
            if not events:
                return jsonify({
                    'error': 'No events found',
                    'status': 'error'
                }), 404

            output = io.StringIO()
            fieldnames = events[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

            return output.getvalue(), 200, {'Content-Type': 'text/csv'}

        return jsonify({
            'status': 'success',
            'events': events,
            'count': len(events),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'export_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    telemetry_logger.log_error(error, {'context': 'flask_error_handler'})
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

@app.route('/ws/status', methods=['GET'])
def websocket_status():
    """Get WebSocket connection status"""
    try:
        connection_count = asyncio.run(websocket_manager.get_connection_count())
        client_count = asyncio.run(websocket_manager.get_client_count())

        return jsonify({
            'status': 'success',
            'active_connections': connection_count,
            'unique_clients': client_count,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'websocket_status_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/storage/export', methods=['POST'])
@limiter.limit("1 per minute")
@require_auth
def export_to_cloud_storage():
    """
    Export telemetry data to cloud storage

    Expected JSON payload:
    {
        "operation": "optional_operation_filter",
        "limit": 1000,
        "format": "json",
        "providers": ["aws", "gcs", "azure"],
        "filename_prefix": "telemetry_export"
    }
    """
    try:
        request_data = request.get_json()

        if not request_data:
            return jsonify({
                'error': 'No export configuration provided',
                'status': 'error'
            }), 400

        operation = request_data.get('operation')
        limit = request_data.get('limit', 1000)
        export_format = request_data.get('format', 'json').lower()
        providers = request_data.get('providers', list(cloud_storage_manager.providers.keys()))
        filename_prefix = request_data.get('filename_prefix', 'telemetry_export')

        if limit <= 0 or limit > 10000:
            return jsonify({
                'error': 'Limit must be between 1 and 10000',
                'status': 'error'
            }), 400

        if export_format not in DataFormatConverter.get_supported_formats():
            return jsonify({
                'error': f'Unsupported format. Supported formats: {DataFormatConverter.get_supported_formats()}',
                'status': 'error'
            }), 400

        # Get events from telemetry handler
        events = telemetry_handler.export_events(operation, limit)

        if not events:
            return jsonify({
                'error': 'No events found to export',
                'status': 'error'
            }), 404

        # Export to cloud storage
        results = cloud_storage_manager.export_telemetry_data(
            data=events,
            filename_prefix=filename_prefix,
            format_type=export_format,
            providers=providers
        )

        return jsonify({
            'status': 'success',
            'message': f'Data exported to {len([r for r in results.values() if not r.startswith("ERROR")])} providers',
            'export_results': results,
            'exported_records': len(events),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'cloud_export_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/data/convert', methods=['POST'])
@limiter.limit("5 per minute")
def convert_data_format():
    try:
        from src.data_conversion_handler import convert_data_format_logic

        response = convert_data_format_logic(request.get_json())
        # The helper returns either a Flask Response object or a tuple
        if isinstance(response, tuple):
            return response
        return response
    except Exception as exc:  # keep narrow refactorable in next pass
        telemetry_logger.log_error(exc, {'context': 'convert_data_format'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'data_conversion_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/data/formats', methods=['GET'])
def get_supported_formats():
    """Get list of supported data formats"""
    try:
        return jsonify({
            'status': 'success',
            'import_formats': DataFormatConverter.get_supported_import_formats(),
            'export_formats': DataFormatConverter.get_supported_formats(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'formats_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

# MCP Server Integration Endpoints

@app.route('/mcp/repos', methods=['GET'])
@limiter.limit("10 per minute")
@require_auth
def search_repositories():
    """
    Search GitHub repositories using MCP Server

    Query parameters:
    - query: Search query (default: "")
    - per_page: Number of results per page (default: 10)
    """
    try:
        query = request.args.get('query', '')
        per_page = request.args.get('per_page', 10, type=int)

        if per_page <= 0 or per_page > 100:
            return jsonify({
                'error': 'per_page must be between 1 and 100',
                'status': 'error'
            }), 400

        repositories = mcp_client.list_repositories(query, per_page)

        return jsonify({
            'status': 'success',
            'repositories': repositories,
            'count': len(repositories),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'mcp_repos_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/mcp/issues/<owner>/<repo>', methods=['GET'])
@limiter.limit("10 per minute")
@require_auth
def list_issues(owner, repo):
    """
    List issues for a GitHub repository using MCP Server

    Query parameters:
    - state: Issue state (open, closed, all) (default: open)
    - per_page: Number of results per page (default: 10)
    """
    try:
        state = request.args.get('state', 'open')
        per_page = request.args.get('per_page', 10, type=int)

        if state not in ['open', 'closed', 'all']:
            return jsonify({
                'error': 'state must be open, closed, or all',
                'status': 'error'
            }), 400

        if per_page <= 0 or per_page > 100:
            return jsonify({
                'error': 'per_page must be between 1 and 100',
                'status': 'error'
            }), 400

        issues = mcp_client.list_issues(owner, repo, state, per_page)

        return jsonify({
            'status': 'success',
            'issues': issues,
            'count': len(issues),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'mcp_issues_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/mcp/issues/<owner>/<repo>', methods=['POST'])
@limiter.limit("5 per minute")
@require_auth
def create_issue(owner, repo):
    """
    Create a new issue in a GitHub repository using MCP Server

    Expected JSON payload:
    {
        "title": "Issue title",
        "body": "Issue description",
        "assignees": ["username1", "username2"]
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'title' not in request_data:
            return jsonify({
                'error': 'title is required',
                'status': 'error'
            }), 400

        title = request_data['title']
        body = request_data.get('body', '')
        assignees = request_data.get('assignees', [])

        if not isinstance(assignees, list):
            return jsonify({
                'error': 'assignees must be a list',
                'status': 'error'
            }), 400

        result = mcp_client.create_issue(owner, repo, title, body, assignees)

        return jsonify({
            'status': 'success',
            'issue': result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'mcp_create_issue_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

# Authentication Endpoints

@app.route('/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """
    User login endpoint

    Expected JSON payload:
    {
        "username": "user",
        "password": "password"
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'username' not in request_data or 'password' not in request_data:
            return jsonify({
                'error': 'Username and password are required',
                'status': 'error'
            }), 400

        username = request_data['username']
        password = request_data['password']

        # Authenticate user
        success, error_message = user_manager.authenticate_user(username, password)

        if not success:
            return jsonify({
                'error': error_message,
                'status': 'error'
            }), 401

        # Create session token
        session_token = user_manager.create_session_token(username)

        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'token': session_token,
            'user': user_manager.get_user_info(username),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'login_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/auth/logout', methods=['POST'])
@limiter.limit("10 per minute")
@require_auth
def logout():
    """
    User logout endpoint

    Requires Authorization header with Bearer token
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'Missing or invalid authorization header',
                'status': 'error'
            }), 401

        token = auth_header.split(' ')[1]

        # Logout user
        success = user_manager.logout_user(token)

        if success:
            return jsonify({
                'status': 'success',
                'message': 'Logout successful',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                'error': 'Invalid session token',
                'status': 'error'
            }), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'logout_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/user/register', methods=['POST'])
@limiter.limit("2 per minute")
def public_register():
    """
    Public user registration endpoint

    Expected JSON payload:
    {
        "username": "newuser",
        "password": "password123",
        "email": "user@example.com"
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'username' not in request_data or 'password' not in request_data:
            return jsonify({
                'error': 'Username and password are required',
                'status': 'error'
            }), 400

        username = request_data['username']
        password = request_data['password']
        email = request_data.get('email', '')
        role = 'user'  # Default role for public registration

        # Create user
        success, message = user_manager.create_user(username=username, password=password, email=email, role=role)

        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'user': user_manager.get_user_info(username),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 201
        else:
            return jsonify({
                'error': message,
                'status': 'error'
            }), 400

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'public_register_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/auth/register', methods=['POST'])
@limiter.limit("2 per minute")
@require_auth
def register():
    """
    User registration endpoint (admin only)

    Expected JSON payload:
    {
        "username": "newuser",
        "password": "password123",
        "email": "user@example.com",
        "role": "user"
    }

    Requires Authorization header with Bearer token (admin role required)
    """
    try:
        # Check if user is admin
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'Missing or invalid authorization header',
                'status': 'error'
            }), 401

        token = auth_header.split(' ')[1]
        user_info = user_manager.validate_session_token(token)

        if not user_info or user_info[1]['role'] != 'admin':
            return jsonify({
                'error': 'Admin privileges required',
                'status': 'error'
            }), 403

        request_data = request.get_json()

        if not request_data or 'username' not in request_data or 'password' not in request_data:
            return jsonify({
                'error': 'Username and password are required',
                'status': 'error'
            }), 400

        username = request_data['username']
        password = request_data['password']
        email = request_data.get('email', '')
        role = request_data.get('role', 'user')

        # Create user
        success, message = user_manager.create_user(username=username, password=password, email=email, role=role)

        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'user': user_manager.get_user_info(username),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 201
        else:
            return jsonify({
                'error': message,
                'status': 'error'
            }), 400

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'register_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/auth/me', methods=['GET'])
@limiter.limit("10 per minute")
@require_auth
def get_current_user():
    """
    Get current user information

    Requires Authorization header with Bearer token
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'Missing or invalid authorization header',
                'status': 'error'
            }), 401

        token = auth_header.split(' ')[1]
        user_info = user_manager.validate_session_token(token)

        if not user_info:
            return jsonify({
                'error': 'Invalid session token',
                'status': 'error'
            }), 401

        username = user_info[1]['username']
        user_data = user_manager.get_user_info(username)

        return jsonify({
            'status': 'success',
            'user': user_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_current_user_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/auth/users', methods=['GET'])
@limiter.limit("5 per minute")
@require_auth
def list_users():
    """
    List all users (admin only)

    Requires Authorization header with Bearer token (admin role required)
    """
    try:
        # Check if user is admin
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'Missing or invalid authorization header',
                'status': 'error'
            }), 401

        token = auth_header.split(' ')[1]
        user_info = user_manager.validate_session_token(token)

        if not user_info or user_info[1]['role'] != 'admin':
            return jsonify({
                'error': 'Admin privileges required',
                'status': 'error'
            }), 403

        users = user_manager.list_users()

        return jsonify({
            'status': 'success',
            'users': users,
            'count': len(users),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_users_endpoint'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint for API information"""
    return jsonify({
        'message': 'Welcome to JPMorgan Financial APIs',
        'version': get_version(),
        'description': 'Enterprise-grade API for telemetry processing, ML anomaly detection, and cloud integration',
        'endpoints': [
            '/health - Health check',
            '/auth/login - User login',
            '/auth/logout - User logout',
            '/user/register - Public user registration',
            '/auth/register - User registration (admin only)',
            '/auth/me - Current user info',
            '/auth/users - List users (admin only)',
            '/telemetry - Process telemetry events',
            '/telemetry/batch - Batch telemetry processing',
            '/telemetry/metrics - Telemetry metrics',
            '/ml/anomalies - ML anomaly detection',
            '/ml/train - Train ML model',
            '/telemetry/export - Export telemetry data',
            '/metrics - Prometheus metrics',
            '/ws/status - WebSocket status',
            '/storage/export - Cloud storage export',
            '/data/convert - Data format conversion',
            '/data/formats - Supported formats',
            '/mcp/repos - GitHub repositories',
            '/mcp/issues/<owner>/<repo> - GitHub issues',
            '/dashboard - Web dashboard'
        ],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the web dashboard"""
    try:
        with open('dashboard.html', 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html'}
    except FileNotFoundError:
        return jsonify({
            'error': 'Dashboard file not found',
            'status': 'error'
        }), 404

@app.route('/api/jpmorgan-data', methods=['GET'])
@limiter.limit("10 per minute")
@require_auth
def get_jpmorgan_data():
    """
    Get live JPMorgan financial data for dashboard

    Requires Authorization header with Bearer token
    """
    try:
        # Generate mock live financial data
        # In production, this would fetch real data from JPMorgan APIs
        data = {
            'financial_metrics': {
                'revenue': 125000000000 + random.randint(-1000000000, 1000000000),
                'net_income': 48000000000 + random.randint(-500000000, 500000000),
                'total_assets': 3200000000000 + random.randint(-10000000000, 10000000000),
                'market_cap': 450000000000 + random.randint(-5000000000, 5000000000),
                'pe_ratio': round(12.5 + random.uniform(-0.5, 0.5), 2),
                'dividend_yield': round(0.0275 + random.uniform(-0.002, 0.002), 4)
            },
            'stock_ticker': {
                'symbol': 'JPM',
                'company_name': 'JPMorgan Chase & Co.',
                'exchange': 'NYSE',
                'current_price': round(145.50 + random.uniform(-2, 2), 2),
                'change': round(random.uniform(-3, 3), 2),
                'change_percent': round(random.uniform(-2, 2), 2),
                'volume': 12500000 + random.randint(-1000000, 1000000)
            },
            'assets': [
                {'name': 'Cash & Equivalents', 'value': 850000000000},
                {'name': 'Securities', 'value': 650000000000},
                {'name': 'Loans', 'value': 1100000000000},
                {'name': 'Trading Assets', 'value': 450000000000},
                {'name': 'Other Assets', 'value': 150000000000}
            ],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'live'
        }

        return jsonify(data), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_jpmorgan_data'})
        return jsonify({
            'error': 'Failed to fetch financial data',
            'status': 'error'
        }), 500

@app.route('/user/login', methods=['POST'])
@limiter.limit("5 per minute")
def user_login():
    """
    User login endpoint for dashboard

    Expected JSON payload:
    {
        "username": "user",
        "password": "password"
    }
    """
    try:
        request_data = request.get_json()

        if not request_data or 'username' not in request_data or 'password' not in request_data:
            return jsonify({
                'error': 'Username and password are required',
                'status': 'error'
            }), 400

        username = request_data['username']
        password = request_data['password']

        # Authenticate user
        success, error_message = user_manager.authenticate_user(username, password)

        if not success:
            return jsonify({
                'error': error_message,
                'status': 'error'
            }), 401

        # Create session token
        session_token = user_manager.create_session_token(username)

        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'token': session_token,
            'user': user_manager.get_user_info(username),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON format',
            'status': 'error'
        }), 400
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'user_login'})
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Log application startup
    telemetry_logger.get_logger().info("Starting Telemetry API Server")

    # Print configuration
    telemetry_logger.get_logger().info("Configuration: %s", _settings)

    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('FLASK_RUN_PORT', 5000)),
        debug=_settings.get('LOG_LEVEL') == 'DEBUG'
    )
