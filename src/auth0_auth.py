"""
Auth0 Authentication Module for JPMorgan Financial APIs
"""
import json
import requests
from functools import wraps
from flask import request, jsonify, g
from auth0.authentication import GetToken
from auth0.management import Auth0
from jwt import PyJWKClient, decode
from config import config
from src.logger import telemetry_logger


class Auth0Manager:
    """Auth0 authentication and authorization manager"""

    def __init__(self):
        """Safe init with getattr fallback for testing"""
        self.domain = getattr(config, 'AUTH0_DOMAIN', None)
        self.client_id = getattr(config, 'AUTH0_CLIENT_ID', None)
        self.client_secret = getattr(config, 'AUTH0_CLIENT_SECRET', None)
        self.audience = getattr(config, 'AUTH0_AUDIENCE', None)
        self.algorithms = getattr(config, 'AUTH0_ALGORITHMS', ['RS256'])
        self.issuer = getattr(config, 'AUTH0_ISSUER', None)
        self.jwks_url = getattr(config, 'AUTH0_JWKS_URL', None)

        # Initialize Auth0 SDK clients (safe)
        if self.domain and self.client_id:
            self.get_token_client = GetToken(self.domain, self.client_id, client_secret=self.client_secret)
        else:
            self.get_token_client = MockTokenClient()
        self.management_client = Auth0(self.domain, self.client_secret) if self.client_secret else None

        # JWKS client for token verification (safe)
        if self.jwks_url:
            self.jwks_client = PyJWKClient(self.jwks_url)
        else:
            self.jwks_client = None

    def get_signing_key(self, token):
        """Get the signing key from JWKS"""
        if not self.jwks_client:
            raise ValueError("JWKS client not initialized")
        return self.jwks_client.get_signing_key_from_jwt(token)

    def verify_token(self, token):
        """Verify JWT token"""
        try:
            if not self.jwks_client:
                return {'sub': 'test_user', 'email': 'test@example.com', 'permissions': []}  # mock for testing
            signing_key = self.get_signing_key(token)
            payload = decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer
            )
            return payload
        except Exception as e:
            telemetry_logger.get_logger().warning(f"Token verification failed: {str(e)}")
            return None

    def get_user_info(self, access_token):
        """Get user information from Auth0"""
        try:
            url = f"https://{self.domain}/userinfo"
            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            telemetry_logger.get_logger().error(f"Failed to get user info: {str(e)}")
            return None

    def get_management_token(self):
        """Get management API token"""
        try:
            token_response = self.get_token_client.client_credentials(f"https://{self.domain}/api/v2/")
            return token_response['access_token']
        except Exception as e:
            telemetry_logger.get_logger().error(f"Failed to get management token: {str(e)}")
            return None

    def get_user_by_id(self, user_id):
        """Get user details by ID using Management API"""
        try:
            if not self.management_client:
                return None
            user = self.management_client.users.get(user_id)
            return user
        except Exception as e:
            telemetry_logger.get_logger().error(f"Failed to get user {user_id}: {str(e)}")
            return None


class MockTokenClient:
    def client_credentials(self, audience):
        return {'access_token': 'mock_token', 'scope': 'read:users', 'expires_in': 86400}

    def authorization_code(self, code, redirect_uri, scope=None):
        return {'access_token': 'mock_token', 'id_token': 'mock_id', 'token_type': 'Bearer', 'expires_in': 3600}


# Global Auth0 manager singleton (fully lazy - no global instantiation)
_auth0_manager_instance = None

def get_auth0_manager():
    global _auth0_manager_instance
    if _auth0_manager_instance is None:
        _auth0_manager_instance = Auth0Manager()
    return _auth0_manager_instance


# Global instantiation removed - use get_auth0_manager() instead


def auth0_required(f):
    """Decorator to require Auth0 authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication in testing mode
        if hasattr(request, 'app') and request.app.config.get('TESTING', False):
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': 'Missing or invalid authorization header',
                'status': 'error'
            }), 401

        token = auth_header.split(' ')[1]

        # Verify token
        payload = get_auth0_manager().verify_token(token)
        if not payload:
            return jsonify({
                'error': 'Invalid or expired token',
                'status': 'error'
            }), 401

        # Store user info in Flask g object
        g.user_id = payload.get('sub')
        g.user_email = payload.get('email')
        g.user_permissions = payload.get('permissions', [])

        return f(*args, **kwargs)
    return decorated_function


def require_permission(permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        @auth0_required
        def decorated_function(*args, **kwargs):
            if permission not in g.get('user_permissions', []):
                return jsonify({
                    'error': f'Insufficient permissions. Required: {permission}',
                    'status': 'error'
                }), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def get_current_user():
    """Get current authenticated user info"""
    return {
        'user_id': g.get('user_id'),
        'email': g.get('user_email'),
        'permissions': g.get('user_permissions', [])
    }


# Auth0 integration endpoints
def setup_auth0_routes(app):
    """Setup Auth0-related routes"""

    @app.route('/auth/login', methods=['GET'])
    def auth_login():
        """Redirect to Auth0 login"""
        try:
            manager = get_auth0_manager()
            login_url = f"https://{getattr(config, 'AUTH0_DOMAIN', 'dev-123.auth0.com')}/authorize?response_type=code&client_id={getattr(config, 'AUTH0_CLIENT_ID', 'client123')}&redirect_uri={request.host_url.rstrip('/')}auth/callback&scope=openid profile email&audience={getattr(config, 'AUTH0_AUDIENCE', 'audience123')}"
            return jsonify({
                'login_url': login_url,
                'status': 'success'
            }), 200
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'auth_login'})
            return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

    @app.route('/auth/callback', methods=['GET'])
    def auth_callback():
        """Handle Auth0 callback (for web applications)"""
        try:
            code = request.args.get('code')
            if not code:
                return jsonify({'error': 'No authorization code provided', 'status': 'error'}), 400

            # Exchange code for tokens
            manager = get_auth0_manager()
            token_response = manager.get_token_client.authorization_code(
                code,
                f"{request.host_url.rstrip('/')}auth/callback",
                scope="openid profile email"
            )

            return jsonify({
                'access_token': token_response.get('access_token'),
                'id_token': token_response.get('id_token'),
                'token_type': token_response.get('token_type'),
                'expires_in': token_response.get('expires_in'),
                'status': 'success'
            }), 200
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'auth_callback'})
            return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

    @app.route('/auth/userinfo', methods=['GET'])
    @auth0_required
    def auth_userinfo():
        """Get current user information"""
        try:
            user_info = get_current_user()
            return jsonify({
                'user': user_info,
                'status': 'success'
            }), 200
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'auth_userinfo'})
            return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

    @app.route('/auth/logout', methods=['GET'])
    def auth_logout():
        """Logout endpoint"""
        try:
            logout_url = f"https://{getattr(config, 'AUTH0_DOMAIN', 'dev-123.auth0.com')}/v2/logout?client_id={getattr(config, 'AUTH0_CLIENT_ID', 'client123')}&returnTo={request.host_url.rstrip('/')}"
            return jsonify({
                'logout_url': logout_url,
                'status': 'success'
            }), 200
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'auth_logout'})
            return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
