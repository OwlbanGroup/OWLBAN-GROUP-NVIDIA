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
        self.domain = config.AUTH0_DOMAIN
        self.client_id = config.AUTH0_CLIENT_ID
        self.client_secret = config.AUTH0_CLIENT_SECRET
        self.audience = config.AUTH0_AUDIENCE
        self.algorithms = config.AUTH0_ALGORITHMS
        self.issuer = config.AUTH0_ISSUER
        self.jwks_url = config.AUTH0_JWKS_URL

        # Initialize Auth0 SDK clients
        self.get_token_client = GetToken(self.domain, self.client_id, client_secret=self.client_secret)
        self.management_client = Auth0(self.domain, self.client_secret) if self.client_secret else None

        # JWKS client for token verification
        self.jwks_client = PyJWKClient(self.jwks_url) if self.jwks_url else None

    def get_signing_key(self, token):
        """Get the signing key from JWKS"""
        if not self.jwks_client:
            raise ValueError("JWKS client not initialized")
        return self.jwks_client.get_signing_key_from_jwt(token)

    def verify_token(self, token):
        """Verify JWT token"""
        try:
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


# Global Auth0 manager instance
auth0_manager = Auth0Manager()


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
        payload = auth0_manager.verify_token(token)
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
            # This would typically redirect to Auth0 Universal Login
            # For API-only, return login URL
            login_url = f"https://{config.AUTH0_DOMAIN}/authorize?response_type=code&client_id={config.AUTH0_CLIENT_ID}&redirect_uri={request.host_url}auth/callback&scope=openid profile email&audience={config.AUTH0_AUDIENCE}"
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
            token_response = auth0_manager.get_token_client.authorization_code(
                code,
                f"{request.host_url}auth/callback",
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
            logout_url = f"https://{config.AUTH0_DOMAIN}/v2/logout?client_id={config.AUTH0_CLIENT_ID}&returnTo={request.host_url}"
            return jsonify({
                'logout_url': logout_url,
                'status': 'success'
            }), 200
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'auth_logout'})
            return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
