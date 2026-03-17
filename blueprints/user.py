"""
User Blueprint for JPMorgan Financial APIs
Provides user management functionality including profiles, authentication, and user data.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, List

# Import services and utilities
from src.user_manager import user_manager
from src.logger import telemetry_logger

# Import authentication and rate limiting decorators
try:
    from src.auth import token_auth_required
    from src.rate_limiting import conditional_limit
except ImportError:
    # Fallback if not found - these would need to be implemented
    def token_auth_required(f):
        return f
    def conditional_limit(rate):
        def decorator(f):
            return f
        return decorator

bp = Blueprint('user', __name__)
user_bp = bp


# =============================================================================
# USER PROFILE ENDPOINTS
# =============================================================================


@bp.route('/profile', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_profile():
    """
    Get current user profile information
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        user_data = user_manager.get_user_info(user_id)

        if not user_data:
            return jsonify({'error': 'User not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'user': user_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_profile'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/profile', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_profile():
    """
    Update user profile information
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Update user profile (in real implementation, this would update database)
        updated_data = {
            'username': data.get('username', user_id),
            'email': data.get('email', ''),
            'first_name': data.get('first_name', ''),
            'last_name': data.get('last_name', ''),
            'phone': data.get('phone', ''),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        telemetry_logger.log_info(f"Profile updated for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': 'Profile updated successfully',
            'user': updated_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_profile'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/profile/password', methods=['PUT'])
@token_auth_required
@conditional_limit("5 per minute")
def change_password():
    """
    Change user password
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        current_password = data.get('current_password')
        new_password = data.get('new_password')

        if not current_password or not new_password:
            return jsonify({'error': 'Current password and new password are required', 'status': 'error'}), 400

        # Verify current password and update (in real implementation)
        success = user_manager.change_password(user_id, current_password, new_password)

        if success:
            telemetry_logger.log_info(f"Password changed for user {user_id}")
            return jsonify({
                'status': 'success',
                'message': 'Password changed successfully'
            }), 200
        else:
            return jsonify({
                'error': 'Current password is incorrect',
                'status': 'error'
            }), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'change_password'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# USER MANAGEMENT ENDPOINTS (Admin Only)
# =============================================================================

@user_bp.route('/users', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def list_users():
    """
    List all users (admin only)
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Check if user is admin (in real implementation)
        if user_id != 'admin':
            return jsonify({'error': 'Admin privileges required', 'status': 'error'}), 403

        users = user_manager.list_users()

        return jsonify({
            'status': 'success',
            'users': users,
            'count': len(users)
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_users'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/users/<user_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_user(user_id):
    """
    Get specific user information (admin only)
    """
    try:
        current_user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Check if user is admin or requesting their own profile
        if current_user_id != 'admin' and current_user_id != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        user_data = user_manager.get_user_info(user_id)

        if not user_data:
            return jsonify({'error': 'User not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'user': user_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/users/<user_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_user(user_id):
    """
    Update user information (admin only)
    """
    try:
        current_user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Check if user is admin
        if current_user_id != 'admin':
            return jsonify({'error': 'Admin privileges required', 'status': 'error'}), 403

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Update user (in real implementation)
        updated_data = {
            'username': user_id,
            'email': data.get('email', ''),
            'role': data.get('role', 'user'),
            'is_active': data.get('is_active', True),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        telemetry_logger.log_info(f"User {user_id} updated by admin {current_user_id}")

        return jsonify({
            'status': 'success',
            'message': 'User updated successfully',
            'user': updated_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/users/<user_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_user(user_id):
    """
    Delete user (admin only)
    """
    try:
        current_user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Check if user is admin
        if current_user_id != 'admin':
            return jsonify({'error': 'Admin privileges required', 'status': 'error'}), 403

        # Delete user (in real implementation)
        success = user_manager.delete_user(user_id)

        if success:
            telemetry_logger.log_info(f"User {user_id} deleted by admin {current_user_id}")
            return jsonify({
                'status': 'success',
                'message': 'User deleted successfully'
            }), 200
        else:
            return jsonify({
                'error': 'User not found',
                'status': 'error'
            }), 404

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_user'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# USER PREFERENCES ENDPOINTS
# =============================================================================

@user_bp.route('/preferences', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_preferences():
    """
    Get user preferences
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get user preferences (mock data for demo)
        preferences = {
            'theme': 'light',
            'language': 'en',
            'timezone': 'UTC',
            'notifications': {
                'email': True,
                'sms': False,
                'push': True
            },
            'privacy': {
                'profile_visible': True,
                'activity_visible': False
            }
        }

        return jsonify({
            'status': 'success',
            'preferences': preferences
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_preferences'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/preferences', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_preferences():
    """
    Update user preferences
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Update preferences (in real implementation, this would be stored)
        updated_preferences = {
            'theme': data.get('theme', 'light'),
            'language': data.get('language', 'en'),
            'timezone': data.get('timezone', 'UTC'),
            'notifications': data.get('notifications', {}),
            'privacy': data.get('privacy', {}),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        telemetry_logger.log_info(f"Preferences updated for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': 'Preferences updated successfully',
            'preferences': updated_preferences
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_preferences'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# USER ACTIVITY ENDPOINTS
# =============================================================================

@user_bp.route('/activity', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_activity():
    """
    Get user activity history
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = int(request.args.get('offset', 0))

        # Get user activity (mock data for demo)
        activities = [
            {
                'id': str(uuid.uuid4()),
                'type': 'login',
                'description': 'User logged in',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'ip_address': '192.168.1.1',
                'user_agent': 'Mozilla/5.0...'
            }
        ] * min(limit, 10)  # Mock 10 activities

        return jsonify({
            'status': 'success',
            'activities': activities[:limit],
            'count': len(activities),
            'total_count': len(activities)
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_activity'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@user_bp.route('/dashboard', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_user_dashboard():
    """
    Get user dashboard data
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get dashboard data (mock for demo)
        dashboard_data = {
            'user_info': user_manager.get_user_info(user_id),
            'recent_activity': [
                {
                    'type': 'login',
                    'description': 'Recent login',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            ],
            'stats': {
                'total_logins': 42,
                'last_login': datetime.now(timezone.utc).isoformat(),
                'account_age_days': 365
            }
        }

        return jsonify({
            'status': 'success',
            'dashboard': dashboard_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_user_dashboard'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
