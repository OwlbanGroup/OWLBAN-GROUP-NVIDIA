"""
User management and authentication system for JPMorgan Financial APIs
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from src.logger import telemetry_logger


class UserManager:
    """Manages user authentication, sessions, and user data"""

    def __init__(self, users_file: str = 'users.json'):
        self.users_file = users_file
        self.users = self._load_users()
        self.sessions: Dict[str, Dict] = {}  # session_token -> user_data
        self.session_timeout = timedelta(hours=24)  # 24 hour sessions

    def _load_users(self) -> Dict[str, Dict]:
        """Load users from JSON file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            else:
                # Create default admin user
                default_users = {
                    'admin': {
                        'password_hash': generate_password_hash('admin123'),
                        'role': 'admin',
                        'email': 'admin@jpmorgan.com',
                        'created_at': datetime.now().isoformat(),
                        'active': True
                    }
                }
                self._save_users(default_users)
                telemetry_logger.get_logger().info("Created default admin user")
                return default_users
        except Exception as e:
            telemetry_logger.get_logger().error(f"Failed to load users: {e}")
            return {}

    def _save_users(self, users: Dict[str, Dict]):
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
        except Exception as e:
            telemetry_logger.get_logger().error(f"Failed to save users: {e}")

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate a user with username and password

        Returns:
            Tuple of (success, error_message)
        """
        if username not in self.users:
            return False, "User not found"

        user = self.users[username]
        if not user.get('active', True):
            return False, "Account is disabled"

        if not check_password_hash(user['password_hash'], password):
            return False, "Invalid password"

        return True, None

    def create_session_token(self, username: str) -> str:
        """Create a new session token for the user"""
        session_token = secrets.token_urlsafe(32)

        self.sessions[session_token] = {
            'username': username,
            'role': self.users[username]['role'],
            'created_at': datetime.now(),
            'expires_at': datetime.now() + self.session_timeout
        }

        telemetry_logger.get_logger().info(f"Created session for user: {username}")
        return session_token

    def validate_session_token(self, session_token: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate a session token

        Returns:
            Tuple of (valid, user_data)
        """
        if session_token not in self.sessions:
            return False, None

        session_data = self.sessions[session_token]

        # Check if session has expired
        if datetime.now() > session_data['expires_at']:
            del self.sessions[session_token]
            return False, None

        # Update expiration time (sliding window)
        session_data['expires_at'] = datetime.now() + self.session_timeout

        return True, session_data

    def logout_user(self, session_token: str) -> bool:
        """Log out a user by removing their session"""
        if session_token in self.sessions:
            username = self.sessions[session_token]['username']
            del self.sessions[session_token]
            telemetry_logger.get_logger().info(f"Logged out user: {username}")
            return True
        return False

    def create_user(self, username: str, password: str, email: str, role: str = 'user') -> Tuple[bool, str]:
        """
        Create a new user account

        Returns:
            Tuple of (success, message)
        """
        if username in self.users:
            return False, "User already exists"

        if len(password) < 8:
            return False, "Password must be at least 8 characters long"

        if role not in ['user', 'admin', 'analyst']:
            return False, "Invalid role. Must be 'user', 'admin', or 'analyst'"

        self.users[username] = {
            'password_hash': generate_password_hash(password),
            'role': role,
            'email': email,
            'created_at': datetime.now().isoformat(),
            'active': True
        }

        self._save_users(self.users)
        telemetry_logger.get_logger().info(f"Created user: {username} with role: {role}")
        return True, "User created successfully"

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information (without password hash)"""
        if username not in self.users:
            return None

        user = self.users[username].copy()
        user['username'] = username
        del user['password_hash']
        return user

    def list_users(self) -> Dict[str, Dict]:
        """List all users (without password hashes)"""
        users_list = {}
        for username, user_data in self.users.items():
            users_list[username] = {
                'role': user_data['role'],
                'email': user_data['email'],
                'created_at': user_data['created_at'],
                'active': user_data.get('active', True)
            }
        return users_list

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_tokens = []
        for token, session_data in self.sessions.items():
            if datetime.now() > session_data['expires_at']:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.sessions[token]

        if expired_tokens:
            telemetry_logger.get_logger().info(f"Cleaned up {len(expired_tokens)} expired sessions")


# Global user manager instance
user_manager = UserManager()
