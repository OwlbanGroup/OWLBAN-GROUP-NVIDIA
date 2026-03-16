"""
User Manager for Database Operations
"""
from datetime import datetime, timezone
import secrets
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import generate_password_hash, check_password_hash
from src.models.user import User
from src.database_fixed import db_manager


class UserManager:
    """Manages user database operations"""

    @staticmethod
    def create_user(username: str, password: str, email: str | None = None, role: str = 'user'):
        """Create a new user. Accepts optional email and role."""
        try:
            with db_manager.get_session() as session:
                # Check if user exists
                existing_user = session.query(User).filter_by(username=username).first()
                if existing_user:
                    return None, "User already exists"

                # Create new user
                user = User(
                    username=username,
                    password_hash=generate_password_hash(password),
                    email=email,
                    role=role,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(user)
                session.commit()
                return user, "User created successfully"
        except SQLAlchemyError as _err:
            return None, f"Error creating user: {_err}"

    @staticmethod
    def verify_user(username: str, password: str):
        """Verify user credentials"""
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username=username).first()
            if not user:
                return False, None

            if check_password_hash(user.password_hash, password):
                return True, user
            return False, None

    @staticmethod
    def update_token(username: str, token: str) -> bool:
        """Update user token"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(username=username).first()
                if user:
                    user.token = token
                    user.token_created_at = datetime.now(timezone.utc)
                    session.commit()
                    return True
                return False
        except SQLAlchemyError:
            return False

    @staticmethod
    def get_user_by_token(token: str):
        """Get user by token"""
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(token=token).first()
            return user

    @staticmethod
    def get_user_by_username(username: str):
        """Get user by username"""
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username=username).first()
            return user


    @staticmethod
    def authenticate_user(username: str, password: str):
        """Authenticate user with username and password"""
        success, user = UserManager.verify_user(username, password)
        if success:
            return True, None
        return False, "Invalid username or password"

    @staticmethod
    def create_session_token(username: str):
        """Create a new session token for user"""
        token = secrets.token_urlsafe(32)
        UserManager.update_token(username, token)
        return token

    @staticmethod
    def get_user_info(username: str):
        """Get user information"""
        user = UserManager.get_user_by_username(username)
        if user:
            return {
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'created_at': user.created_at.isoformat() if user.created_at else None
            }
        return None

    @staticmethod
    def logout_user(token: str):
        """Logout user by invalidating token"""
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(token=token).first()
                if user:
                    user.token = None
                    user.token_created_at = None
                    session.commit()
                    return True
                return False
        except SQLAlchemyError:
            return False

    @staticmethod
    def validate_session_token(token: str):
        """Validate session token and return user info"""
        user = UserManager.get_user_by_token(token)
        if user:
            return (True, {
                'username': user.username,
                'email': user.email,
                'role': user.role
            })
        return (False, None)

    @staticmethod
    def list_users():
        """List all users"""
        with db_manager.get_session() as session:
            users = session.query(User).all()
            return [
                {
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'created_at': user.created_at.isoformat() if user.created_at else None
                }
                for user in users
            ]

    @staticmethod
    def ensure_user(username: str, password: str, email: str | None = None, role: str = 'user'):
        """Ensure user exists; create if missing."""
        with db_manager.get_session() as session:
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                return existing_user, "User already exists"

            user = User(
                username=username,
                password_hash=generate_password_hash(password),
                email=email,
                role=role,
                created_at=datetime.now(timezone.utc)
            )
            session.add(user)
            session.commit()
            return user, "User created successfully"

    @staticmethod
    def ensure_owlban_team_users():
        """Ensure OWLBAN internal team users are provisioned."""
        provisioned = []
        defaults = [
            {
                "username": "oscar.broome",
                "password": "Owlban#TempPass2026!",
                "email": "oscar.broome@owlban.internal",
                "role": "manager"
            },
            {
                "username": "david.leeper",
                "password": "Owlban#TempPass2026!",
                "email": "david.leeper@owlban.internal",
                "role": "manager"
            }
        ]

        for user_data in defaults:
            user, message = UserManager.ensure_user(
                username=user_data["username"],
                password=user_data["password"],
                email=user_data["email"],
                role=user_data["role"]
            )
            provisioned.append({
                "username": user_data["username"],
                "email": user_data["email"],
                "role": user_data["role"],
                "status": "existing" if message == "User already exists" else "created"
            })

        return provisioned



