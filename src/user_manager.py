"""
User Manager for Database Operations
"""
from datetime import datetime, timezone
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


user_manager = UserManager()

