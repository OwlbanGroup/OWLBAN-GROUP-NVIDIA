"""
User Manager for Database Operations
"""
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from src.models.user import User
from src.database_fixed import db_manager

class UserManager:
    """Manages user database operations"""

    @staticmethod
    def create_user(username, password):
        """Create a new user"""
        session = db_manager.get_session()
        try:
            # Check if user exists
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                return None, "User already exists"

            # Create new user
            user = User(
                username=username,
                password_hash=generate_password_hash(password),
                created_at=datetime.now(timezone.utc)
            )
            session.add(user)
            session.commit()
            return user, "User created successfully"
        except Exception as e:
            session.rollback()
            return None, f"Error creating user: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def verify_user(username, password):
        """Verify user credentials"""
        session = db_manager.get_session()
        try:
            user = session.query(User).filter_by(username=username).first()
            if not user:
                return False, None

            if check_password_hash(user.password_hash, password):
                return True, user
            return False, None
        finally:
            session.close()

    @staticmethod
    def update_token(username, token):
        """Update user token"""
        session = db_manager.get_session()
        try:
            user = session.query(User).filter_by(username=username).first()
            if user:
                user.token = token
                user.token_created_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            return False
        finally:
            session.close()

    @staticmethod
    def get_user_by_token(token):
        """Get user by token"""
        session = db_manager.get_session()
        try:
            user = session.query(User).filter_by(token=token).first()
            return user
        finally:
            session.close()

    @staticmethod
    def get_user_by_username(username):
        """Get user by username"""
        session = db_manager.get_session()
        try:
            user = session.query(User).filter_by(username=username).first()
            return user
        finally:
            session.close()

user_manager = UserManager()
