"""
Unified Authentication Library for OWLBAN GROUP
Provides JWT-based authentication, password management, and session handling
"""

import jwt
import bcrypt
import secrets
import hashlib
import hmac
import base64
import struct
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import os
from dataclasses import dataclass, asdict

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TOTP:
    """RFC 6238 TOTP (dependency-free, stdlib only).

    Time-based one-time password generation and verification compatible with
    standard authenticator apps. Uses SHA-1, a 30s period, and 6-digit codes
    per the common authenticator profile (RFC 4226/6238).
    """

    DIGITS = 6
    PERIOD = 30
    ALGORITHM = hashlib.sha1

    @staticmethod
    def generate_secret(byte_length: int = 20) -> str:
        """Return a base32-encoded random secret suitable for OTP."""
        return base64.b32encode(secrets.token_bytes(byte_length)).decode().rstrip('=')

    @staticmethod
    def _hotp(secret: bytes, counter: int) -> str:
        """RFC 4226 HOTP for the given 8-byte big-endian counter."""
        msg = struct.pack('>Q', counter)
        digest = hmac.new(secret, msg, TOTP.ALGORITHM).digest()
        offset = digest[-1] & 0x0F
        binary = struct.unpack('>I', digest[offset:offset + 4])[0] & 0x7FFFFFFF
        return str(binary % (10 ** TOTP.DIGITS)).zfill(TOTP.DIGITS)

    @staticmethod
    def _secret_to_bytes(secret: str) -> bytes:
        padded = secret.upper() + '=' * ((8 - len(secret) % 8) % 8)
        return base64.b32decode(padded)

    @classmethod
    def code_for_time(cls, secret: str, at_time: Optional[float] = None) -> str:
        """Generate the current (or given epoch) TOTP code for a secret."""
        import time as _time
        now = at_time if at_time is not None else _time.time()
        counter = int(now // cls.PERIOD)
        return cls._hotp(cls._secret_to_bytes(secret), counter)

    @classmethod
    def verify(cls, secret: str, code: str, window: int = 1) -> bool:
        """Verify a TOTP code, allowing `window` steps of clock drift."""
        import time as _time
        if not code or not code.isdigit():
            return False
        current_counter = int(_time.time() // cls.PERIOD)
        for counter in range(current_counter - window, current_counter + window + 1):
            expected = cls._hotp(cls._secret_to_bytes(secret), counter)
            if hmac.compare_digest(expected, code):
                return True
        return False

    @classmethod
    def provisioning_uri(cls, secret: str, email: str, issuer: str = "OWLBAN GROUP") -> str:
        """Return an otpauth:// provisioning URI for authenticator apps."""
        from urllib.parse import quote
        otpauth = f"otpauth://totp/{quote(issuer)}:{quote(email)}?secret={secret}"
        otpauth += f"&issuer={quote(issuer)}&period={cls.PERIOD}&digits={cls.DIGITS}"
        return otpauth


@dataclass
class User:
    """User data structure"""
    id: str
    email: str
    username: str
    password_hash: str
    role: str
    company: str
    permissions: List[str]
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'last_login', 'locked_until']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)

@dataclass
class Session:
    """Session data structure"""
    session_id: str
    user_id: str
    email: str
    role: str
    permissions: List[str]
    company: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    active: bool = True


@dataclass
class OAuthClient:
    """Registered OAuth2 client (confidential or public)."""
    client_id: str
    client_secret: Optional[str]
    name: str
    redirect_uris: List[str]
    scopes: List[str]
    created_at: datetime = None
    active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthClient":
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


def pkce_s256_challenge(verifier: str) -> str:
    """Return the PKCE S256 code_challenge for a code_verifier (RFC 7636)."""
    import hashlib as _hashlib
    digest = _hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_pkce_verifier() -> str:
    """Generate a high-entropy PKCE code_verifier (43-128 chars)."""
    return secrets.token_urlsafe(64)[:64]


class AuthConfig:
    """Authentication configuration"""
    JWT_SECRET = os.getenv('JWT_SECRET', secrets.token_hex(32))
    JWT_REFRESH_SECRET = os.getenv('JWT_REFRESH_SECRET', secrets.token_hex(32))
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '15'))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7'))

    PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', '8'))
    PASSWORD_REQUIRE_UPPERCASE = os.getenv('PASSWORD_REQUIRE_UPPERCASE', 'true').lower() == 'true'
    PASSWORD_REQUIRE_LOWERCASE = os.getenv('PASSWORD_REQUIRE_LOWERCASE', 'true').lower() == 'true'
    PASSWORD_REQUIRE_NUMBERS = os.getenv('PASSWORD_REQUIRE_NUMBERS', 'true').lower() == 'true'
    PASSWORD_REQUIRE_SPECIAL = os.getenv('PASSWORD_REQUIRE_SPECIAL', 'false').lower() == 'true'

    MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
    LOCKOUT_DURATION_MINUTES = int(os.getenv('LOCKOUT_DURATION_MINUTES', '15'))

    COMPANIES = ['OWLBAN_GROUP', 'OSCAR_BROOME', 'BLACKBOX_AI', 'NVIDIA_INTEGRATION']
    ROLES = ['admin', 'user', 'executive', 'developer', 'analyst']

class AuthManager:
    """Unified authentication manager for all OWLBAN GROUP systems"""

    def __init__(self, user_store_file: str = 'users.json', session_store_file: str = 'sessions.json'):
        self.config = AuthConfig()
        self.user_store_file = user_store_file
        self.session_store_file = session_store_file
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self._load_data()

        # Create default admin user if no users exist
        if not self.users:
            self._create_default_admin()

    def _load_data(self):
        """Load users and sessions from storage"""
        try:
            if os.path.exists(self.user_store_file):
                with open(self.user_store_file, 'r') as f:
                    user_data = json.load(f)
                    self.users = {email: User.from_dict(data) for email, data in user_data.items()}
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")

        try:
            if os.path.exists(self.session_store_file):
                with open(self.session_store_file, 'r') as f:
                    session_data = json.load(f)
                    self.sessions = {sid: Session(**data) for sid, data in session_data.items()}
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")

    def _save_data(self):
        """Save users and sessions to storage"""
        try:
            user_data = {email: user.to_dict() for email, user in self.users.items()}
            with open(self.user_store_file, 'w') as f:
                json.dump(user_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")

        try:
            session_data = {sid: asdict(session) for sid, session in self.sessions.items()}
            # Convert datetime objects to ISO strings
            for data in session_data.values():
                for key in ['created_at', 'expires_at']:
                    if isinstance(data[key], datetime):
                        data[key] = data[key].isoformat()
            with open(self.session_store_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")

    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = User(
            id='admin-001',
            email='admin@owlban.com',
            username='admin',
            password_hash=bcrypt.hashpw('Admin2024!'.encode(), bcrypt.gensalt()).decode(),
            role='admin',
            company='OWLBAN_GROUP',
            permissions=['read', 'write', 'delete', 'admin', 'manage_users']
        )
        self.users[admin_user.email] = admin_user
        self._save_data()
        logger.info("Default admin user created")

    def validate_password_policy(self, password: str) -> Tuple[bool, str]:
        """Validate password against policy"""
        if len(password) < self.config.PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {self.config.PASSWORD_MIN_LENGTH} characters long"

        if self.config.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        if self.config.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        if self.config.PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"

        if self.config.PASSWORD_REQUIRE_SPECIAL and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            return False, "Password must contain at least one special character"

        return True, "Password is valid"

    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode(), password_hash.encode())

    def create_user(self, email: str, username: str, password: str, role: str = 'user',
                   company: str = 'OWLBAN_GROUP', permissions: List[str] = None) -> Tuple[bool, str]:
        """Create a new user"""
        if email in self.users:
            return False, "User already exists"

        if not EMAIL_REGEX.match(email):
            return False, "Invalid email format"

        if role not in self.config.ROLES:
            return False, f"Invalid role. Must be one of: {', '.join(self.config.ROLES)}"

        if company not in self.config.COMPANIES:
            return False, f"Invalid company. Must be one of: {', '.join(self.config.COMPANIES)}"

        # Validate password
        valid, message = self.validate_password_policy(password)
        if not valid:
            return False, message

        if permissions is None:
            permissions = ['read']

        user = User(
            # 64-bit random ID namespace: supports up to ~18 quintillion unique IDs,
            # scaling comfortably beyond the 10B-user target.
            id=f"{company.lower()}-{secrets.token_hex(8)}",
            email=email,
            username=username,
            password_hash=self.hash_password(password),
            role=role,
            company=company,
            permissions=permissions
        )

        self.users[email] = user
        self._save_data()
        logger.info(f"User created: {email}")
        return True, "User created successfully"

    def authenticate_user(self, email: str, password: str, ip_address: str = None,
                         user_agent: str = None) -> Tuple[bool, str, Optional[User]]:
        """Authenticate a user"""
        user = self.users.get(email)
        if not user:
            logger.warning(f"Login attempt for non-existent user: {email}")
            return False, "Invalid credentials", None

        # Check if account is active
        if not user.active:
            logger.warning(f"Login attempt for inactive user: {email}")
            return False, "Account is deactivated", None

        # Check if account is locked
        if user.locked_until and datetime.now(timezone.utc) < user.locked_until:
            return False, "Account is temporarily locked", None

        # Verify password
        if not self.verify_password(password, user.password_hash):
            user.login_attempts += 1
            if user.login_attempts >= self.config.MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.LOCKOUT_DURATION_MINUTES)
                logger.warning(f"Account locked for user: {email}")
            self._save_data()
            logger.warning(f"Invalid password for user: {email}")
            return False, "Invalid credentials", None

        # Reset login attempts on successful login
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        self._save_data()

        logger.info(f"Successful login for user: {email}")
        return True, "Login successful", user

    def generate_tokens(self, user: User) -> Tuple[str, str]:
        """Generate access and refresh tokens"""
        now = datetime.now(timezone.utc)

        access_token_payload = {
            # jti (JWT ID) guarantees per-token uniqueness even within the same
            # second, and provides a revocation index for scale-out deployments.
            'jti': secrets.token_hex(8),
            'user_id': user.id,
            'email': user.email,
            'username': user.username,
            'role': user.role,
            'company': user.company,
            'permissions': user.permissions,
            'type': 'access',
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(minutes=self.config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp())
        }

        refresh_token_payload = {
            'jti': secrets.token_hex(8),
            'user_id': user.id,
            'email': user.email,
            'type': 'refresh',
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(days=self.config.JWT_REFRESH_TOKEN_EXPIRE_DAYS)).timestamp())
        }

        access_token = jwt.encode(access_token_payload, self.config.JWT_SECRET, algorithm='HS256')
        refresh_token = jwt.encode(refresh_token_payload, self.config.JWT_REFRESH_SECRET, algorithm='HS256')

        return access_token, refresh_token

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify an access token"""
        try:
            payload = jwt.decode(token, self.config.JWT_SECRET, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """Refresh an access token using a refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.config.JWT_REFRESH_SECRET, algorithms=['HS256'])
            if payload.get('type') != 'refresh':
                return None

            user = self.users.get(payload['email'])
            if not user:
                return None

            return self.generate_tokens(user)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def create_session(self, user: User, ip_address: str = None, user_agent: str = None) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)

        session = Session(
            session_id=session_id,
            user_id=user.id,
            email=user.email,
            role=user.role,
            permissions=user.permissions,
            company=user.company,
            created_at=now,
            expires_at=now + timedelta(hours=24),  # Sessions last 24 hours
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        self._save_data()
        logger.info(f"Session created for user: {user.email}")
        return session_id

    def verify_session(self, session_id: str) -> Optional[Session]:
        """Verify a session"""
        session = self.sessions.get(session_id)
        if not session or not session.active:
            return None

        if datetime.now(timezone.utc) > session.expires_at:
            session.active = False
            self._save_data()
            return None

        return session

    def destroy_session(self, session_id: str):
        """Destroy a session"""
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            self._save_data()
            logger.info(f"Session destroyed: {session_id}")

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.users.get(email)

    def update_user(self, email: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        user = self.users.get(email)
        if not user:
            return False

        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        self._save_data()
        logger.info(f"User updated: {email}")
        return True

    def delete_user(self, email: str) -> bool:
        """Delete a user"""
        if email in self.users:
            del self.users[email]
            self._save_data()
            logger.info(f"User deleted: {email}")
            return True
        return False

    def list_users(self, company: str = None) -> List[Dict[str, Any]]:
        """List all users, optionally filtered by company"""
        users = []
        for user in self.users.values():
            if company is None or user.company == company:
                user_dict = user.to_dict()
                # Remove sensitive information
                user_dict.pop('password_hash', None)
                users.append(user_dict)
        return users

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now(timezone.utc)
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if not session.active or now > session.expires_at:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            self._save_data()
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def create_password_reset_token(self, email: str) -> Optional[str]:
        """Create a password reset token for a user. Returns the token or None."""
        user = self.users.get(email)
        if not user:
            return None
        reset_token = secrets.token_urlsafe(48)
        if not hasattr(self, '_password_reset_tokens'):
            self._password_reset_tokens = {}
        self._password_reset_tokens[reset_token] = {
            'email': email,
            'expires_at': datetime.now(timezone.utc) + timedelta(hours=1),
            'used': False,
        }
        logger.info(f"Password reset token created for {email}")
        return reset_token

    def verify_password_reset_token(self, reset_token: str) -> Optional[str]:
        """Verify a reset token. Returns the associated email if valid, else None."""
        tokens = getattr(self, '_password_reset_tokens', {})
        record = tokens.get(reset_token)
        if not record or record['used']:
            return None
        if datetime.now(timezone.utc) > record['expires_at']:
            return None
        return record['email']

    def reset_password(self, reset_token: str, new_password: str) -> Tuple[bool, str]:
        """Reset a user's password using a valid reset token."""
        email = self.verify_password_reset_token(reset_token)
        if not email:
            return False, "Invalid or expired reset token"
        valid, message = self.validate_password_policy(new_password)
        if not valid:
            return False, message
        user = self.users.get(email)
        if not user:
            return False, "User not found"
        user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self._password_reset_tokens[reset_token]['used'] = True
        for session in self.sessions.values():
            if session.user_id == user.id:
                session.active = False
        self._save_data()
        logger.info(f"Password reset successful for {email}")
        return True, "Password reset successful"

    def log_audit_event(self, event_type: str, user_email: str, details: Dict[str, Any],
                        ip_address: str = None, severity: str = "info"):
        """Log an audit event for security tracking."""
        if not hasattr(self, '_audit_log'):
            self._audit_log = []
        event = {
            'event_id': secrets.token_hex(16),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'user_email': user_email,
            'ip_address': ip_address,
            'details': details,
            'severity': severity,
        }
        self._audit_log.append(event)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]
        try:
            audit_file = self.user_store_file.replace('.json', '_audit.json')
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(self._audit_log, f, indent=2)
        except Exception:
            logger.warning("Failed to persist audit log to file")
        logger.info(f"Audit event: {event_type} for {user_email}")

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        log = getattr(self, '_audit_log', [])
        return log[-limit:]

    def generate_api_key(self, email: str, name: str = "default") -> Optional[str]:
        """Generate a new API key for a user."""
        user = self.users.get(email)
        if not user:
            return None
        api_key = f"owlban_{secrets.token_urlsafe(48)}"
        if not hasattr(self, '_api_keys'):
            self._api_keys = {}
        self._api_keys[api_key] = {
            'email': email,
            'name': name,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'active': True,
            'last_used': None,
        }
        self._save_api_keys()
        logger.info(f"API key created for {email}")
        return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key. Returns user info if valid."""
        keys = getattr(self, '_api_keys', {})
        record = keys.get(api_key)
        if not record or not record['active']:
            return None
        user = self.users.get(record['email'])
        if not user or not user.active:
            return None
        record['last_used'] = datetime.now(timezone.utc).isoformat()
        return {'email': record['email'], 'name': record['name'], 'user': user}

    def revoke_api_key(self, email: str, api_key: str) -> bool:
        """Revoke a user's API key."""
        keys = getattr(self, '_api_keys', {})
        record = keys.get(api_key)
        if not record or record['email'] != email:
            return False
        record['active'] = False
        self._save_api_keys()
        logger.info(f"API key revoked for {email}")
        return True

    def list_api_keys(self, email: str) -> List[Dict[str, Any]]:
        """List all API keys for a user."""
        keys = getattr(self, '_api_keys', {})
        result = []
        for key, record in keys.items():
            if record['email'] == email:
                result.append({
                    'name': record['name'],
                    'created_at': record['created_at'],
                    'active': record['active'],
                    'last_used': record['last_used'],
                    'key_preview': key[:12] + '...',
                })
        return result

    def _save_api_keys(self):
        """Persist API keys to file."""
        try:
            keys_file = self.user_store_file.replace('.json', '_apikeys.json')
            with open(keys_file, 'w', encoding='utf-8') as f:
                json.dump(getattr(self, '_api_keys', {}), f, indent=2)
        except Exception:
            logger.warning("Failed to save API keys")

    # -------------------- Multi-Factor Authentication (TOTP) --------------------

    def setup_mfa(self, email: str) -> Optional[Dict[str, Any]]:
        """Generate and store a TOTP secret for a user.

        Returns dict with 'secret' and 'provisioning_uri', or None if the user
        does not exist or MFA is already enabled.
        """
        user = self.users.get(email)
        if not user:
            return None
        if user.mfa_enabled:
            return None
        secret = TOTP.generate_secret()
        user.mfa_secret = secret
        uri = TOTP.provisioning_uri(secret, email)
        self._save_data()
        logger.info(f"MFA setup initiated for {email}")
        return {"secret": secret, "provisioning_uri": uri}

    def enable_mfa(self, email: str, code: str) -> Tuple[bool, str]:
        """Verify a TOTP code and enable MFA for the user."""
        user = self.users.get(email)
        if not user:
            return False, "User not found"
        if not user.mfa_secret:
            return False, "MFA not initialized. Call setup_mfa first."
        if not TOTP.verify(user.mfa_secret, code):
            self.log_audit_event("mfa_enable_failed", email, {}, severity="warning")
            return False, "Invalid TOTP code"
        user.mfa_enabled = True
        self._save_data()
        self.log_audit_event("mfa_enabled", email, {})
        logger.info(f"MFA enabled for {email}")
        return True, "MFA enabled"

    def disable_mfa(self, email: str, code: str) -> Tuple[bool, str]:
        """Disable MFA for a user after verifying a valid TOTP code."""
        user = self.users.get(email)
        if not user:
            return False, "User not found"
        if not user.mfa_enabled:
            return False, "MFA not enabled"
        if not TOTP.verify(user.mfa_secret, code):
            self.log_audit_event("mfa_disable_failed", email, {}, severity="warning")
            return False, "Invalid TOTP code"
        user.mfa_enabled = False
        user.mfa_secret = None
        self._save_data()
        self.log_audit_event("mfa_disabled", email, {})
        logger.info(f"MFA disabled for {email}")
        return True, "MFA disabled"

    def verify_mfa_code(self, email: str, code: str) -> bool:
        """Verify a TOTP code for a user (used during MFA login step)."""
        user = self.users.get(email)
        if not user or not user.mfa_secret:
            return False
        ok = TOTP.verify(user.mfa_secret, code)
        if not ok:
            self.log_audit_event("mfa_login_failed", email, {}, severity="warning")
        return ok

    def mfa_required(self, email: str) -> bool:
        """Return whether MFA is required for a user."""
        user = self.users.get(email)
        return bool(user and user.mfa_enabled)

    def _load_api_keys(self):
        """Load API keys from file."""
        try:
            keys_file = self.user_store_file.replace('.json', '_apikeys.json')
            with open(keys_file, 'r', encoding='utf-8') as f:
                self._api_keys = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._api_keys = {}

    # -------------------- OAuth2 Authorization Server --------------------

    def register_oauth_client(self, name: str, redirect_uris: List[str],
                              scopes: List[str], confidential: bool = True) -> Dict[str, str]:
        """Register a new OAuth2 client. Returns client_id/client_secret."""
        client_id = f"oac_{secrets.token_urlsafe(24)}"
        client_secret = secrets.token_urlsafe(48) if confidential else None
        client = OAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            name=name,
            redirect_uris=redirect_uris,
            scopes=scopes,
        )
        self._oauth_clients[client_id] = client
        self._save_oauth_clients()
        logger.info(f"OAuth2 client registered: {name} ({client_id})")
        return {"client_id": client_id, "client_secret": client_secret}

    def get_oauth_client(self, client_id: str) -> Optional[OAuthClient]:
        """Fetch a registered OAuth2 client by ID."""
        client = self._oauth_clients.get(client_id)
        if client and client.active:
            return client
        return None

    def revoke_oauth_client(self, client_id: str) -> bool:
        """Deactivate an OAuth2 client."""
        client = self._oauth_clients.get(client_id)
        if not client:
            return False
        client.active = False
        self._save_oauth_clients()
        logger.info(f"OAuth2 client revoked: {client_id}")
        return True

    def list_oauth_clients(self) -> List[Dict[str, Any]]:
        """Return registered OAuth clients (excluding client_secret)."""
        return [
            {
                "client_id": c.client_id,
                "name": c.name,
                "redirect_uris": c.redirect_uris,
                "scopes": c.scopes,
                "created_at": c.created_at.isoformat(),
                "active": c.active,
            }
            for c in self._oauth_clients.values()
        ]

    def preauthorize_code(self, client_id: str, user_email: str, redirect_uri: str,
                          code_challenge: Optional[str] = None,
                          code_challenge_method: str = "S256",
                          scope: Optional[List[str]] = None,
                          expires_seconds: int = 600) -> Optional[str]:
        """Create a short-lived, single-use authorization code for a user.

        Called after the resource owner authenticates and grants consent.
        Returns the opaque code, or None if the client/redirect is invalid.
        """
        client = self.get_oauth_client(client_id)
        if not client:
            return None
        if redirect_uri not in client.redirect_uris:
            logger.warning(f"OAuth redirect_uri not registered: {redirect_uri}")
            return None
        if code_challenge_method not in ("S256", "plain"):
            return None

        code = secrets.token_urlsafe(48)
        self._oauth_codes[code] = {
            "client_id": client_id,
            "user_email": user_email,
            "redirect_uri": redirect_uri,
            "scope": scope or client.scopes,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=expires_seconds),
            "used": False,
        }
        self._save_oauth_codes()
        logger.info(f"OAuth authorization code issued to {client_id} for {user_email}")
        return code

    # === OAuth methods continue below ===

    def exchange_code_for_tokens(self, code: str, redirect_uri: str,
                                 code_verifier: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Exchange an authorization code for JWT access/refresh tokens (RFC 6749 §4.1.3).

        Enforces single-use, expiry, redirect_uri match, and PKCE verification.
        Returns a dict with access_token/refresh_token/token_type/expires_in and
        user context, or None on any validation failure.
        """
        record = self._oauth_codes.get(code)
        if not record or record["used"]:
            return None
        if datetime.now(timezone.utc) > record["expires_at"]:
            return None
        if record["redirect_uri"] != redirect_uri:
            logger.warning("OAuth token exchange redirect_uri mismatch")
            return None

        if record.get("code_challenge"):
            if not code_verifier:
                logger.warning("OAuth token exchange missing PKCE code_verifier")
                return None
            if record["code_challenge_method"] == "S256":
                if pkce_s256_challenge(code_verifier) != record["code_challenge"]:
                    logger.warning("OAuth PKCE S256 verification failed")
                    return None
            else:  # plain
                if code_verifier != record["code_challenge"]:
                    logger.warning("OAuth PKCE plain verification failed")
                    return None

        record["used"] = True
        self._save_oauth_codes()

        user = self.users.get(record["user_email"])
        if not user or not user.active:
            return None

        access_token, refresh_token = self.generate_tokens(user)
        granted = record["scope"] or user.permissions
        self.log_audit_event("oauth_token_exchange", user.email,
                             {"client_id": record["client_id"], "scope": granted})
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "scope": granted,
            "email": user.email,
        }

    @property
    def _oauth_clients(self) -> Dict[str, OAuthClient]:
        if not hasattr(self, "_oauth_clients_store"):
            self._oauth_clients_store = {}
            fname = self.user_store_file.replace('.json', '_oauth_clients.json')
            try:
                if os.path.exists(fname):
                    with open(fname, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                        self._oauth_clients_store = {
                            cid: OAuthClient.from_dict(data) for cid, data in raw.items()
                        }
            except Exception:
                self._oauth_clients_store = {}
        return self._oauth_clients_store

    @property
    def _oauth_codes(self) -> Dict[str, Dict[str, Any]]:
        if not hasattr(self, "_oauth_codes_store"):
            self._oauth_codes_store = {}
            fname = self.user_store_file.replace('.json', '_oauth_codes.json')
            try:
                if os.path.exists(fname):
                    with open(fname, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                        for rec in raw.values():
                            if rec.get("expires_at") and isinstance(rec["expires_at"], str):
                                rec["expires_at"] = datetime.fromisoformat(rec["expires_at"])
                        self._oauth_codes_store = raw
            except Exception:
                self._oauth_codes_store = {}
        return self._oauth_codes_store

    def _save_oauth_clients(self):
        try:
            fname = self.user_store_file.replace('.json', '_oauth_clients.json')
            data = {cid: c.to_dict() for cid, c in self._oauth_clients.items()}
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.warning("Failed to save OAuth clients")

    def _save_oauth_codes(self):
        try:
            fname = self.user_store_file.replace('.json', '_oauth_codes.json')
            data = dict(self._oauth_codes)
            for rec in data.values():
                if isinstance(rec.get("expires_at"), datetime):
                    rec["expires_at"] = rec["expires_at"].isoformat()
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.warning("Failed to save OAuth authorization codes")

auth_manager = AuthManager()

# Convenience functions
def authenticate_user(email: str, password: str, ip_address: str = None, user_agent: str = None):
    return auth_manager.authenticate_user(email, password, ip_address, user_agent)

def verify_token(token: str):
    return auth_manager.verify_access_token(token)

def create_user(email: str, username: str, password: str, role: str = 'user',
               company: str = 'OWLBAN_GROUP', permissions: List[str] = None):
    return auth_manager.create_user(email, username, password, role, company, permissions)

def get_user_by_email(email: str):
    return auth_manager.get_user_by_email(email)

def request_password_reset(email: str):
    return auth_manager.create_password_reset_token(email)

def reset_password(reset_token: str, new_password: str):
    return auth_manager.reset_password(reset_token, new_password)

def generate_api_key(email: str, name: str = "default"):
    return auth_manager.generate_api_key(email, name)

def verify_api_key(api_key: str):
    return auth_manager.verify_api_key(api_key)

def setup_mfa(email: str):
    return auth_manager.setup_mfa(email)

def enable_mfa(email: str, code: str):
    return auth_manager.enable_mfa(email, code)

def disable_mfa(email: str, code: str):
    return auth_manager.disable_mfa(email, code)

def verify_mfa_code(email: str, code: str):
    return auth_manager.verify_mfa_code(email, code)

def mfa_required(email: str):
    return auth_manager.mfa_required(email)

def register_oauth_client(name: str, redirect_uris: List[str], scopes: List[str],
                          confidential: bool = True):
    return auth_manager.register_oauth_client(name, redirect_uris, scopes, confidential)

def get_oauth_client(client_id: str):
    return auth_manager.get_oauth_client(client_id)

def list_oauth_clients():
    return auth_manager.list_oauth_clients()

def revoke_oauth_client(client_id: str):
    return auth_manager.revoke_oauth_client(client_id)

def preauthorize_code(client_id: str, user_email: str, redirect_uri: str,
                      code_challenge: str = None, code_challenge_method: str = "S256",
                      scope: List[str] = None, expires_seconds: int = 600):
    return auth_manager.preauthorize_code(client_id, user_email, redirect_uri,
                                          code_challenge, code_challenge_method,
                                          scope, expires_seconds)

def exchange_code_for_tokens(code: str, redirect_uri: str, code_verifier: str = None):
    return auth_manager.exchange_code_for_tokens(code, redirect_uri, code_verifier)

if __name__ == '__main__':
    # Test the auth system
    print("Testing OWLBAN GROUP Authentication System")

    # Create a test user
    success, message = create_user('test@owlban.com', 'testuser', 'TestPass123!', 'user', 'OWLBAN_GROUP')
    print(f"Create user: {success} - {message}")

    # Test authentication
    success, message, user = authenticate_user('test@owlban.com', 'TestPass123!')
    print(f"Authenticate: {success} - {message}")

    if user:
        # Generate tokens
        access_token, refresh_token = auth_manager.generate_tokens(user)
        print(f"Access token: {access_token[:20]}...")
        print(f"Refresh token: {refresh_token[:20]}...")

        # Verify token
        payload = verify_token(access_token)
        print(f"Token valid: {payload is not None}")
        if payload:
            print(f"User from token: {payload['email']}")
