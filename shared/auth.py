"""
Shared authentication utilities
"""
from typing import Optional
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
import os

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: list[str] = []


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify JWT token and return token data"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        user_id: Optional[str] = payload.get("user_id")
        scopes: list[str] = payload.get("scopes", [])

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenData(username=username, user_id=user_id, scopes=scopes)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Dependency to require authentication"""
    token = credentials.credentials
    return verify_token(token)


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """Dependency for optional authentication"""
    if credentials is None:
        return None
    token = credentials.credentials
    try:
        return verify_token(token)
    except HTTPException:
        return None


class AuthService:
    """Authentication service wrapper for token operations."""

    def create_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        return create_access_token(data, expires_delta)

    def verify_token(self, token: str) -> TokenData:
        return verify_token(token)


class AuthorizationService:
    """Authorization service wrapper for role/scope checks."""

    def __init__(self):
        self.auth_service = AuthService()

    def has_scope(self, token_data: TokenData, required_scope: str) -> bool:
        return required_scope in (token_data.scopes or [])

    def has_any_scope(self, token_data: TokenData, required_scopes: list[str]) -> bool:
        token_scopes = set(token_data.scopes or [])
        return any(scope in token_scopes for scope in required_scopes)
