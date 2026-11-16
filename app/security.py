"""Security helpers for the production-ready app layer."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings

security = HTTPBasic(auto_error=False)


def _validate_basic_auth(credentials: HTTPBasicCredentials | None) -> str:
    settings = get_settings()
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic realm=owlban"},
        )

    valid_username = secrets.compare_digest(credentials.username, settings.api_username)
    valid_password = secrets.compare_digest(credentials.password, settings.api_password)
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic realm=owlban"},
        )

    return credentials.username


def _validate_bearer_token(token: str) -> str:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp"]},
        )
    except jwt.PyJWTError as exc:  # pragma: no cover - defensive branch
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        ) from exc

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token missing subject",
        )
    return str(subject)


def _generate_bearer_token(subject: str) -> str:
    settings = get_settings()
    issued_at = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + timedelta(minutes=60)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_credentials(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(security)],
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
) -> str:
    """Validate Basic Auth, bearer tokens, or API keys for protected routes."""
    if credentials is not None:
        return _validate_basic_auth(credentials)

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return _validate_bearer_token(token)

    if api_key:
        settings = get_settings()
        for configured_key, subject in settings.api_keys:
            if secrets.compare_digest(api_key, configured_key):
                return subject

        if settings.environment != "production":
            from auth_lib import verify_api_key as verify_api_key_record

            record = verify_api_key_record(api_key)
            if record is not None:
                email = record.get("email") or record.get("user")
                if isinstance(email, dict):
                    email = email.get("email")
                if email:
                    return str(email)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic realm=owlban"},
    )


__all__ = [
    "verify_credentials",
    "_generate_bearer_token",
    "_validate_bearer_token",
]
