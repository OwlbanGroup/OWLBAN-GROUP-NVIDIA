"""Application configuration for the production-ready OWLBAN layer."""

from __future__ import annotations

import os
import json
import secrets
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Simple runtime configuration container."""

    api_username: str = "owlban_admin"
    api_password: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    cors_origins: tuple[str, ...] = ()
    database_path: str = "owlban_app.db"
    rate_limit_per_minute: int = 60
    api_keys: tuple[tuple[str, str], ...] = ()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings from environment variables."""
    environment = os.getenv("APP_ENV", "development").strip().lower()
    production = environment == "production"
    username = os.getenv("API_USERNAME", "owlban_admin")
    password = os.getenv("API_PASSWORD", "")
    jwt_secret = os.getenv("JWT_SECRET", "")
    if production:
        if not password:
            raise ValueError("API_PASSWORD must be set when APP_ENV=production")
        if password.startswith("replace-with-"):
            raise ValueError("Replace the API_PASSWORD example value before production deployment")
        if len(jwt_secret.encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET must be at least 32 bytes when APP_ENV=production")
        if jwt_secret.startswith("replace-with-"):
            raise ValueError("Replace the JWT_SECRET example value before production deployment")
    else:
        password = password or secrets.token_urlsafe(24)
        jwt_secret = jwt_secret or secrets.token_urlsafe(48)

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    configured_origins = os.getenv("CORS_ORIGINS")
    if configured_origins is None:
        origins = () if production else ("http://localhost:3000", "http://localhost:8501")
    else:
        origins = tuple(origin.strip() for origin in configured_origins.split(",") if origin.strip())

    if "*" in origins:
        raise ValueError("CORS_ORIGINS must not contain '*'")

    raw_api_keys = os.getenv("APP_API_KEYS", "{}")
    try:
        api_keys_data = json.loads(raw_api_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("APP_API_KEYS must be a JSON object mapping API keys to subjects") from exc
    if not isinstance(api_keys_data, dict) or any(
        not isinstance(key, str) or not key or not isinstance(subject, str) or not subject
        for key, subject in api_keys_data.items()
    ):
        raise ValueError("APP_API_KEYS must map non-empty string keys to non-empty subjects")

    return Settings(
        api_username=username,
        api_password=password,
        api_host=host,
        api_port=port,
        debug=debug,
        environment=environment,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        jwt_secret=jwt_secret,
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        cors_origins=origins,
        database_path=os.getenv("APP_DB_PATH", "owlban_app.db"),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        api_keys=tuple(api_keys_data.items()),
    )
