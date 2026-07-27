"""
Configuration settings for JP Morgan Financial APIs
"""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra='allow'
    )

    @property
    def environment(self) -> str:
        """Backward-compatible lowercase alias expected by older tests/modules."""
        return self.FLASK_ENV

    @property
    def debug(self) -> bool:
        """Backward-compatible lowercase alias expected by older tests/modules."""
        return self.DEBUG

    @property
    def version(self) -> str:
        """Backward-compatible lowercase alias expected by older tests/modules."""
        return self.VERSION

    @property
    def max_concurrent_requests(self) -> int:
        """Backward-compatible lowercase alias expected by telemetry processor."""
        return self.MAX_CONCURRENT_REQUESTS

    @property
    def max_batch_size(self) -> int:
        """Backward-compatible lowercase alias expected by telemetry batch processor."""
        return self.TELEMETRY_BATCH_SIZE

    # Application
    APP_NAME: str = "JPMorgan Financial APIs"
    DEBUG: bool = False
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/jpmorgan_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # GitHub
    GITHUB_TOKEN: str = ""

    # JPMorgan API
    JPMORGAN_API_KEY: str = ""
    JPMORGAN_API_SECRET: str = ""
    JPMORGAN_BASE_URL: str = "https://api.jpmorgan.com"

    # Telemetry
    TELEMETRY_BATCH_SIZE: int = 100
    MAX_CONCURRENT_REQUESTS: int = 100

    # Additional settings for compatibility
    OAUTH_CLIENT_SECRET: str = ""
    FLASK_ENV: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 5000


settings = Settings()
