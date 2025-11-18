"""
Configuration settings for JP Morgan Financial APIs
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

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

    class Config:
        """Pydantic config"""
        env_file = ".env"
        case_sensitive = True


settings = Settings()
