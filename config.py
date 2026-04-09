"""
Configuration settings for JPMorgan Financial APIs
"""
import os
from typing import Dict, Any, Optional


class Config:
    """Configuration class for the application"""

    def __init__(self) -> None:
        """Initialize configuration with all required attributes as instance attributes"""
        # API Settings
        self.API_BASE_URL: str = os.getenv('API_BASE_URL', 'https://api.jpmorgan.com')
        self.API_VERSION: str = os.getenv('API_VERSION', 'v1')

        # JPMorgan Environment Configuration
        self.JPMORGAN_ENVIRONMENT: str = os.getenv('JPMORGAN_ENVIRONMENT', 'dev')

        # JPMorgan Merchant API Endpoints (Treasury Services API)
        self.JPMORGAN_MERCHANT_PRODUCTION_URL: str = os.getenv(
            'JPMORGAN_MERCHANT_PRODUCTION_URL',
            'https://api.merchant.jpmorgan.com/tsapi/v1'
        )
        self.JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL: str = os.getenv(
            'JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL',
            'https://api-mtls.merchant.jpmorgan.com/tsapi/v1'
        )
        self.JPMORGAN_MERCHANT_UAT_URL: str = os.getenv(
            'JPMORGAN_MERCHANT_UAT_URL',
            'https://api-pci-uat.jpmorgan.com/tsapi/v1'
        )
        self.JPMORGAN_MERCHANT_MTLS_UAT_URL: str = os.getenv(
            'JPMORGAN_MERCHANT_MTLS_UAT_URL',
            'https://api-mtls-pci-uat.jpmorgan.com/tsapi/v1'
        )

        # JPMorgan OpenBanking API Endpoints (Legacy)
        self.JPMORGAN_OPENBANKING_PRODUCTION_URL: str = os.getenv(
            'JPMORGAN_OPENBANKING_PRODUCTION_URL',
            'https://openbanking.jpmorgan.com/accessapi'
        )
        self.JPMORGAN_OPENBANKING_UAT_URL: str = os.getenv(
            'JPMORGAN_OPENBANKING_UAT_URL',
            'https://openbankinguat.jpmorgan.com/accessapi'
        )

        # JPMorgan API Gateway Endpoints (Legacy)
        self.JPMORGAN_APIGATEWAY_PRODUCTION_URL: str = os.getenv(
            'JPMORGAN_APIGATEWAY_PRODUCTION_URL',
            'https://apigateway.jpmorgan.com/accessapi'
        )
        self.JPMORGAN_APIGATEWAY_QAF_URL: str = os.getenv(
            'JPMORGAN_APIGATEWAY_QAF_URL',
            'https://apigatewayqaf.jpmorgan.com/accessapi'
        )

        # OpenBanking API Credentials
        self.JPMORGAN_OPENBANKING_CLIENT_ID: Optional[str] = os.getenv('JPMORGAN_OPENBANKING_CLIENT_ID')
        self.JPMORGAN_OPENBANKING_CLIENT_SECRET: Optional[str] = os.getenv('JPMORGAN_OPENBANKING_CLIENT_SECRET')
        self.JPMORGAN_OPENBANKING_API_KEY: Optional[str] = os.getenv('JPMORGAN_OPENBANKING_API_KEY')

        # API Gateway Credentials
        self.JPMORGAN_APIGATEWAY_CLIENT_ID: Optional[str] = os.getenv('JPMORGAN_APIGATEWAY_CLIENT_ID')
        self.JPMORGAN_APIGATEWAY_CLIENT_SECRET: Optional[str] = os.getenv('JPMORGAN_APIGATEWAY_CLIENT_SECRET')
        self.JPMORGAN_APIGATEWAY_API_KEY: Optional[str] = os.getenv('JPMORGAN_APIGATEWAY_API_KEY')

        # Logging Settings
        self.LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE: str = os.getenv('LOG_FILE', 'logs/telemetry.log')

        # Telemetry Settings
        self.TELEMETRY_ENABLED: bool = os.getenv('TELEMETRY_ENABLED', 'true').lower() == 'true'
        self.TELEMETRY_BATCH_SIZE: int = int(os.getenv('TELEMETRY_BATCH_SIZE', '100'))

        # Database URL for telemetry
        self.DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///telemetry.db')

        # Additional compatibility settings
        self.OAUTH_CLIENT_SECRET: str = os.getenv('OAUTH_CLIENT_SECRET', '')
        self.FLASK_ENV: str = os.getenv('FLASK_ENV', 'development')
        self.HOST: str = os.getenv('HOST', '127.0.0.1')
        self.PORT: int = int(os.getenv('PORT', '5000'))

        # Database Settings
        self.DATABASE_TYPE: str = os.getenv('DATABASE_TYPE', 'sqlite')
        self.DATABASE_HOST: str = os.getenv('DATABASE_HOST', 'localhost')
        self.DATABASE_PORT: int = int(os.getenv('DATABASE_PORT', '5432'))
        self.DATABASE_NAME: str = os.getenv('DATABASE_NAME', 'jpmorgan_financial_apis')
        self.DATABASE_USER: str = os.getenv('DATABASE_USER', '')
        self.DATABASE_PASSWORD: str = os.getenv('DATABASE_PASSWORD', '')
        self.DATABASE_SSL_MODE: str = os.getenv('DATABASE_SSL_MODE', 'require')
        self.DATABASE_CONNECTION_POOL_SIZE: int = int(os.getenv('DATABASE_CONNECTION_POOL_SIZE', '10'))
        self.DATABASE_CONNECTION_POOL_MAX_OVERFLOW: int = int(os.getenv('DATABASE_CONNECTION_POOL_MAX_OVERFLOW', '20'))
        self.DATABASE_CONNECTION_POOL_TIMEOUT: int = int(os.getenv('DATABASE_CONNECTION_POOL_TIMEOUT', '30'))
        self.DATABASE_CONNECTION_POOL_RECYCLE: int = int(os.getenv('DATABASE_CONNECTION_POOL_RECYCLE', '3600'))

        # Redis Settings
        self.REDIS_URL: Optional[str] = os.getenv('REDIS_URL')

        # Token Management Settings
        self.TOKEN_CLIENT_ID: Optional[str] = os.getenv('TOKEN_CLIENT_ID')
        self.TOKEN_CLIENT_SECRET: Optional[str] = os.getenv('TOKEN_CLIENT_SECRET')
        self.TOKEN_URL: str = os.getenv(
            'TOKEN_URL',
            'https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token'
        )
        self.TOKEN_SCOPE: str = os.getenv('TOKEN_SCOPE', 'openid profile')

        # GitHub MCP Server Settings
        self.MCP_SERVER_COMMAND: str = os.getenv(
            'MCP_SERVER_COMMAND',
            'docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN '
            'ghcr.io/github/github-mcp-server'
        )
        self.GITHUB_PERSONAL_ACCESS_TOKEN: Optional[str] = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        self.MCP_SERVER_TOOLSETS: str = os.getenv('MCP_SERVER_TOOLSETS', 'all')
        self.MCP_SERVER_HOST: str = os.getenv('MCP_SERVER_HOST', '')

        # NGC (NVIDIA GPU Cloud) Settings
        self.NGC_API_KEY: Optional[str] = os.getenv('NGC_API_KEY')
        self.NGC_CLI_PATH: str = os.getenv('NGC_CLI_PATH', 'ngc')
        self.NGC_REGISTRY_URL: str = os.getenv('NGC_REGISTRY_URL', 'nvcr.io')
        self.NGC_ORG: str = os.getenv('NGC_ORG', 'nvidia')

        # GPU Settings
        self.NVIDIA_VISIBLE_DEVICES: str = os.getenv('NVIDIA_VISIBLE_DEVICES', 'all')
        self.NVIDIA_DRIVER_CAPABILITIES: str = os.getenv('NVIDIA_DRIVER_CAPABILITIES', 'compute,utility,video')
        self.CUDA_VISIBLE_DEVICES: str = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        self.GPU_MEMORY_FRACTION: float = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))

        # Multi-GPU Support Configuration
        self.MULTI_GPU_ENABLED: bool = os.getenv('MULTI_GPU_ENABLED', 'false').lower() == 'true'
        self.GPU_COUNT: int = int(os.getenv('GPU_COUNT', '1'))
        self.GPU_STRATEGY: str = os.getenv('GPU_STRATEGY', 'mirrored')
        self.GPU_MEMORY_GROWTH: bool = os.getenv('GPU_MEMORY_GROWTH', 'true').lower() == 'true'
        self.GPU_PER_PROCESS_MEMORY_FRACTION: float = float(os.getenv('GPU_PER_PROCESS_MEMORY_FRACTION', '0.8'))
        self.GPU_ALLOW_GROWTH: bool = os.getenv('GPU_ALLOW_GROWTH', 'true').lower() == 'true'

        # TensorFlow/Keras GPU Configuration
        self.TF_GPU_MEMORY_LIMIT_MB: int = int(os.getenv('TF_GPU_MEMORY_LIMIT_MB', '4096'))
        self.TF_FORCE_GPU_ALLOW_GROWTH: bool = os.getenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true').lower() == 'true'
        self.TF_VISIBLE_DEVICES: str = os.getenv('TF_VISIBLE_DEVICES', '0')

        # PyTorch GPU Configuration
        self.TORCH_GPU_COUNT: int = int(os.getenv('TORCH_GPU_COUNT', '1'))
        self.TORCH_CUDA_VISIBLE_DEVICES: str = os.getenv('TORCH_CUDA_VISIBLE_DEVICES', '0')
        self.TORCH_DISTRIBUTED_BACKEND: str = os.getenv('TORCH_DISTRIBUTED_BACKEND', 'nccl')

        # Security Settings - test-friendly fallback
        secret_key = os.getenv('SECRET_KEY')
        if not secret_key:
            secret_key = 'dummy_secret_key_for_testing'
        self.SECRET_KEY: str = secret_key

        # JWT Settings
        self.JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY', self.SECRET_KEY)
        self.JWT_ALGORITHM: str = os.getenv('JWT_ALGORITHM', 'HS256')
        self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
        self.JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7'))

        # Rate Limiting Settings
        self.RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        self.RATE_LIMIT_DEFAULT: str = os.getenv('RATE_LIMIT_DEFAULT', '100 per minute')
        self.RATE_LIMIT_STORAGE_URL: str = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')
        self.RATE_LIMIT_STRATEGY: str = os.getenv('RATE_LIMIT_STRATEGY', 'fixed-window')
        self.RATE_LIMIT_HEADERS_ENABLED: bool = os.getenv('RATE_LIMIT_HEADERS_ENABLED', 'true').lower() == 'true'
        self.RATE_LIMIT_AUTH: str = os.getenv('RATE_LIMIT_AUTH', '5 per minute')
        self.RATE_LIMIT_API: str = os.getenv('RATE_LIMIT_API', '100 per minute')
        self.RATE_LIMIT_PAYMENTS: str = os.getenv('RATE_LIMIT_PAYMENTS', '50 per minute')
        self.RATE_LIMIT_TELEMETRY: str = os.getenv('RATE_LIMIT_TELEMETRY', '200 per minute')
        self.RATE_LIMIT_ADMIN: str = os.getenv('RATE_LIMIT_ADMIN', '20 per minute')

        # Caching Settings
        self.CACHE_ENABLED: bool = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.CACHE_TYPE: str = os.getenv('CACHE_TYPE', 'simple')
        self.CACHE_DEFAULT_TIMEOUT: int = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '300'))
        self.CACHE_THRESHOLD: int = int(os.getenv('CACHE_THRESHOLD', '500'))
        self.CACHE_REDIS_URL: Optional[str] = os.getenv('CACHE_REDIS_URL') or self.REDIS_URL
        self.CACHE_KEY_PREFIX: str = os.getenv('CACHE_KEY_PREFIX', 'jpmorgan_api_')

        # CORS Settings
        allowed_origins_str = os.getenv('ALLOWED_ORIGINS')
        if not allowed_origins_str:
            self.ALLOWED_ORIGINS = [
                'https://app.jpmorgan.com',
                'https://dashboard.jpmorgan.com',
                'https://api.jpmorgan.com'
            ]
        else:
            self.ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]
            if os.getenv('FLASK_ENV') == 'production':
                self.ALLOWED_ORIGINS = [
                    origin for origin in self.ALLOWED_ORIGINS
                    if not origin.startswith('http://localhost') and not origin.startswith('http://127.0.0.1')
                ]

        # Audit Logging Settings
        self.AUDIT_LOG_ENABLED: bool = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'
        self.AUDIT_LOG_RETENTION_DAYS: int = int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))
        self.AUDIT_LOG_MAX_SIZE: int = int(os.getenv('AUDIT_LOG_MAX_SIZE', '10000000'))
        self.AUDIT_ALERT_ENABLED: bool = os.getenv('AUDIT_ALERT_ENABLED', 'true').lower() == 'true'
        self.AUDIT_FAILED_LOGIN_THRESHOLD: int = int(os.getenv('AUDIT_FAILED_LOGIN_THRESHOLD', '5'))
        self.AUDIT_RATE_LIMIT_THRESHOLD: int = int(os.getenv('AUDIT_RATE_LIMIT_THRESHOLD', '100'))
        self.AUDIT_BRUTE_FORCE_THRESHOLD: int = int(os.getenv('AUDIT_BRUTE_FORCE_THRESHOLD', '10'))
        self.AUDIT_SUSPICIOUS_IP_THRESHOLD: int = int(os.getenv('AUDIT_SUSPICIOUS_IP_THRESHOLD', '5'))
        self.AUDIT_ALERT_NOTIFICATION_METHOD: str = os.getenv('AUDIT_ALERT_NOTIFICATION_METHOD', 'log')
        self.AUDIT_CLEANUP_ENABLED: bool = os.getenv('AUDIT_CLEANUP_ENABLED', 'true').lower() == 'true'
        self.AUDIT_HASH_CHAIN_ENABLED: bool = os.getenv('AUDIT_HASH_CHAIN_ENABLED', 'true').lower() == 'true'

        # Auth0 Authentication Settings
        self.AUTH0_DOMAIN: Optional[str] = os.getenv('AUTH0_DOMAIN')
        self.AUTH0_CLIENT_ID: Optional[str] = os.getenv('AUTH0_CLIENT_ID')
        self.AUTH0_CLIENT_SECRET: Optional[str] = os.getenv('AUTH0_CLIENT_SECRET')
        self.AUTH0_AUDIENCE: Optional[str] = os.getenv('AUTH0_AUDIENCE')
        self.AUTH0_ALGORITHMS: list = os.getenv('AUTH0_ALGORITHMS', 'RS256').split(',')
        self.AUTH0_ISSUER: Optional[str] = f"https://{self.AUTH0_DOMAIN}/" if self.AUTH0_DOMAIN else None
        self.AUTH0_JWKS_URL: Optional[str] = f"https://{self.AUTH0_DOMAIN}/.well-known/jwks.json" if self.AUTH0_DOMAIN else None

        # LangSmith Settings for AI tracing and monitoring
        self.LANGCHAIN_API_KEY: Optional[str] = os.getenv('LANGCHAIN_API_KEY')
        self.LANGCHAIN_PROJECT: str = os.getenv('LANGCHAIN_PROJECT', 'jpmorgan-financial-apis')
        self.LANGCHAIN_ENDPOINT: str = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')

        # OpenAI Settings for LangChain
        self.OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
        self.OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.OPENAI_TEMPERATURE: float = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))

        # Blackbox AI Settings (OpenAI-compatible)
        self.BLACKBOX_BASE_URL: str = os.getenv('BLACKBOX_BASE_URL', 'https://cloud.blackbox.ai/')
        self.BLACKBOX_MODEL: str = os.getenv('BLACKBOX_MODEL', 'gpt-3.5-turbo')
        self.BLACKBOX_TEMPERATURE: float = float(os.getenv('BLACKBOX_TEMPERATURE', '0.1'))
        self.BLACKBOX_API_KEY: Optional[str] = os.getenv('BLACKBOX_API_KEY')

        # Stripe Payment Processing Settings
        self.STRIPE_PUBLISHABLE_KEY: Optional[str] = os.getenv('STRIPE_PUBLISHABLE_KEY')
        self.STRIPE_SECRET_KEY: Optional[str] = os.getenv('STRIPE_SECRET_KEY')
        self.STRIPE_WEBHOOK_SECRET: Optional[str] = os.getenv('STRIPE_WEBHOOK_SECRET')
        self.STRIPE_API_VERSION: str = os.getenv('STRIPE_API_VERSION', '2023-10-16')
        self.STRIPE_CURRENCY: str = os.getenv('STRIPE_CURRENCY', 'usd')

    def get_database_url(self) -> str:
        """Generate database URL based on configuration"""
        if self.DATABASE_TYPE == 'postgresql':
            return (
                f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
                f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/"
                f"{self.DATABASE_NAME}"
            )
        return self.DATABASE_URL

    def get_jpmorgan_endpoint_url(self, service: str, use_mtls: bool = False) -> str:
        """Get JPMorgan endpoint URL based on environment and service"""
        environment = self.JPMORGAN_ENVIRONMENT.lower()

        if service == 'merchant':
            if environment in ['dev', 'staging']:
                return self.JPMORGAN_MERCHANT_MTLS_UAT_URL if use_mtls else self.JPMORGAN_MERCHANT_UAT_URL
            return self.JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL if use_mtls else self.JPMORGAN_MERCHANT_PRODUCTION_URL
        if service == 'openbanking':
            if environment in ['dev', 'staging']:
                return self.JPMORGAN_OPENBANKING_UAT_URL
            return self.JPMORGAN_OPENBANKING_PRODUCTION_URL
        if service == 'apigateway':
            if environment == 'qaf':
                return self.JPMORGAN_APIGATEWAY_QAF_URL
            return self.JPMORGAN_APIGATEWAY_PRODUCTION_URL

        raise ValueError("Unknown service: {service}. Must be 'merchant', 'openbanking', or 'apigateway'")

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        return {
            'api_base_url': self.API_BASE_URL,
            'api_version': self.API_VERSION,
            'log_level': self.LOG_LEVEL,
            'log_file': self.LOG_FILE,
            'telemetry_enabled': self.TELEMETRY_ENABLED,
            'telemetry_batch_size': self.TELEMETRY_BATCH_SIZE,
            'database_url': self.get_database_url(),
            'database_type': self.DATABASE_TYPE,
            'database_host': self.DATABASE_HOST,
            'database_port': self.DATABASE_PORT,
            'database_name': self.DATABASE_NAME,
            'database_connection_pool_size': self.DATABASE_CONNECTION_POOL_SIZE,
            'database_connection_pool_max_overflow': self.DATABASE_CONNECTION_POOL_MAX_OVERFLOW,
            'database_connection_pool_timeout': self.DATABASE_CONNECTION_POOL_TIMEOUT,
            'database_connection_pool_recycle': self.DATABASE_CONNECTION_POOL_RECYCLE,
            'token_client_id': self.TOKEN_CLIENT_ID,
            'token_client_secret': self.TOKEN_CLIENT_SECRET,
            'token_url': self.TOKEN_URL,
            'token_scope': self.TOKEN_SCOPE,
            'mcp_server_command': self.MCP_SERVER_COMMAND,
            'github_personal_access_token': self.GITHUB_PERSONAL_ACCESS_TOKEN,
            'mcp_server_toolsets': self.MCP_SERVER_TOOLSETS,
            'mcp_server_host': self.MCP_SERVER_HOST,
            'jpmorgan_environment': self.JPMORGAN_ENVIRONMENT,
            'jpmorgan_merchant_production_url': self.JPMORGAN_MERCHANT_PRODUCTION_URL,
            'jpmorgan_merchant_mtls_production_url': self.JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL,
            'jpmorgan_merchant_uat_url': self.JPMORGAN_MERCHANT_UAT_URL,
            'jpmorgan_merchant_mtls_uat_url': self.JPMORGAN_MERCHANT_MTLS_UAT_URL,
            'jpmorgan_openbanking_production_url': self.JPMORGAN_OPENBANKING_PRODUCTION_URL,
            'jpmorgan_openbanking_uat_url': self.JPMORGAN_OPENBANKING_UAT_URL,
            'jpmorgan_apigateway_production_url': self.JPMORGAN_APIGATEWAY_PRODUCTION_URL,
            'jpmorgan_apigateway_qaf_url': self.JPMORGAN_APIGATEWAY_QAF_URL,
            'audit_log_enabled': self.AUDIT_LOG_ENABLED,
            'audit_log_retention_days': self.AUDIT_LOG_RETENTION_DAYS,
            'audit_alert_enabled': self.AUDIT_ALERT_ENABLED,
            'audit_failed_login_threshold': self.AUDIT_FAILED_LOGIN_THRESHOLD,
            'audit_rate_limit_threshold': self.AUDIT_RATE_LIMIT_THRESHOLD,
            'auth0_domain': self.AUTH0_DOMAIN,
            'auth0_client_id': self.AUTH0_CLIENT_ID,
            'auth0_client_secret': self.AUTH0_CLIENT_SECRET,
            'auth0_audience': self.AUTH0_AUDIENCE,
            'auth0_algorithms': self.AUTH0_ALGORITHMS,
            'auth0_issuer': self.AUTH0_ISSUER,
            'auth0_jwks_url': self.AUTH0_JWKS_URL,
            'stripe_publishable_key': self.STRIPE_PUBLISHABLE_KEY,
            'stripe_secret_key': self.STRIPE_SECRET_KEY,
            'stripe_webhook_secret': self.STRIPE_WEBHOOK_SECRET,
            'stripe_api_version': self.STRIPE_API_VERSION,
            'stripe_currency': self.STRIPE_CURRENCY
        }


config = Config()
