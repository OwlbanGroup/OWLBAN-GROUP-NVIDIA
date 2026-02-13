"""
Configuration settings for JPMorgan Financial APIs
"""
import os
from typing import Dict, Any

class Config:
    """Configuration class for the application"""

    # API Settings
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.jpmorgan.com')
    API_VERSION = os.getenv('API_VERSION', 'v1')

    # JPMorgan Environment Configuration
    JPMORGAN_ENVIRONMENT = os.getenv('JPMORGAN_ENVIRONMENT', 'dev')  # dev, staging, prod

    # JPMorgan Merchant API Endpoints (Treasury Services API)
    JPMORGAN_MERCHANT_PRODUCTION_URL = os.getenv(
        'JPMORGAN_MERCHANT_PRODUCTION_URL',
        'https://api.merchant.jpmorgan.com/tsapi/v1'
    )
    JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL = os.getenv(
        'JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL',
        'https://api-mtls.merchant.jpmorgan.com/tsapi/v1'
    )
    JPMORGAN_MERCHANT_UAT_URL = os.getenv(
        'JPMORGAN_MERCHANT_UAT_URL',
        'https://api-pci-uat.jpmorgan.com/tsapi/v1'
    )
    JPMORGAN_MERCHANT_MTLS_UAT_URL = os.getenv(
        'JPMORGAN_MERCHANT_MTLS_UAT_URL',
        'https://api-mtls-pci-uat.jpmorgan.com/tsapi/v1'
    )

    # JPMorgan OpenBanking API Endpoints (Legacy)
    JPMORGAN_OPENBANKING_PRODUCTION_URL = os.getenv(
        'JPMORGAN_OPENBANKING_PRODUCTION_URL',
        'https://openbanking.jpmorgan.com/accessapi'
    )
    JPMORGAN_OPENBANKING_UAT_URL = os.getenv(
        'JPMORGAN_OPENBANKING_UAT_URL',
        'https://openbankinguat.jpmorgan.com/accessapi'
    )

    # JPMorgan API Gateway Endpoints (Legacy)
    JPMORGAN_APIGATEWAY_PRODUCTION_URL = os.getenv(
        'JPMORGAN_APIGATEWAY_PRODUCTION_URL',
        'https://apigateway.jpmorgan.com/accessapi'
    )
    JPMORGAN_APIGATEWAY_QAF_URL = os.getenv(
        'JPMORGAN_APIGATEWAY_QAF_URL',
        'https://apigatewayqaf.jpmorgan.com/accessapi'
    )

    # OpenBanking API Credentials
    JPMORGAN_OPENBANKING_CLIENT_ID = os.getenv('JPMORGAN_OPENBANKING_CLIENT_ID')
    JPMORGAN_OPENBANKING_CLIENT_SECRET = os.getenv('JPMORGAN_OPENBANKING_CLIENT_SECRET')
    JPMORGAN_OPENBANKING_API_KEY = os.getenv('JPMORGAN_OPENBANKING_API_KEY')

    # API Gateway Credentials
    JPMORGAN_APIGATEWAY_CLIENT_ID = os.getenv('JPMORGAN_APIGATEWAY_CLIENT_ID')
    JPMORGAN_APIGATEWAY_CLIENT_SECRET = os.getenv('JPMORGAN_APIGATEWAY_CLIENT_SECRET')
    JPMORGAN_APIGATEWAY_API_KEY = os.getenv('JPMORGAN_APIGATEWAY_API_KEY')

    # Logging Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/telemetry.log')

    # Telemetry Settings
    TELEMETRY_ENABLED = os.getenv('TELEMETRY_ENABLED', 'true').lower() == 'true'
    TELEMETRY_BATCH_SIZE = int(os.getenv('TELEMETRY_BATCH_SIZE', '100'))

    # Database URL for telemetry
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///telemetry.db')

    # Additional compatibility settings
    OAUTH_CLIENT_SECRET = os.getenv('OAUTH_CLIENT_SECRET', '')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', '5000'))

    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///telemetry.db')
    DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'sqlite')  # sqlite, postgresql
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
    DATABASE_PORT = int(os.getenv('DATABASE_PORT', '5432'))
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'jpmorgan_financial_apis')
    DATABASE_USER = os.getenv('DATABASE_USER', '')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', '')
    DATABASE_SSL_MODE = os.getenv('DATABASE_SSL_MODE', 'require')
    DATABASE_CONNECTION_POOL_SIZE = int(
        os.getenv('DATABASE_CONNECTION_POOL_SIZE', '10'))
    DATABASE_CONNECTION_POOL_MAX_OVERFLOW = int(
        os.getenv('DATABASE_CONNECTION_POOL_MAX_OVERFLOW', '20'))
    DATABASE_CONNECTION_POOL_TIMEOUT = int(os.getenv('DATABASE_CONNECTION_POOL_TIMEOUT', '30'))
    DATABASE_CONNECTION_POOL_RECYCLE = int(os.getenv('DATABASE_CONNECTION_POOL_RECYCLE', '3600'))

    # Redis Settings
    REDIS_URL = os.getenv('REDIS_URL', None)
    # REDIS_URL is optional - if not provided, in-memory cache will be used

    # Token Management Settings - No defaults for security
    TOKEN_CLIENT_ID = os.getenv('TOKEN_CLIENT_ID')
    TOKEN_CLIENT_SECRET = os.getenv('TOKEN_CLIENT_SECRET')
    TOKEN_URL = os.getenv(
        'TOKEN_URL',
        'https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token')
    TOKEN_SCOPE = os.getenv('TOKEN_SCOPE', 'openid profile')

    # Validate required secrets - allow missing for testing
    # if not TOKEN_CLIENT_ID and os.getenv('ALLOW_MISSING_TOKENS', '').lower() != 'true':
    #     raise ValueError("TOKEN_CLIENT_ID environment variable is required")
    # if not TOKEN_CLIENT_SECRET and os.getenv('ALLOW_MISSING_TOKENS', '').lower() != 'true':
    #     raise ValueError("TOKEN_CLIENT_SECRET environment variable is required")

    # GitHub MCP Server Settings
    MCP_SERVER_COMMAND = os.getenv(
        'MCP_SERVER_COMMAND',
        'docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN '
        'ghcr.io/github/github-mcp-server')
    GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
    MCP_SERVER_TOOLSETS = os.getenv('MCP_SERVER_TOOLSETS', 'all')
    MCP_SERVER_HOST = os.getenv('MCP_SERVER_HOST', '')

    # NGC (NVIDIA GPU Cloud) Settings
    NGC_API_KEY = os.getenv('NGC_API_KEY')
    NGC_CLI_PATH = os.getenv('NGC_CLI_PATH', 'ngc')
    NGC_REGISTRY_URL = os.getenv('NGC_REGISTRY_URL', 'nvcr.io')
    NGC_ORG = os.getenv('NGC_ORG', 'nvidia')

    # GPU Settings
    NVIDIA_VISIBLE_DEVICES = os.getenv('NVIDIA_VISIBLE_DEVICES', 'all')
    NVIDIA_DRIVER_CAPABILITIES = os.getenv('NVIDIA_DRIVER_CAPABILITIES', 'compute,utility,video')
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))

    # Multi-GPU Support Configuration
    MULTI_GPU_ENABLED = os.getenv('MULTI_GPU_ENABLED', 'false').lower() == 'true'
    GPU_COUNT = int(os.getenv('GPU_COUNT', '1'))
    GPU_STRATEGY = os.getenv('GPU_STRATEGY', 'mirrored')  # mirrored, parameter_server, central_storage
    GPU_MEMORY_GROWTH = os.getenv('GPU_MEMORY_GROWTH', 'true').lower() == 'true'
    GPU_PER_PROCESS_MEMORY_FRACTION = float(os.getenv('GPU_PER_PROCESS_MEMORY_FRACTION', '0.8'))
    GPU_ALLOW_GROWTH = os.getenv('GPU_ALLOW_GROWTH', 'true').lower() == 'true'

    # TensorFlow/Keras GPU Configuration
    TF_GPU_MEMORY_LIMIT_MB = int(os.getenv('TF_GPU_MEMORY_LIMIT_MB', '4096'))
    TF_FORCE_GPU_ALLOW_GROWTH = os.getenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true').lower() == 'true'
    TF_VISIBLE_DEVICES = os.getenv('TF_VISIBLE_DEVICES', '0')

    # PyTorch GPU Configuration
    TORCH_GPU_COUNT = int(os.getenv('TORCH_GPU_COUNT', '1'))
    TORCH_CUDA_VISIBLE_DEVICES = os.getenv('TORCH_CUDA_VISIBLE_DEVICES', '0')
    TORCH_DISTRIBUTED_BACKEND = os.getenv('TORCH_DISTRIBUTED_BACKEND', 'nccl')

    # Security Settings - No hardcoded defaults for production security
    SECRET_KEY = os.getenv('SECRET_KEY')
    # Allow missing for testing - uncomment the raise for production
    # if not SECRET_KEY:
    #     raise ValueError("SECRET_KEY environment variable is required for session security")
    if not SECRET_KEY:
        SECRET_KEY = 'dummy_secret_key_for_testing'  # Default for testing

    # JWT Settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)  # Use same key if not specified
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7'))

    # Rate Limiting Settings
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '100 per minute')
    RATE_LIMIT_STORAGE_URL = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')
    RATE_LIMIT_STRATEGY = os.getenv('RATE_LIMIT_STRATEGY', 'fixed-window')  # fixed-window, moving-window, leaky-bucket
    RATE_LIMIT_HEADERS_ENABLED = os.getenv('RATE_LIMIT_HEADERS_ENABLED', 'true').lower() == 'true'
    # Per-endpoint rate limits
    RATE_LIMIT_AUTH = os.getenv('RATE_LIMIT_AUTH', '5 per minute')
    RATE_LIMIT_API = os.getenv('RATE_LIMIT_API', '100 per minute')
    RATE_LIMIT_PAYMENTS = os.getenv('RATE_LIMIT_PAYMENTS', '50 per minute')
    RATE_LIMIT_TELEMETRY = os.getenv('RATE_LIMIT_TELEMETRY', '200 per minute')
    RATE_LIMIT_ADMIN = os.getenv('RATE_LIMIT_ADMIN', '20 per minute')

    # Caching Settings
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')  # simple, redis, memcached
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '300'))
    CACHE_THRESHOLD = int(os.getenv('CACHE_THRESHOLD', '500'))
    CACHE_REDIS_URL = os.getenv('CACHE_REDIS_URL', os.getenv('REDIS_URL'))
    CACHE_KEY_PREFIX = os.getenv('CACHE_KEY_PREFIX', 'jpmorgan_api_')

    # CORS Settings - Restrict origins for security (no localhost in production)
    ALLOWED_ORIGINS_STR = os.getenv('ALLOWED_ORIGINS')
    if not ALLOWED_ORIGINS_STR:
        # Default to production domains only - no localhost for security
        ALLOWED_ORIGINS = ['https://app.jpmorgan.com', 'https://dashboard.jpmorgan.com', 'https://api.jpmorgan.com']
    else:
        ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(',') if origin.strip()]
        # Security check: ensure no localhost origins in production
        if os.getenv('FLASK_ENV') == 'production':
            ALLOWED_ORIGINS = [origin for origin in ALLOWED_ORIGINS if not origin.startswith('http://localhost') and not origin.startswith('http://127.0.0.1')]

    # Audit Logging Settings
    AUDIT_LOG_ENABLED = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'
    # Keep logs for 90 days
    AUDIT_LOG_RETENTION_DAYS = int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))
    AUDIT_LOG_MAX_SIZE = int(os.getenv('AUDIT_LOG_MAX_SIZE', '10000000'))  # 10MB max log size
    AUDIT_ALERT_ENABLED = os.getenv('AUDIT_ALERT_ENABLED', 'true').lower() == 'true'
    # Alert after 5 failed logins
    AUDIT_FAILED_LOGIN_THRESHOLD = int(
        os.getenv('AUDIT_FAILED_LOGIN_THRESHOLD', '5'))
    # Alert after 100 requests/min
    AUDIT_RATE_LIMIT_THRESHOLD = int(
        os.getenv('AUDIT_RATE_LIMIT_THRESHOLD', '100'))
    # Alert after 10 failed logins
    AUDIT_BRUTE_FORCE_THRESHOLD = int(
        os.getenv('AUDIT_BRUTE_FORCE_THRESHOLD', '10'))
    # Alert if IP accesses 5+ accounts
    AUDIT_SUSPICIOUS_IP_THRESHOLD = int(
        os.getenv('AUDIT_SUSPICIOUS_IP_THRESHOLD', '5'))
    # log, email, slack
    AUDIT_ALERT_NOTIFICATION_METHOD = os.getenv(
        'AUDIT_ALERT_NOTIFICATION_METHOD', 'log')
    # Auto-cleanup old logs
    AUDIT_CLEANUP_ENABLED = os.getenv(
        'AUDIT_CLEANUP_ENABLED', 'true').lower() == 'true'
    # Enable tamper-proof hash chain
    AUDIT_HASH_CHAIN_ENABLED = os.getenv(
        'AUDIT_HASH_CHAIN_ENABLED', 'true').lower() == 'true'

    # Auth0 Authentication Settings
    AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN')
    AUTH0_CLIENT_ID = os.getenv('AUTH0_CLIENT_ID')
    AUTH0_CLIENT_SECRET = os.getenv('AUTH0_CLIENT_SECRET')
    AUTH0_AUDIENCE = os.getenv('AUTH0_AUDIENCE')
    AUTH0_ALGORITHMS = os.getenv('AUTH0_ALGORITHMS', 'RS256').split(',')
    AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/" if AUTH0_DOMAIN else None
    AUTH0_JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json" if AUTH0_DOMAIN else None

    # LangSmith Settings for AI tracing and monitoring
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'true').lower() == 'true'
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT', 'jpmorgan-financial-apis')
    LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')

    # OpenAI Settings for LangChain
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))

    # Blackbox AI Settings (OpenAI-compatible)
    BLACKBOX_API_KEY = os.getenv('BLACKBOX_API_KEY')
    BLACKBOX_BASE_URL = os.getenv('BLACKBOX_BASE_URL', 'https://cloud.blackbox.ai/')
    BLACKBOX_MODEL = os.getenv('BLACKBOX_MODEL', 'gpt-3.5-turbo')
    BLACKBOX_TEMPERATURE = float(os.getenv('BLACKBOX_TEMPERATURE', '0.1'))

    # Stripe Payment Processing Settings
    STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
    STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
    STRIPE_API_VERSION = os.getenv('STRIPE_API_VERSION', '2023-10-16')
    STRIPE_CURRENCY = os.getenv('STRIPE_CURRENCY', 'usd')

    @classmethod
    def get_database_url(cls) -> str:
        """Generate database URL based on configuration"""
        if cls.DATABASE_TYPE == 'postgresql':
            return (f"postgresql://{cls.DATABASE_USER}:{cls.DATABASE_PASSWORD}"
                    f"@{cls.DATABASE_HOST}:{cls.DATABASE_PORT}/"
                    f"{cls.DATABASE_NAME}")
        else:
            return cls.DATABASE_URL

    @classmethod
    def get_jpmorgan_endpoint_url(cls, service: str, use_mtls: bool = False) -> str:
        """Get JPMorgan endpoint URL based on environment and service

        Args:
            service: 'merchant', 'openbanking', or 'apigateway'
            use_mtls: Whether to use mTLS endpoint (for merchant service)

        Returns:
            The appropriate endpoint URL for the current environment
        """
        environment = cls.JPMORGAN_ENVIRONMENT.lower()

        if service == 'merchant':
            if environment in ['dev', 'staging']:
                return (cls.JPMORGAN_MERCHANT_MTLS_UAT_URL if use_mtls
                        else cls.JPMORGAN_MERCHANT_UAT_URL)
            else:  # prod (default)
                return (cls.JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL if use_mtls
                        else cls.JPMORGAN_MERCHANT_PRODUCTION_URL)
        elif service == 'openbanking':
            if environment in ['dev', 'staging']:
                return cls.JPMORGAN_OPENBANKING_UAT_URL
            else:  # prod (default)
                return cls.JPMORGAN_OPENBANKING_PRODUCTION_URL
        elif service == 'apigateway':
            if environment == 'qaf':
                return cls.JPMORGAN_APIGATEWAY_QAF_URL
            else:  # prod (default)
                return cls.JPMORGAN_APIGATEWAY_PRODUCTION_URL
        else:
            raise ValueError(f"Unknown service: {service}. Must be 'merchant', 'openbanking', or 'apigateway'")

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        return {
            'api_base_url': cls.API_BASE_URL,
            'api_version': cls.API_VERSION,
            'log_level': cls.LOG_LEVEL,
            'log_file': cls.LOG_FILE,
            'telemetry_enabled': cls.TELEMETRY_ENABLED,
            'telemetry_batch_size': cls.TELEMETRY_BATCH_SIZE,
            'database_url': cls.get_database_url(),
            'database_type': cls.DATABASE_TYPE,
            'database_host': cls.DATABASE_HOST,
            'database_port': cls.DATABASE_PORT,
            'database_name': cls.DATABASE_NAME,
            'database_connection_pool_size': cls.DATABASE_CONNECTION_POOL_SIZE,
            'database_connection_pool_max_overflow': cls.DATABASE_CONNECTION_POOL_MAX_OVERFLOW,
            'database_connection_pool_timeout': cls.DATABASE_CONNECTION_POOL_TIMEOUT,
            'database_connection_pool_recycle': cls.DATABASE_CONNECTION_POOL_RECYCLE,
            'token_client_id': cls.TOKEN_CLIENT_ID,
            'token_client_secret': cls.TOKEN_CLIENT_SECRET,
            'token_url': cls.TOKEN_URL,
            'token_scope': cls.TOKEN_SCOPE,
            'mcp_server_command': cls.MCP_SERVER_COMMAND,
            'github_personal_access_token': cls.GITHUB_PERSONAL_ACCESS_TOKEN,
            'mcp_server_toolsets': cls.MCP_SERVER_TOOLSETS,
            'mcp_server_host': cls.MCP_SERVER_HOST,
            'jpmorgan_environment': cls.JPMORGAN_ENVIRONMENT,
            'jpmorgan_merchant_production_url': cls.JPMORGAN_MERCHANT_PRODUCTION_URL,
            'jpmorgan_merchant_mtls_production_url': cls.JPMORGAN_MERCHANT_MTLS_PRODUCTION_URL,
            'jpmorgan_merchant_uat_url': cls.JPMORGAN_MERCHANT_UAT_URL,
            'jpmorgan_merchant_mtls_uat_url': cls.JPMORGAN_MERCHANT_MTLS_UAT_URL,
            'jpmorgan_openbanking_production_url': cls.JPMORGAN_OPENBANKING_PRODUCTION_URL,
            'jpmorgan_openbanking_uat_url': cls.JPMORGAN_OPENBANKING_UAT_URL,
            'jpmorgan_apigateway_production_url': cls.JPMORGAN_APIGATEWAY_PRODUCTION_URL,
            'jpmorgan_apigateway_qaf_url': cls.JPMORGAN_APIGATEWAY_QAF_URL,
            'audit_log_enabled': cls.AUDIT_LOG_ENABLED,
            'audit_log_retention_days': cls.AUDIT_LOG_RETENTION_DAYS,
            'audit_alert_enabled': cls.AUDIT_ALERT_ENABLED,
            'audit_failed_login_threshold': cls.AUDIT_FAILED_LOGIN_THRESHOLD,
            'audit_rate_limit_threshold': cls.AUDIT_RATE_LIMIT_THRESHOLD,
            'auth0_domain': cls.AUTH0_DOMAIN,
            'auth0_client_id': cls.AUTH0_CLIENT_ID,
            'auth0_client_secret': cls.AUTH0_CLIENT_SECRET,
            'auth0_audience': cls.AUTH0_AUDIENCE,
            'auth0_algorithms': cls.AUTH0_ALGORITHMS,
            'auth0_issuer': cls.AUTH0_ISSUER,
            'auth0_jwks_url': cls.AUTH0_JWKS_URL,
            'stripe_publishable_key': cls.STRIPE_PUBLISHABLE_KEY,
            'stripe_secret_key': cls.STRIPE_SECRET_KEY,
            'stripe_webhook_secret': cls.STRIPE_WEBHOOK_SECRET,
            'stripe_api_version': cls.STRIPE_API_VERSION,
            'stripe_currency': cls.STRIPE_CURRENCY
        }


# Global configuration instance
config = Config()
