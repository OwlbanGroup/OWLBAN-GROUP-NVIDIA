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
    JPMORGAN_ENVIRONMENT = os.getenv('JPMORGAN_ENVIRONMENT', 'production')  # production, uat, qaf
    
    # JPMorgan OpenBanking API Endpoints
    JPMORGAN_OPENBANKING_PRODUCTION_URL = os.getenv(
        'JPMORGAN_OPENBANKING_PRODUCTION_URL',
        'https://openbanking.jpmorgan.com/accessapi'
    )
    JPMORGAN_OPENBANKING_UAT_URL = os.getenv(
        'JPMORGAN_OPENBANKING_UAT_URL',
        'https://openbankinguat.jpmorgan.com/accessapi'
    )
    
    # JPMorgan API Gateway Endpoints
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

    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///telemetry.db')
    DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'sqlite')  # sqlite, postgresql
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
    DATABASE_PORT = int(os.getenv('DATABASE_PORT', '5432'))
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'jpmorgan_financial_apis')
    DATABASE_USER = os.getenv('DATABASE_USER', '')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', '')
    DATABASE_SSL_MODE = os.getenv('DATABASE_SSL_MODE', 'require')
    DATABASE_CONNECTION_POOL_SIZE = int(os.getenv('DATABASE_CONNECTION_POOL_SIZE', '10'))
    DATABASE_CONNECTION_POOL_MAX_OVERFLOW = int(os.getenv('DATABASE_CONNECTION_POOL_MAX_OVERFLOW', '20'))
    DATABASE_CONNECTION_POOL_TIMEOUT = int(os.getenv('DATABASE_CONNECTION_POOL_TIMEOUT', '30'))
    DATABASE_CONNECTION_POOL_RECYCLE = int(os.getenv('DATABASE_CONNECTION_POOL_RECYCLE', '3600'))

    # Redis Settings
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    # Token Management Settings - No defaults for security
    TOKEN_CLIENT_ID = os.getenv('TOKEN_CLIENT_ID')
    TOKEN_CLIENT_SECRET = os.getenv('TOKEN_CLIENT_SECRET')
    TOKEN_URL = os.getenv('TOKEN_URL', 'https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token')
    TOKEN_SCOPE = os.getenv('TOKEN_SCOPE', 'openid profile')

    # Validate required secrets - allow missing for testing
    # if not TOKEN_CLIENT_ID and os.getenv('ALLOW_MISSING_TOKENS', '').lower() != 'true':
    #     raise ValueError("TOKEN_CLIENT_ID environment variable is required")
    # if not TOKEN_CLIENT_SECRET and os.getenv('ALLOW_MISSING_TOKENS', '').lower() != 'true':
    #     raise ValueError("TOKEN_CLIENT_SECRET environment variable is required")

    # GitHub MCP Server Settings
    MCP_SERVER_COMMAND = os.getenv('MCP_SERVER_COMMAND', 'docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server')
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

    # Security Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_secret')
    # if not SECRET_KEY and os.getenv('ALLOW_MISSING_TOKENS', '').lower() != 'true':
    #     raise ValueError("SECRET_KEY environment variable is required for session security")
    
    # Audit Logging Settings
    AUDIT_LOG_ENABLED = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'
    AUDIT_LOG_RETENTION_DAYS = int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))  # Keep logs for 90 days
    AUDIT_LOG_MAX_SIZE = int(os.getenv('AUDIT_LOG_MAX_SIZE', '10000000'))  # 10MB max log size
    AUDIT_ALERT_ENABLED = os.getenv('AUDIT_ALERT_ENABLED', 'true').lower() == 'true'
    AUDIT_FAILED_LOGIN_THRESHOLD = int(os.getenv('AUDIT_FAILED_LOGIN_THRESHOLD', '5'))  # Alert after 5 failed logins
    AUDIT_RATE_LIMIT_THRESHOLD = int(os.getenv('AUDIT_RATE_LIMIT_THRESHOLD', '100'))  # Alert after 100 requests/min
    AUDIT_BRUTE_FORCE_THRESHOLD = int(os.getenv('AUDIT_BRUTE_FORCE_THRESHOLD', '10'))  # Alert after 10 failed logins
    AUDIT_SUSPICIOUS_IP_THRESHOLD = int(os.getenv('AUDIT_SUSPICIOUS_IP_THRESHOLD', '5'))  # Alert if IP accesses 5+ accounts
    AUDIT_ALERT_NOTIFICATION_METHOD = os.getenv('AUDIT_ALERT_NOTIFICATION_METHOD', 'log')  # log, email, slack
    AUDIT_CLEANUP_ENABLED = os.getenv('AUDIT_CLEANUP_ENABLED', 'true').lower() == 'true'  # Auto-cleanup old logs
    AUDIT_HASH_CHAIN_ENABLED = os.getenv('AUDIT_HASH_CHAIN_ENABLED', 'true').lower() == 'true'  # Enable tamper-proof hash chain

    @classmethod
    def get_database_url(cls) -> str:
        """Generate database URL based on configuration"""
        if cls.DATABASE_TYPE == 'postgresql':
            return f"postgresql://{cls.DATABASE_USER}:{cls.DATABASE_PASSWORD}@{cls.DATABASE_HOST}:{cls.DATABASE_PORT}/{cls.DATABASE_NAME}"
        else:
            return cls.DATABASE_URL

    @classmethod
    def get_jpmorgan_endpoint_url(cls, service: str) -> str:
        """Get JPMorgan endpoint URL based on environment and service
        
        Args:
            service: 'openbanking' or 'apigateway'
            
        Returns:
            The appropriate endpoint URL for the current environment
        """
        environment = cls.JPMORGAN_ENVIRONMENT.lower()
        
        if service == 'openbanking':
            if environment == 'uat':
                return cls.JPMORGAN_OPENBANKING_UAT_URL
            else:  # production (default)
                return cls.JPMORGAN_OPENBANKING_PRODUCTION_URL
        elif service == 'apigateway':
            if environment == 'qaf':
                return cls.JPMORGAN_APIGATEWAY_QAF_URL
            else:  # production (default)
                return cls.JPMORGAN_APIGATEWAY_PRODUCTION_URL
        else:
            raise ValueError(f"Unknown service: {service}. Must be 'openbanking' or 'apigateway'")
    
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
            'jpmorgan_openbanking_production_url': cls.JPMORGAN_OPENBANKING_PRODUCTION_URL,
            'jpmorgan_openbanking_uat_url': cls.JPMORGAN_OPENBANKING_UAT_URL,
            'jpmorgan_apigateway_production_url': cls.JPMORGAN_APIGATEWAY_PRODUCTION_URL,
            'jpmorgan_apigateway_qaf_url': cls.JPMORGAN_APIGATEWAY_QAF_URL,
            'audit_log_enabled': cls.AUDIT_LOG_ENABLED,
            'audit_log_retention_days': cls.AUDIT_LOG_RETENTION_DAYS,
            'audit_alert_enabled': cls.AUDIT_ALERT_ENABLED,
            'audit_failed_login_threshold': cls.AUDIT_FAILED_LOGIN_THRESHOLD,
            'audit_rate_limit_threshold': cls.AUDIT_RATE_LIMIT_THRESHOLD
        }


# Global configuration instance
config = Config()
