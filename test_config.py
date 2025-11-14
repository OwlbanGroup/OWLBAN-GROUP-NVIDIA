"""
Test configuration settings for JPMorgan Financial APIs
"""
import os
from typing import Dict, Any

class TestConfig:
    """Test configuration class for the application"""

    # API Settings
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.jpmorgan.com')
    API_VERSION = os.getenv('API_VERSION', 'v1')

    # Logging Settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/telemetry.log')

    # Telemetry Settings
    TELEMETRY_ENABLED = os.getenv('TELEMETRY_ENABLED', 'true').lower() == 'true'
    TELEMETRY_BATCH_SIZE = int(os.getenv('TELEMETRY_BATCH_SIZE', '100'))

    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///test_telemetry.db')

    # Redis Settings
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    # Token Management Settings - Test defaults
    TOKEN_CLIENT_ID = os.getenv('TOKEN_CLIENT_ID', 'test_client_id')
    TOKEN_CLIENT_SECRET = os.getenv('TOKEN_CLIENT_SECRET', 'test_client_secret')
    TOKEN_URL = os.getenv('TOKEN_URL', 'https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token')
    TOKEN_SCOPE = os.getenv('TOKEN_SCOPE', 'openid profile')

    # GitHub MCP Server Settings
    MCP_SERVER_COMMAND = os.getenv('MCP_SERVER_COMMAND', 'docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server')
    GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN', 'test_token')
    MCP_SERVER_TOOLSETS = os.getenv('MCP_SERVER_TOOLSETS', 'all')
    MCP_SERVER_HOST = os.getenv('MCP_SERVER_HOST', '')

    # Security Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'test_secret_key_for_testing_only')

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
            'database_url': cls.DATABASE_URL,
            'token_client_id': cls.TOKEN_CLIENT_ID,
            'token_client_secret': cls.TOKEN_CLIENT_SECRET,
            'token_url': cls.TOKEN_URL,
            'token_scope': cls.TOKEN_SCOPE,
            'mcp_server_command': cls.MCP_SERVER_COMMAND,
            'github_personal_access_token': cls.GITHUB_PERSONAL_ACCESS_TOKEN,
            'mcp_server_toolsets': cls.MCP_SERVER_TOOLSETS,
            'mcp_server_host': cls.MCP_SERVER_HOST
        }


# Global test configuration instance
test_config = TestConfig()
