"""
Financial Data Provider Configuration
OWLBAN GROUP - Enterprise Financial Data Configuration Manager
"""

import os
import json
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class BloombergConfig:
    host: str
    port: int
    auth_token: str
    service_uri: str = "//blp/mktdata"
    timeout: int = 5000

@dataclass
class RefinitivConfig:
    app_key: str
    workspace_id: str
    environment: str = "prod"
    auth_url: str = "https://api.refinitiv.com"
    timeout: int = 5000

@dataclass
class MarketDataConfig:
    alpha_vantage_key: str
    iex_token: str
    polygon_key: str
    tiingo_token: str

class FinancialDataConfig:
    """Configuration manager for financial data providers"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            'FINANCIAL_CONFIG_PATH',
            'config/financial_providers.json'
        )
        self._load_config()
        
    def _load_config(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        self.bloomberg = BloombergConfig(**config['bloomberg'])
        self.refinitiv = RefinitivConfig(**config['refinitiv'])
        self.market_data = MarketDataConfig(**config['market_data'])
        
        # Additional provider configurations
        self.provider_configs = {
            'bloomberg': self.bloomberg,
            'refinitiv': self.refinitiv,
            'market_data': self.market_data
        }
        
    def get_provider_config(self, provider: str) -> Dict:
        """Get configuration for specific provider"""
        if provider not in self.provider_configs:
            raise ValueError(f"Unknown provider: {provider}")
        return self.provider_configs[provider]
    
    def update_provider_config(self, provider: str, **kwargs):
        """Update configuration for specific provider"""
        if provider not in self.provider_configs:
            raise ValueError(f"Unknown provider: {provider}")
            
        config = self.provider_configs[provider]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Invalid config parameter for {provider}: {key}")
                
        self._save_config()
        
    def _save_config(self):
        """Save current configuration to file"""
        config = {
            'bloomberg': self.bloomberg.__dict__,
            'refinitiv': self.refinitiv.__dict__,
            'market_data': self.market_data.__dict__
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    @property
    def providers(self) -> Dict:
        """Get all available provider configurations"""
        return self.provider_configs.copy()