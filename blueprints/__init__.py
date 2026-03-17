# Blueprints package - convenience imports for all blueprints

# Core blueprints
from .user import bp as user_bp
from .asset import bp as asset_bp
from .business import bp as business_bp
from .telemetry import bp as telemetry_bp

# Financial services
from .payments import bp as payments_bp
from .pfm import bp as pfm_bp
from .payroll import bp as payroll_bp
from .loans import bp as loans_bp
from .credit import bp as credit_bp
from .transfers import bp as transfers_bp
from .statements import bp as statements_bp

# AI/ML/Data
from .ml import bp as ml_bp
from .ai import bp as ai_bp
from .data import bp as data_bp

# Internal ops
from .internal_ops import bp as internal_ops_bp

__all__ = [
    'user_bp', 'asset_bp', 'business_bp', 'telemetry_bp',
    'payments_bp', 'pfm_bp', 'payroll_bp', 'loans_bp', 
    'credit_bp', 'transfers_bp', 'statements_bp',
    'ml_bp', 'ai_bp', 'data_bp', 'internal_ops_bp'
]

