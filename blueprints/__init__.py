# Blueprints package - convenience imports for all blueprints (with error handling)

# Core blueprints
try:
    from .user import user_bp as user_bp
except ImportError:
    user_bp = None

try:
    from .asset import asset_bp as asset_bp
except ImportError:
    asset_bp = None

try:
    from .business import business_bp as business_bp
except ImportError:
    business_bp = None

try:
    from .telemetry import telemetry_bp as telemetry_bp
except ImportError:
    telemetry_bp = None

try:
    from .financial import financial_bp as financial_bp
except ImportError:
    financial_bp = None

# Financial services
try:
    from .payments import payments_bp as payments_bp
except ImportError:
    payments_bp = None

try:
    from .pfm import pfm_bp as pfm_bp
except ImportError:
    pfm_bp = None

try:
    from .payroll import payroll_bp as payroll_bp
except ImportError:
    payroll_bp = None

try:
    from .loans import loans_bp as loans_bp
except ImportError:
    loans_bp = None

try:
    from .credit import credit_bp as credit_bp
except ImportError:
    credit_bp = None

try:
    from .transfers import transfers_bp as transfers_bp
except ImportError:
    transfers_bp = None

try:
    from .statements import statements_bp as statements_bp
except ImportError:
    statements_bp = None

# AI/ML/Data
try:
    from .ml import ml_bp as ml_bp
except ImportError:
    ml_bp = None

try:
    from .ai import ai_bp as ai_bp
except ImportError:
    ai_bp = None

try:
    from .data import data_bp as data_bp
except ImportError:
    data_bp = None

# Internal ops
try:
    from .internal_ops import internal_ops_bp as internal_ops_bp
except ImportError:
    internal_ops_bp = None

try:
    from .banking import banking_bp as banking_bp
except ImportError:
    banking_bp = None

__all__ = [
    'user_bp', 'asset_bp', 'business_bp', 'telemetry_bp', 'financial_bp',
    'payments_bp', 'pfm_bp', 'payroll_bp', 'loans_bp', 
    'credit_bp', 'transfers_bp', 'statements_bp',
    'ml_bp', 'ai_bp', 'data_bp', 'internal_ops_bp'
]


