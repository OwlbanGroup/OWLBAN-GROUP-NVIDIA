# Blueprints package - SIMPLIFIED for pytest 100% coverage

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("✅ blueprints/__init__.py: SIMPLIFIED - Direct imports")

# Direct imports - NO fallbacks (tests need real blueprints)
from .pfm import pfm_bp
print("✅ pfm_bp imported directly")

# Other blueprints (real or None)
try:
    from .user import user_bp
except ImportError:
    user_bp = None
try:
    from .payments import payments_bp
except ImportError:
    payments_bp = None
try:
    from .payroll import payroll_bp
except ImportError:
    payroll_bp = None

# Financial services - PFM CRITICAL
try:
    from .payments import payments_bp as payments_bp
except ImportError:
    payments_bp = None

# Direct pfm_bp import already done above - NO FALLBACK

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
    'ml_bp', 'ai_bp', 'data_bp', 'internal_ops_bp', 'banking_bp'
]

print(f"✅ Blueprint package ready. pfm_bp available: {pfm_bp is not None}")
sys.modules['blueprints.pfm'] = pfm



