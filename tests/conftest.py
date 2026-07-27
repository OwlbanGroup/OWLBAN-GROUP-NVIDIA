"""Pytest configuration for JPMorgan Financial APIs
Fixes blueprints.pfm import issue during test collection"""

import os
import sys
import pytest
from pathlib import Path

# ✅ CRITICAL: FIX pytest COLLECTION - Module level sys.path (runs BEFORE fixtures)
# Fix PYTHONPATH for blueprint imports during pytest COLLECTION phase
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
BLUEPRINTS_PATH = PROJECT_ROOT / "blueprints"
SRC_PATH = PROJECT_ROOT / "src"

# Add paths at INDEX 0 to override stdlib
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BLUEPRINTS_PATH) not in sys.path:
    sys.path.insert(0, str(BLUEPRINTS_PATH)) 
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

print(f"✅ conftest.py PATHS FIXED: {PROJECT_ROOT=}, {BLUEPRINTS_PATH=}, {SRC_PATH=}")

# FIXED pytest COLLECTION - Direct project-relative paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

print("✅ conftest.py PATHS: ", str(PROJECT_ROOT))

# PRE-IMPORT with project prefix for pytest
try:
    from blueprints.pfm import pfm_bp
    PFM_AVAILABLE = True
    print("✅ conftest.py: PFM blueprint pre-imported (", pfm_bp.name, ")")
except ImportError as e:
    print(f"❌ conftest.py PFM pre-import failed: {e}")
    PFM_AVAILABLE = False

@pytest.fixture(scope="session")
def project_root():
    """Provide project root path to tests"""
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def pfm_test_app():
    """✅ ISOLATED PFM test app for 100% coverage"""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Mock globals for tests (import from same module path used by tests)
    from blueprints.pfm import _mock_accounts, _mock_budgets  # noqa
    app._mock_accounts = _mock_accounts
    app._mock_budgets = _mock_budgets
    
    # Register ONLY PFM for focused testing
    try:
        from blueprints.pfm import pfm_bp
        app.register_blueprint(pfm_bp, url_prefix='/pfm')
        print("✅ PFM test_app ready - 100% coverage target")
    except ImportError as e:
        print(f"❌ PFM blueprint unavailable: {e}")
    
    return app

@pytest.fixture(scope="session")
def test_app(project_root):
    """Create test Flask app with all blueprints"""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    # Register all available blueprints
    try:
        from blueprints import (
            banking_bp, user_bp, pfm_bp, payments_bp, payroll_bp, 
            loans_bp, credit_bp, asset_bp, business_bp
        )
        blueprints = {
            banking_bp: '/banking',
            user_bp: '/user', 
            pfm_bp: '/pfm',
            payments_bp: '/payments',
            payroll_bp: '/payroll',
            loans_bp: '/loans',
            credit_bp: '/credit',
            asset_bp: '/asset',
            business_bp: '/business'
        }
        
        for bp, prefix in blueprints.items():
            if bp:
                app.register_blueprint(bp, url_prefix=prefix)
        print("✅ All blueprints registered in test app")
    except ImportError as e:
        print(f"⚠️  Some blueprints unavailable: {e}")
    
    return app

@pytest.fixture
def pfm_client(pfm_test_app):
    """✅ PFM-specific test client"""
    with pfm_test_app.test_client() as client:
        yield client

@pytest.fixture
def client(test_app):
    """Full test client fixture"""
    with test_app.test_client() as client:
        yield client

@pytest.fixture
def coverage_client(test_app):
    """Compatibility fixture used by coverage-focused test modules."""
    with test_app.test_client() as client:
        client.environ_base['HTTP_AUTHORIZATION'] = 'Bearer test_token'
        yield client

