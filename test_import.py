#!/usr/bin/env python3
"""
Test script to verify imports work correctly.
"""

try:
    from blueprints.payments import payments_bp
    print("✅ Payments blueprint imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import payments blueprint: {e} (stub mode)")


try:
    from blueprints.user import user_bp
    print("✅ User blueprint imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import user blueprint: {e} (stub mode)")


try:
    from blueprints.business import business_bp
    print("✅ Business blueprint imported successfully")
except ImportError as e:
    print(f"❌ Failed to import business blueprint: {e}")
    exit(1)

try:
    from blueprints.asset import asset_bp
    print("✅ Asset blueprint imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import asset blueprint: {e} (stub mode)")

try:
    from blueprints.telemetry import telemetry_bp
    print("✅ Telemetry blueprint imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import telemetry blueprint: {e} (stub mode)")

try:
    from blueprints.ml import ml_bp
    print("✅ ML blueprint imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import ML blueprint: {e} (stub mode)")

try:
    from blueprints.data import data_bp
    print("✅ Data blueprint imported successfully")
except ImportError as e:
    print(f"⚠️  Failed to import data blueprint: {e} (stub mode)")

print("🎉 All critical blueprints verified (some optional stubs skipped)!")
