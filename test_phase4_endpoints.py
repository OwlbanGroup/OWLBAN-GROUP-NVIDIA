#!/usr/bin/env python3
"""
Test script for Phase 4 AI Integration enhancements.
Tests syntax, imports, and route registration for new AI, ML, and Data financial endpoints.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ai_blueprint_import():
    """Test that the AI blueprint can be imported without errors."""
    try:
        from blueprints.ai import ai_bp
        print("✓ AI blueprint imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import AI blueprint: {e}")
        return False

def test_data_blueprint_import():
    """Test that the Data blueprint can be imported without errors."""
    try:
        from blueprints.data import data_bp
        print("✓ Data blueprint imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import Data blueprint: {e}")
        return False

def test_ml_blueprint_import():
    """Test that the ML blueprint can be imported without errors."""
    try:
        from blueprints.ml import ml_bp
        print("✓ ML blueprint imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import ML blueprint: {e}")
        return False

def test_ai_functions_exist():
    """Test that AI blueprint functions are defined."""
    try:
        import blueprints.ai as ai_module

        required_functions = [
            'analyze_financial_context',
            'verify_identity',
            'process_agentic_commerce'
        ]

        missing_functions = []
        for func_name in required_functions:
            if not hasattr(ai_module, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            print(f"✗ Missing AI functions: {missing_functions}")
            return False
        else:
            print("✓ All AI functions are defined")
            return True

    except Exception as e:
        print(f"✗ Failed to check AI functions: {e}")
        return False

def test_data_financial_functions_exist():
    """Test that Data blueprint financial functions are defined."""
    try:
        import blueprints.data as data_module

        required_functions = [
            'create_financial_transaction',
            'get_financial_transactions',
            'get_financial_accounts',
            'get_user_financial_data'
        ]

        missing_functions = []
        for func_name in required_functions:
            if not hasattr(data_module, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            print(f"✗ Missing Data financial functions: {missing_functions}")
            return False
        else:
            print("✓ All Data financial functions are defined")
            return True

    except Exception as e:
        print(f"✗ Failed to check Data financial functions: {e}")
        return False

def test_ml_financial_functions_exist():
    """Test that ML blueprint financial functions are defined."""
    try:
        import blueprints.ml as ml_module

        required_functions = [
            'analyze_financial_context',
            'analyze_transaction_patterns',
            'get_spending_insights',
            'analyze_cash_flow'
        ]

        missing_functions = []
        for func_name in required_functions:
            if not hasattr(ml_module, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            print(f"✗ Missing ML financial functions: {missing_functions}")
            return False
        else:
            print("✓ All ML financial functions are defined")
            return True

    except Exception as e:
        print(f"✗ Failed to check ML financial functions: {e}")
        return False

def test_blueprint_route_prefixes():
    """Test that blueprints have correct route prefixes."""
    try:
        from blueprints.ai import ai_bp
        from blueprints.data import data_bp
        from blueprints.ml import ml_bp

        # Check if blueprints have routes (basic check)
        ai_routes = [rule.rule for rule in ai_bp.url_map.iter_rules()]
        data_routes = [rule.rule for rule in data_bp.url_map.iter_rules()]
        ml_routes = [rule.rule for rule in ml_bp.url_map.iter_rules()]

        print(f"✓ AI routes found: {len(ai_routes)}")
        print(f"✓ Data routes found: {len(data_routes)}")
        print(f"✓ ML routes found: {len(ml_routes)}")

        # Check for financial-specific routes
        financial_routes = [r for r in data_routes if 'financial' in r]
        ml_financial_routes = [r for r in ml_routes if any(x in r for x in ['financial-context', 'transaction-patterns', 'spending-insights', 'cash-flow-analysis'])]

        if len(financial_routes) >= 4:
            print("✓ Data financial routes detected")
        else:
            print(f"✗ Expected 4+ financial routes in data, found {len(financial_routes)}")
            return False

        if len(ml_financial_routes) >= 4:
            print("✓ ML financial routes detected")
        else:
            print(f"✗ Expected 4+ financial routes in ML, found {len(ml_financial_routes)}")
            return False

        return True

    except Exception as e:
        print(f"✗ Failed to check route prefixes: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Phase 4 AI Integration tests...\n")

    tests = [
        test_ai_blueprint_import,
        test_data_blueprint_import,
        test_ml_blueprint_import,
        test_ai_functions_exist,
        test_data_financial_functions_exist,
        test_ml_financial_functions_exist,
        test_blueprint_route_prefixes
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Phase 4 implementation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
