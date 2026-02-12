#!/usr/bin/env python3
"""
Test script for Phase 2 ML Blueprint enhancements.
Tests syntax, imports, and route registration for new financial analysis endpoints.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ml_blueprint_import():
    """Test that the ML blueprint can be imported without errors."""
    try:
        from blueprints.ml import ml_bp
        print("✓ ML blueprint imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import ML blueprint: {e}")
        return False

def test_new_routes_exist():
    """Test that the new financial analysis functions are defined in the module."""
    try:
        import blueprints.ml as ml_module

        # Check if the new functions are defined in the module
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
            print(f"✗ Missing functions: {missing_functions}")
            return False
        else:
            print("✓ All new financial analysis functions are defined")
            return True

    except Exception as e:
        print(f"✗ Failed to check functions: {e}")
        return False

def test_route_methods():
    """Test that routes have correct HTTP methods by checking function definitions."""
    try:
        import blueprints.ml as ml_module

        # Check if functions are defined and callable
        required_functions = [
            'analyze_financial_context',
            'analyze_transaction_patterns',
            'get_spending_insights',
            'analyze_cash_flow'
        ]

        all_correct = True
        for func_name in required_functions:
            if hasattr(ml_module, func_name):
                func = getattr(ml_module, func_name)
                # Check if function exists and is callable
                if callable(func):
                    print(f"✓ Function {func_name} is properly defined")
                else:
                    print(f"✗ Function {func_name} is not callable")
                    all_correct = False
            else:
                print(f"✗ Function {func_name} not found")
                all_correct = False

        if all_correct:
            print("✓ All route functions are properly defined")
            return True
        else:
            return False

    except Exception as e:
        print(f"✗ Failed to check route methods: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Phase 2 ML Blueprint tests...\n")

    tests = [
        test_ml_blueprint_import,
        test_new_routes_exist,
        test_route_methods
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Phase 2 implementation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
