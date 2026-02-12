#!/usr/bin/env python3
"""
Test script for Phase 3 PFM Budgeting and Goals features.
Tests syntax, imports, and function definitions for budget and goal management endpoints.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pfm_blueprint_import():
    """Test that the PFM blueprint can be imported without errors."""
    try:
        from blueprints.pfm import pfm_bp
        print("✓ PFM blueprint imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import PFM blueprint: {e}")
        return False

def test_budget_functions_exist():
    """Test that budget management functions are defined."""
    try:
        import blueprints.pfm as pfm_module

        required_functions = [
            'create_budget',
            'get_budgets',
            'update_budget',
            'get_budget_progress'
        ]

        missing_functions = []
        for func_name in required_functions:
            if not hasattr(pfm_module, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            print(f"✗ Missing budget functions: {missing_functions}")
            return False
        else:
            print("✓ All budget management functions are defined")
            return True

    except Exception as e:
        print(f"✗ Failed to check budget functions: {e}")
        return False

def test_goal_functions_exist():
    """Test that financial goal functions are defined."""
    try:
        import blueprints.pfm as pfm_module

        required_functions = [
            'create_financial_goal',
            'get_financial_goals',
            'contribute_to_goal'
        ]

        missing_functions = []
        for func_name in required_functions:
            if not hasattr(pfm_module, func_name):
                missing_functions.append(func_name)

        if missing_functions:
            print(f"✗ Missing goal functions: {missing_functions}")
            return False
        else:
            print("✓ All financial goal functions are defined")
            return True

    except Exception as e:
        print(f"✗ Failed to check goal functions: {e}")
        return False

def test_categorization_function():
    """Test that transaction categorization is working."""
    try:
        from blueprints.pfm import categorize_transaction

        test_descriptions = [
            'Grocery Store Purchase',
            'Starbucks Coffee',
            'Gas Station Fill-up',
            'Amazon Purchase',
            'Salary Deposit'
        ]

        for desc in test_descriptions:
            category = categorize_transaction(desc)
            if not category or category == 'other':
                print(f"✗ Categorization failed for '{desc}' -> '{category}'")
                return False

        print("✓ Transaction categorization is working correctly")
        return True

    except Exception as e:
        print(f"✗ Failed to test categorization: {e}")
        return False

def test_budget_calculation_function():
    """Test that budget spent calculation function exists."""
    try:
        import blueprints.pfm as pfm_module

        if hasattr(pfm_module, 'calculate_budget_spent'):
            print("✓ Budget spent calculation function is defined")
            return True
        else:
            print("✗ calculate_budget_spent function not found")
            return False

    except Exception as e:
        print(f"✗ Failed to check budget calculation: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Phase 3 PFM Budgeting and Goals tests...\n")

    tests = [
        test_pfm_blueprint_import,
        test_budget_functions_exist,
        test_goal_functions_exist,
        test_categorization_function,
        test_budget_calculation_function
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Phase 3 implementation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
