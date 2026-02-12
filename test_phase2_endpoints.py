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
    """Test that the new financial analysis routes are registered."""
    try:
        from blueprints.ml import ml_bp

        # Get all registered routes
        routes = []
        for rule in ml_bp.url_map.iter_rules():
            routes.append(str(rule))

        # Check for new Phase 2 endpoints
        required_routes = [
            '/ml/financial-context',
            '/ml/transaction-patterns',
            '/ml/spending-insights',
            '/ml/cash-flow-analysis'
        ]

        missing_routes = []
        for route in required_routes:
            if not any(route in r for r in routes):
                missing_routes.append(route)

        if missing_routes:
            print(f"✗ Missing routes: {missing_routes}")
            return False
        else:
            print("✓ All new financial analysis routes are registered")
            print(f"  Total routes in ML blueprint: {len(routes)}")
            return True

    except Exception as e:
        print(f"✗ Failed to check routes: {e}")
        return False

def test_route_methods():
    """Test that routes have correct HTTP methods."""
    try:
        from blueprints.ml import ml_bp

        route_methods = {}
        for rule in ml_bp.url_map.iter_rules():
            route_methods[str(rule)] = list(rule.methods - {'HEAD', 'OPTIONS'})

        # Check specific routes
        checks = [
            ('/ml/financial-context', ['POST']),
            ('/ml/transaction-patterns', ['POST']),
            ('/ml/spending-insights', ['GET']),
            ('/ml/cash-flow-analysis', ['POST'])
        ]

        all_correct = True
        for route, expected_methods in checks:
            actual_methods = route_methods.get(route, [])
            if set(actual_methods) != set(expected_methods):
                print(f"✗ Route {route}: expected {expected_methods}, got {actual_methods}")
                all_correct = False

        if all_correct:
            print("✓ All routes have correct HTTP methods")
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
