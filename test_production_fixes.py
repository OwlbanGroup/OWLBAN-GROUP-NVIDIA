#!/usr/bin/env python3
"""
Test script to verify all production_server.py diagnostic fixes
"""
import sys
import os

def test_syntax():
    """Test 1: Verify Python syntax is valid"""
    print("Test 1: Checking Python syntax...")
    try:
        import py_compile  # pylint: disable=import-outside-toplevel
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_path = os.path.join(script_dir, 'production_server.py')
        py_compile.compile(server_path, doraise=True)
        print("✅ PASS: Python syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ FAIL: Syntax error - {e}")
        return False

def test_imports():
    """Test 2: Verify all imports work"""
    print("\nTest 2: Checking imports...")
    try:
        # Test waitress import with type ignore
        from waitress import serve  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel,unused-import
        print("✅ PASS: waitress import successful")

        # Test flask_limiter imports at top level
        from flask_limiter import Limiter  # pylint: disable=import-outside-toplevel,unused-import
        from flask_limiter.util import get_remote_address  # pylint: disable=import-outside-toplevel,unused-import
        print("✅ PASS: flask_limiter imports successful")

        return True
    except ImportError as e:
        print(f"❌ FAIL: Import error - {e}")
        return False

def test_code_structure():
    """Test 3: Verify code structure fixes"""
    print("\nTest 3: Checking code structure...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, 'production_server.py')

    with open(server_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = []

    # Check 1: Type ignore comment for waitress
    if '# type: ignore[import-untyped]' in content:
        print("✅ PASS: Type ignore comment present for waitress")
        checks.append(True)
    else:
        print("❌ FAIL: Missing type ignore comment for waitress")
        checks.append(False)

    # Check 2: Pylint disable comment for app import
    if '# pylint: disable=import-error' in content:
        print("✅ PASS: Pylint disable comment present for app import")
        checks.append(True)
    else:
        print("❌ FAIL: Missing pylint disable comment for app import")
        checks.append(False)

    # Check 3: flask_limiter imports at top level
    lines = content.split('\n')
    import_section = '\n'.join(lines[:15])  # Check first 15 lines
    if 'from flask_limiter import Limiter' in import_section:
        print("✅ PASS: flask_limiter imports moved to top level")
        checks.append(True)
    else:
        print("❌ FAIL: flask_limiter imports not at top level")
        checks.append(False)

    # Check 4: Limiter stored in app.config
    if "app.config['LIMITER'] = limiter" in content:
        print("✅ PASS: Limiter stored in app.config (unused variable fixed)")
        checks.append(True)
    else:
        print("❌ FAIL: Limiter not stored in app.config")
        checks.append(False)

    # Check 5: Specific exception types (not broad Exception)
    if '(ImportError, ConnectionError, RuntimeError)' in content:
        print("✅ PASS: Specific exception types used for rate limiting")
        checks.append(True)
    else:
        print("❌ FAIL: Still using broad Exception for rate limiting")
        checks.append(False)

    if '(OSError, RuntimeError, ValueError)' in content:
        print("✅ PASS: Specific exception types used for server startup")
        checks.append(True)
    else:
        print("❌ FAIL: Still using broad Exception for server startup")
        checks.append(False)

    # Check 6: Lazy % formatting in logging (no f-strings)
    f_string_patterns = [
        'logger.error(f"Failed to configure',
        'logger.info(f"📍 Server will be',
        'logger.info(f"🔧 Using Waitress',
        'logger.error(f"Failed to start'
    ]

    has_f_strings = any(pattern in content for pattern in f_string_patterns)
    if not has_f_strings:
        print("✅ PASS: All logging uses lazy % formatting (no f-strings)")
        checks.append(True)
    else:
        print("❌ FAIL: Still using f-strings in logging")
        checks.append(False)

    # Check 7: Line length fix (split long line)
    if 'os.environ.setdefault(\n        \'SECRET_KEY\'' in content:
        print("✅ PASS: Long line properly split")
        checks.append(True)
    else:
        print("❌ FAIL: Long line not properly split")
        checks.append(False)

    return all(checks)

def test_logging_format():
    """Test 4: Verify logging format changes"""
    print("\nTest 4: Checking logging format...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, 'production_server.py')

    with open(server_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for % formatting patterns
    percent_patterns = [
        'logger.error("Failed to configure rate limiting: %s"',
        'logger.info("📍 Server will be available at: http://%s:%s"',
        'logger.info("🔧 Using Waitress WSGI server with %s threads"',
        'logger.error("Failed to start production server: %s"'
    ]

    checks = []
    for pattern in percent_patterns:
        if pattern in content:
            checks.append(True)
        else:
            print(f"❌ Missing pattern: {pattern}")
            checks.append(False)

    if all(checks):
        print(f"✅ PASS: All {len(checks)} logging statements use % formatting")
        return True
    else:
        print(f"❌ FAIL: {sum(checks)}/{len(checks)} logging statements correct")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("Production Server Diagnostic Fixes - Test Suite")
    print("=" * 70)

    results = []

    results.append(("Syntax Validation", test_syntax()))
    results.append(("Import Verification", test_imports()))
    results.append(("Code Structure", test_code_structure()))
    results.append(("Logging Format", test_logging_format()))

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All diagnostic fixes verified successfully!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
