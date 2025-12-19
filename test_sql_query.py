#!/usr/bin/env python3
"""
Critical-path testing for SQL query functionality on failed payments
Tests the implementation of: SELECT payment_id, amount, error_code, error_message, processed_at
FROM payments WHERE status = 'failed' ORDER BY processed_at DESC LIMIT 100
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.payments_service import payments_service
from src.models.payments import PaymentType

def test_error_message_functionality():
    """Test critical-path functionality for failed payments SQL query"""
    print("🧪 Starting critical-path testing for failed payments SQL query...")

    # Test 1: Create and process payments to generate failures
    print("\n1. Creating and processing payments to generate failures...")

    # Create multiple payments
    payment_ids = []
    for i in range(5):
        payment = payments_service.create_payment(
            amount=100.0 + i * 50,
            payment_type=PaymentType.CARD,
            user_id=f"user_{i}",
            description=f"Test payment {i}"
        )
        payment_ids.append(payment.id)
        print(f"   ✓ Created payment {payment.id} for ${payment.amount}")

    # Process payments (some will fail randomly)
    print("\n2. Processing payments (some will fail to generate error messages)...")
    for payment_id in payment_ids:
        success = payments_service.process_payment(payment_id)
        payment = payments_service.get_payment(payment_id)
        status = "SUCCESS" if success else "FAILED"
        error_info = f" - {payment.error_code}: {payment.error_message}" if payment.error_code else ""
        print(f"   {'✓' if success else '✗'} Payment {payment_id} {status}{error_info}")

    # Test 2: Test the SQL-style query method
    print("\n3. Testing get_failed_payments_sql_style() method...")
    failed_payments = payments_service.get_failed_payments_sql_style(limit=100)

    print(f"   Found {len(failed_payments)} failed payments")

    # Test 3: Verify data structure and content
    print("\n4. Verifying data structure and content...")
    required_fields = ['payment_id', 'amount', 'error_code', 'error_message', 'processed_at']

    for payment in failed_payments:
        # Check all required fields are present
        for field in required_fields:
            if field not in payment:
                print(f"   ✗ Missing field: {field}")
                return False
            else:
                print(f"   ✓ Field '{field}' present: {payment[field]}")

        # Verify error_code and error_message are not None for failed payments
        if payment['error_code'] is None:
            print(f"   ✗ error_code is None for payment {payment['payment_id']}")
            return False
        if payment['error_message'] is None:
            print(f"   ✗ error_message is None for payment {payment['payment_id']}")
            return False

        # Verify processed_at is a string (ISO format)
        if not isinstance(payment['processed_at'], str):
            print(f"   ✗ processed_at is not a string for payment {payment['payment_id']}")
            return False

        print(f"   ✓ Payment {payment['payment_id']}: ${payment['amount']}, {payment['error_code']}, processed at {payment['processed_at']}")

    # Test 4: Verify ordering (most recent first)
    print("\n5. Verifying ordering (most recent first)...")
    if len(failed_payments) > 1:
        for i in range(len(failed_payments) - 1):
            current_time = datetime.fromisoformat(failed_payments[i]['processed_at'].replace('Z', '+00:00'))
            next_time = datetime.fromisoformat(failed_payments[i + 1]['processed_at'].replace('Z', '+00:00'))
            if current_time < next_time:
                print(f"   ✗ Ordering incorrect: {current_time} should be after {next_time}")
                return False
        print("   ✓ Ordering is correct (most recent first)")

    # Test 5: Verify limit functionality
    print("\n6. Testing limit functionality...")
    limited_results = payments_service.get_failed_payments_sql_style(limit=2)
    if len(limited_results) > 2:
        print(f"   ✗ Limit not working: returned {len(limited_results)} instead of max 2")
        return False
    print(f"   ✓ Limit working correctly: returned {len(limited_results)} payments (limit=2)")

    print("\n🎉 All critical-path tests passed!")
    print(f"   - Error messages are properly stored and retrieved")
    print(f"   - SQL-style query method returns correct data structure")
    print(f"   - Failed payments include error_code and error_message")
    print(f"   - Results are ordered by processed_at DESC")
    print(f"   - Limit parameter works correctly")

    return True

if __name__ == "__main__":
    try:
        success = test_error_message_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
