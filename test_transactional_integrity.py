#!/usr/bin/env python3
"""
Test Transactional Integrity for JPMorgan Financial APIs
Tests ACID compliance and rollback mechanisms
"""
import sys
import os
from datetime import datetime, timezone

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.transaction_manager import (
        transaction_manager,
        business_transaction_manager,
        payment_transaction_manager
    )
    from src.database_fixed import db_manager
    from src.logger import telemetry_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_basic_transaction_commit():
    """Test basic transaction commit"""
    print("🧪 Testing basic transaction commit...")

    try:
        import uuid
        # Create a test business
        test_business = {
            'name': 'Test Transaction Company',
            'type': 'corporation',
            'registration_number': f'TXN{uuid.uuid4().hex[:8].upper()}',
            'address': '123 Transaction St',
            'contact_info': {'email': 'test@transaction.com', 'phone': '555-0123'}
        }

        with transaction_manager.transaction() as session:
            business = db_manager.create_business(test_business)
            print(f"  ✅ Business created: {business.id}")

        # Verify business exists
        retrieved = db_manager.get_business_by_id(business.id)
        assert retrieved is not None, "Business should exist after commit"
        assert retrieved.name == test_business['name'], "Business data should match"

        print("  ✅ Transaction committed successfully")
        return True

    except Exception as e:
        print(f"  ❌ Basic transaction commit failed: {e}")
        return False

def test_transaction_rollback():
    """Test transaction rollback on error"""
    print("🧪 Testing transaction rollback...")

    try:
        # Try to create business with invalid data that should cause rollback
        invalid_business = {
            'name': None,  # Invalid: name cannot be null
            'type': 'corporation',
            'registration_number': 'ROLLBACK123',
        }

        try:
            with transaction_manager.transaction() as session:
                business = db_manager.create_business(invalid_business)
                # This should not be reached if rollback works
                print("  ❌ Transaction should have rolled back")
                return False
        except Exception as e:
            print(f"  ✅ Transaction rolled back as expected: {type(e).__name__}")

        # Verify no business was created (check by registration number)
        businesses = db_manager.get_all_businesses()
        rollback_business = next((b for b in businesses if b.registration_number == 'ROLLBACK123'), None)
        assert rollback_business is None, "Business should not exist after rollback"

        print("  ✅ Transaction rollback verified")
        return True

    except Exception as e:
        print(f"  ❌ Transaction rollback test failed: {e}")
        return False

def test_business_with_assets_transaction():
    """Test creating business with assets in single transaction"""
    print("🧪 Testing business with assets transaction...")

    try:
        import uuid
        business_data = {
            'name': 'Asset Management Corp',
            'type': 'corporation',
            'registration_number': f'ASSET{uuid.uuid4().hex[:8].upper()}',
            'address': '456 Asset Ave',
            'contact_info': {'email': 'assets@corp.com'}
        }

        assets_data = [
            {
                'name': 'Office Building',
                'type': 'real_estate',
                'value': 1000000.0,
                'acquisition_date': '2023-01-15T00:00:00Z',
                'description': 'Headquarters building'
            },
            {
                'name': 'Company Vehicles',
                'type': 'equipment',
                'value': 50000.0,
                'acquisition_date': '2023-02-01T00:00:00Z',
                'description': 'Fleet of company vehicles'
            }
        ]

        business, assets = business_transaction_manager.create_business_with_assets(
            business_data, assets_data
        )

        print(f"  ✅ Business created: {business.id}")
        print(f"  ✅ Assets created: {len(assets)}")

        # Verify business and assets exist
        retrieved_business = db_manager.get_business_by_id(business.id)
        assert retrieved_business is not None, "Business should exist"

        retrieved_assets = db_manager.get_assets_by_business_id(business.id)
        assert len(retrieved_assets) == 2, "Both assets should exist"

        print("  ✅ Business with assets transaction successful")
        return True

    except Exception as e:
        print(f"  ❌ Business with assets transaction failed: {e}")
        return False

def test_asset_transfer_transaction():
    """Test asset ownership transfer between businesses"""
    print("🧪 Testing asset transfer transaction...")

    try:
        # Create two businesses
        business1_data = {
            'name': 'Transfer Source Inc',
            'type': 'corporation',
            'registration_number': 'SRC001',
        }

        business2_data = {
            'name': 'Transfer Target LLC',
            'type': 'llc',
            'registration_number': 'TGT001',
        }

        business1, _ = business_transaction_manager.create_business_with_assets(
            business1_data, []
        )
        business2, _ = business_transaction_manager.create_business_with_assets(
            business2_data, []
        )

        # Create an asset for business1
        asset_data = {
            'business_id': business1.id,
            'name': 'Transferable Asset',
            'type': 'equipment',
            'value': 10000.0,
            'acquisition_date': '2023-03-01T00:00:00Z',
        }

        with transaction_manager.transaction() as session:
            asset = db_manager.create_asset(asset_data)

        # Transfer asset ownership
        transfer_success = business_transaction_manager.transfer_asset_ownership(
            asset.id, business2.id, 12000.0  # New value after transfer
        )

        assert transfer_success, "Asset transfer should succeed"

        # Verify asset ownership changed
        updated_asset = db_manager.get_asset_by_id(asset.id)
        assert updated_asset.business_id == business2.id, "Asset should belong to new business"
        assert updated_asset.current_value == 12000.0, "Asset value should be updated"

        print("  ✅ Asset transfer transaction successful")
        return True

    except Exception as e:
        print(f"  ❌ Asset transfer transaction failed: {e}")
        return False

def test_payment_processing_transaction():
    """Test payment processing with fee calculation"""
    print("🧪 Testing payment processing transaction...")

    try:
        from src.payments_service import payments_service

        payment_data = {
            'amount': 100.0,
            'payment_type': 'card',
            'user_id': 'test_user_txn',
            'description': 'Test transaction payment',
            'currency': 'USD'
        }

        fee_data = {
            'fee_type': 'processing_fee',
            'fee_percentage': 0.029  # 2.9%
        }

        payment, fee = payment_transaction_manager.process_payment_with_fee(
            payment_data, fee_data
        )

        print(f"  ✅ Payment created: {payment.id}")
        print(f"  ✅ Fee calculated: ${fee['amount']:.2f}")

        # Verify payment exists
        retrieved_payment = payments_service.get_payment(payment.id)
        assert retrieved_payment is not None, "Payment should exist"
        assert retrieved_payment.amount == 100.0, "Payment amount should match"

        print("  ✅ Payment processing transaction successful")
        return True

    except Exception as e:
        print(f"  ❌ Payment processing transaction failed: {e}")
        return False

def test_transaction_retry_mechanism():
    """Test transaction retry on transient failures"""
    print("🧪 Testing transaction retry mechanism...")

    retry_count = 0
    max_retries = 3

    def failing_operation(session):
        nonlocal retry_count
        retry_count += 1
        if retry_count < max_retries:
            # Simulate transient failure
            raise RuntimeError("Simulated transient failure")
        return "success"

    try:
        result = transaction_manager.execute_with_retry(failing_operation, max_retries=max_retries)
        assert result == "success", "Operation should eventually succeed"
        assert retry_count == max_retries, f"Should have retried {max_retries} times"

        print(f"  ✅ Transaction retry successful after {retry_count} attempts")
        return True

    except Exception as e:
        print(f"  ❌ Transaction retry test failed: {e}")
        return False

def test_transaction_health_check():
    """Test transaction manager health check"""
    print("🧪 Testing transaction health check...")

    try:
        health = transaction_manager.health_check()
        assert health['status'] == 'healthy', "Health check should pass"
        assert 'timestamp' in health, "Health check should include timestamp"

        print("  ✅ Transaction health check passed")
        return True

    except Exception as e:
        print(f"  ❌ Transaction health check failed: {e}")
        return False

def run_transactional_integrity_tests():
    """Run all transactional integrity tests"""
    print("🚀 Starting Transactional Integrity Tests...")
    print(f"⏰ Start time: {datetime.now(timezone.utc).isoformat()}")

    tests = [
        test_basic_transaction_commit,
        test_transaction_rollback,
        test_business_with_assets_transaction,
        test_asset_transfer_transaction,
        test_payment_processing_transaction,
        test_transaction_retry_mechanism,
        test_transaction_health_check,
    ]

    results = []
    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
            if result:
                passed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            results.append((test.__name__, False))

        print()  # Add spacing between tests

    print("📋 Transactional Integrity Test Results:")
    print(f"  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print(f"  Success rate: {passed/total:.1%}")
    print(f"⏰ End time: {datetime.now(timezone.utc).isoformat()}")

    # Log results
    logger = telemetry_logger.get_logger()
    logger.info("Transactional integrity tests completed", extra={
        'test_results': results,
        'passed': passed,
        'total': total,
        'success_rate': passed/total
    })

    return passed == total

def main():
    """Main test function"""
    success = run_transactional_integrity_tests()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
