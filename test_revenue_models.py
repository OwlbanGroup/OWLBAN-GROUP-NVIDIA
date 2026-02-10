#!/usr/bin/env python3
"""
Test script to verify revenue models can be imported and instantiated
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.revenue import RevenueTransaction, RevenueMetrics, RevenueType, TransactionStatus
    print("✓ Successfully imported revenue models")

    # Test enum values
    print(f"✓ RevenueType values: {[e.value for e in RevenueType]}")
    print(f"✓ TransactionStatus values: {[e.value for e in TransactionStatus]}")

    # Test model instantiation (without database)
    transaction = RevenueTransaction(
        transaction_id="test-123",
        user_id="user-456",
        revenue_type=RevenueType.PURCHASE,
        amount=100.0,
        net_amount=95.0,
        status=TransactionStatus.PENDING
    )
    print("✓ Successfully created RevenueTransaction instance")

    metrics = RevenueMetrics(
        date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        revenue_type=RevenueType.PURCHASE
    )
    print("✓ Successfully created RevenueMetrics instance")

    # Test to_dict methods
    transaction_dict = transaction.to_dict()
    metrics_dict = metrics.to_dict()
    print("✓ Successfully converted models to dictionaries")

    print("\nAll tests passed! Revenue models are working correctly.")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
