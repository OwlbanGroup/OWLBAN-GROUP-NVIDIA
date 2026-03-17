#!/usr/bin/env python3
"""
Test script for Phase 8: Advanced Features
Tests bill tracking and payment scheduling, recurring transaction detection,
investment tracking, and financial planning tools
"""

import os
import requests
import json
import time
from datetime import datetime, timedelta, timezone
from test_server_utils import ensure_local_test_server, stop_local_test_server

# Test configuration
BASE_URL = os.getenv("PHASE8_BASE_URL", "http://127.0.0.1:5000")
TEST_USER_ID = "test_user_phase8"

def request_with_retries(method, path, **kwargs):
    """Request helper with short retry/backoff for server readiness and transient errors."""
    url = f"{BASE_URL}{path}"
    last_exc = None
    for attempt in range(6):
        try:
            return requests.request(method, url, timeout=5, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.4 * (attempt + 1))
    raise last_exc

def test_bill_scheduling():
    """Test bill payment scheduling"""
    print("Testing Bill Scheduling...")

    # First create a bill
    bill_data = {
        "user_id": TEST_USER_ID,
        "name": "Internet Bill",
        "amount": 80.00,
        "due_date": (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat(),
        "category": "utilities",
        "frequency": "monthly"
    }

    response = request_with_retries("POST", "/pfm/bills", json=bill_data)
    print(f"Create bill response: {response.status_code}")
    
    if response.status_code == 201:
        bill = response.json()['bill']
        bill_id = bill['bill_id']
        print(f"Bill created: {bill_id}")

        # Schedule a payment
        schedule_data = {
            "user_id": TEST_USER_ID,
            "bill_id": bill_id,
            "payment_date": (datetime.now(timezone.utc) + timedelta(days=5)).date().isoformat(),
            "payment_method": "bank_transfer",
            "is_recurring": True
        }

        response = request_with_retries("POST", "/pfm/bills/schedule", json=schedule_data)
        print(f"Schedule payment response: {response.status_code}")
        if response.status_code == 201:
            scheduled = response.json()['scheduled_payment']
            print(f"Payment scheduled: {scheduled.get('schedule_id')}")

        # Get scheduled payments
        response = request_with_retries("GET", f"/pfm/bills/scheduled?user_id={TEST_USER_ID}")
        print(f"Get scheduled payments response: {response.status_code}")
        if response.status_code == 200:
            scheduled_list = response.json()['scheduled_payments']
            print(f"Found {len(scheduled_list)} scheduled payments")

    return True


def test_recurring_transaction_detection():
    """Test recurring transaction detection"""
    print("Testing Recurring Transaction Detection...")

    # First, add some transactions to detect recurring patterns
    transactions = [
        {
            "user_id": TEST_USER_ID,
            "transaction_id": "txn_rec_1",
            "description": "Netflix Subscription",
            "amount": -15.99,
            "date": "2024-01-01"
        },
        {
            "user_id": TEST_USER_ID,
            "transaction_id": "txn_rec_2",
            "description": "Netflix Subscription",
            "amount": -15.99,
            "date": "2024-02-01"
        },
        {
            "user_id": TEST_USER_ID,
            "transaction_id": "txn_rec_3",
            "description": "Netflix Subscription",
            "amount": -15.99,
            "date": "2024-03-01"
        },
        {
            "user_id": TEST_USER_ID,
            "transaction_id": "txn_rec_4",
            "description": "Spotify Premium",
            "amount": -9.99,
            "date": "2024-01-15"
        },
        {
            "user_id": TEST_USER_ID,
            "transaction_id": "txn_rec_5",
            "description": "Spotify Premium",
            "amount": -9.99,
            "date": "2024-02-15"
        }
    ]

    # Categorize transactions first
    response = request_with_retries("POST", "/pfm/transactions/categorize", json={
        "user_id": TEST_USER_ID,
        "transactions": transactions
    })
    print(f"Categorize transactions response: {response.status_code}")

    # Detect recurring transactions
    detect_data = {
        "user_id": TEST_USER_ID,
        "min_occurrences": 2
    }

    response = request_with_retries("POST", "/pfm/transactions/recurring/detect", json=detect_data)
    print(f"Detect recurring response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        recurring = result.get('recurring_transactions', [])
        print(f"Detected {len(recurring)} recurring transactions")
        for r in recurring:
            print(f"  - {r.get('description')}: ${r.get('average_amount')}/month")

    # Get recurring transactions
    response = request_with_retries("GET", f"/pfm/transactions/recurring?user_id={TEST_USER_ID}")
    print(f"Get recurring response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Summary: {result.get('summary', {})}")

    return True


def test_investment_tracking():
    """Test investment tracking"""
    print("Testing Investment Tracking...")

    # Add an investment
    investment_data = {
        "user_id": TEST_USER_ID,
        "name": "Apple Inc.",
        "type": "stock",
        "symbol": "AAPL",
        "shares": 10,
        "purchase_price": 150.00,
        "current_price": 175.50,
        "purchase_date": "2023-06-15"
    }

    response = request_with_retries("POST", "/pfm/investments", json=investment_data)
    print(f"Add investment response: {response.status_code}")
    if response.status_code == 201:
        investment = response.json()['investment']
        print(f"Investment added: {investment.get('investment_id')}")
        print(f"  Gain/Loss: ${investment.get('gain_loss')} ({investment.get('gain_loss_percentage')}%)")

    # Add another investment
    investment_data2 = {
        "user_id": TEST_USER_ID,
        "name": "Vanguard Total Stock Market ETF",
        "type": "etf",
        "symbol": "VTI",
        "shares": 50,
        "purchase_price": 200.00,
        "current_price": 220.00,
        "purchase_date": "2023-01-10"
    }

    response = request_with_retries("POST", "/pfm/investments", json=investment_data2)
    print(f"Add investment 2 response: {response.status_code}")

    # Get all investments
    response = request_with_retries("GET", f"/pfm/investments?user_id={TEST_USER_ID}")
    print(f"Get investments response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        investments = result.get('investments', [])
        summary = result.get('summary', {})
        print(f"Found {len(investments)} investments")
        print(f"Portfolio Summary:")
        print(f"  Total Value: ${summary.get('total_value')}")
        print(f"  Total Cost: ${summary.get('total_cost')}")
        print(f"  Total Gain/Loss: ${summary.get('total_gain_loss')} ({summary.get('total_gain_loss_percentage')}%)")

    return True


def test_retirement_planning():
    """Test retirement planning calculator"""
    print("Testing Retirement Planning...")

    retirement_data = {
        "user_id": TEST_USER_ID,
        "current_age": 30,
        "retirement_age": 65,
        "current_savings": 50000,
        "monthly_contribution": 1000,
        "desired_income_annual": 60000,
        "expected_return_annual": 0.07
    }

    response = request_with_retries("POST", "/pfm/planning/retirement", json=retirement_data)
    print(f"Retirement planning response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        plan = result.get('retirement_plan', {})
        projections = plan.get('projections', {})
        analysis = plan.get('analysis', {})
        
        print(f"Retirement Plan:")
        print(f"  Years to retirement: {plan.get('inputs', {}).get('years_to_retirement')}")
        print(f"  Total retirement savings: ${projections.get('total_retirement_savings')}")
        print(f"  Monthly income (4% rule): ${projections.get('monthly_income_4_percent')}")
        print(f"  Meets income goal: {analysis.get('meets_income_goal')}")
        if not analysis.get('meets_income_goal'):
            print(f"  Additional monthly needed: ${analysis.get('additional_monthly_needed')}")

    return True


def test_debt_payoff_planning():
    """Test debt payoff planning"""
    print("Testing Debt Payoff Planning...")

    debt_data = {
        "user_id": TEST_USER_ID,
        "debts": [
            {
                "name": "Credit Card A",
                "balance": 5000,
                "interest_rate": 19.99,
                "minimum_payment": 100
            },
            {
                "name": "Car Loan",
                "balance": 15000,
                "interest_rate": 6.5,
                "minimum_payment": 350
            },
            {
                "name": "Student Loan",
                "balance": 25000,
                "interest_rate": 4.5,
                "minimum_payment": 250
            }
        ],
        "monthly_budget": 1000,
        "strategy": "avalanche"  # Highest interest first
    }

    response = request_with_retries("POST", "/pfm/planning/debt-payoff", json=debt_data)
    print(f"Debt payoff planning response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        plan = result.get('debt_plan', {})
        summary = plan.get('summary', {})
        
        print(f"Debt Payoff Plan (Avalanche Method):")
        print(f"  Total debt: ${summary.get('total_debt')}")
        print(f"  Estimated months to payoff: {summary.get('estimated_months')}")
        print(f"  Debt-free date: {summary.get('debt_free_date')}")
        
        payoff_order = plan.get('payoff_order', [])
        print(f"  Payoff order:")
        for debt in payoff_order:
            print(f"    - {debt.get('name')}: {debt.get('months_to_payoff')} months")

    # Test snowball method
    debt_data["strategy"] = "snowball"
    response = request_with_retries("POST", "/pfm/planning/debt-payoff", json=debt_data)
    print(f"Debt payoff (snowball) response: {response.status_code}")

    return True


def test_savings_goal_planning():
    """Test savings goal planning"""
    print("Testing Savings Goal Planning...")

    goal_data = {
        "user_id": TEST_USER_ID,
        "name": "Emergency Fund",
        "target_amount": 10000,
        "current_savings": 2500,
        "monthly_contribution": 500,
        "expected_return_annual": 0.04
    }

    response = request_with_retries("POST", "/pfm/planning/savings-goal", json=goal_data)
    print(f"Savings goal planning response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        plan = result.get('savings_plan', {})
        
        print(f"Savings Goal Plan:")
        print(f"  Target: ${plan.get('target_amount')}")
        print(f"  Current: ${plan.get('current_savings')}")
        print(f"  Monthly contribution: ${plan.get('monthly_contribution')}")
        print(f"  Months to goal: {plan.get('months_to_goal')}")
        print(f"  Goal date: {plan.get('goal_date')}")
        
        projections = plan.get('projections', {})
        print(f"  Total contributions: ${projections.get('total_contributions')}")
        print(f"  Interest earned: ${projections.get('total_interest_earned')}")

    # Test case where goal is already reached
    goal_data2 = {
        "user_id": TEST_USER_ID,
        "name": "Already Reached Goal",
        "target_amount": 5000,
        "current_savings": 6000,
        "monthly_contribution": 100
    }

    response = request_with_retries("POST", "/pfm/planning/savings-goal", json=goal_data2)
    print(f"Already reached goal response: {response.status_code}")
    if response.status_code == 200:
        print(f"  Goal already reached!")

    return True


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("Testing Error Handling...")

    # Test missing user_id for bill scheduling
    response = request_with_retries("POST", "/pfm/bills/schedule", json={
        "bill_id": "some_bill_id",
        "payment_date": "2024-12-31"
    })
    print(f"Missing user_id response: {response.status_code}")

    # Test invalid investment data
    response = request_with_retries("POST", "/pfm/investments", json={
        "user_id": TEST_USER_ID,
        "amount": -100  # Invalid - missing required fields
    })
    print(f"Invalid investment response: {response.status_code}")

    # Test invalid retirement planning (retirement age < current age)
    response = request_with_retries("POST", "/pfm/planning/retirement", json={
        "user_id": TEST_USER_ID,
        "current_age": 70,
        "retirement_age": 65  # Invalid - less than current age
    })
    print(f"Invalid retirement age response: {response.status_code}")

    # Test invalid debt planning (no monthly budget)
    response = request_with_retries("POST", "/pfm/planning/debt-payoff", json={
        "user_id": TEST_USER_ID,
        "debts": [{"name": "Test", "balance": 1000, "interest_rate": 5}],
        "monthly_budget": 0
    })
    print(f"Invalid debt budget response: {response.status_code}")

    return True


def main():
    """Run all Phase 8 tests"""
    print("Starting Phase 8: Advanced Features Testing")
    print("=" * 50)

    started_here = False
    try:
        started_here, _ = ensure_local_test_server(BASE_URL)
        if started_here:
            print("Started local in-process PFM test server")
        else:
            print("Using existing healthy PFM server")

        # Wait a moment for server readiness
        time.sleep(1)

        # Test bill scheduling
        test_bill_scheduling()
        print()

        # Test recurring transaction detection
        test_recurring_transaction_detection()
        print()

        # Test investment tracking
        test_investment_tracking()
        print()

        # Test retirement planning
        test_retirement_planning()
        print()

        # Test debt payoff planning
        test_debt_payoff_planning()
        print()

        # Test savings goal planning
        test_savings_goal_planning()
        print()

        # Test error handling
        test_error_handling()
        print()

        print("Phase 8 testing completed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        if started_here:
            stop_local_test_server()
            print("Stopped local in-process PFM test server")


if __name__ == "__main__":
    main()
