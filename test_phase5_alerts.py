#!/usr/bin/env python3
"""
Test script for Phase 5: Notifications and Alerts
Tests bill tracking, payment reminders, budget alerts, goal achievement notifications
"""

import requests
import json
import time
from datetime import datetime, timedelta, timezone
from test_server_utils import ensure_local_test_server, stop_local_test_server

# Test configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_USER_ID = "test_user_123"

def test_bill_management():
    """Test bill creation, retrieval, and payment marking"""
    print("Testing Bill Management...")

    # Create a bill
    bill_data = {
        "user_id": TEST_USER_ID,
        "name": "Electricity Bill",
        "amount": 150.00,
        "due_date": (datetime.now(timezone.utc) + timedelta(days=5)).date().isoformat(),
        "category": "utilities",
        "frequency": "monthly",
        "reminder_days": 3
    }

    response = requests.post(f"{BASE_URL}/pfm/bills", json=bill_data)
    print(f"Create bill response: {response.status_code}")
    if response.status_code == 201:
        bill = response.json()
        bill_id = bill['bill']['bill_id']
        print(f"Bill created: {bill_id}")

        # Get bills
        response = requests.get(f"{BASE_URL}/pfm/bills?user_id={TEST_USER_ID}")
        print(f"Get bills response: {response.status_code}")
        if response.status_code == 200:
            bills_data = response.json()
            print(f"Found {len(bills_data['bills'])} bills")

        # Mark bill as paid
        pay_data = {
            "user_id": TEST_USER_ID,
            "amount": 150.00
        }
        response = requests.post(f"{BASE_URL}/pfm/bills/{bill_id}/pay", json=pay_data)
        print(f"Mark bill paid response: {response.status_code}")

    return True

def test_budget_alerts():
    """Test budget alert generation"""
    print("Testing Budget Alerts...")

    # Create a budget
    budget_data = {
        "user_id": TEST_USER_ID,
        "name": "Groceries Budget",
        "category": "groceries",
        "amount": 500.00
    }

    response = requests.post(f"{BASE_URL}/pfm/budgets", json=budget_data)
    print(f"Create budget response: {response.status_code}")
    if response.status_code == 201:
        budget = response.json()['budget']
        budget_id = budget['budget_id']

        # Simulate high spending (update budget spent amount)
        # In real implementation, this would be calculated from transactions
        # For testing, we'll manually set the spent amount
        budget['spent'] = 450.00  # 90% of budget

        # Check alerts
        response = requests.get(f"{BASE_URL}/pfm/alerts/check?user_id={TEST_USER_ID}")
        print(f"Check alerts response: {response.status_code}")
        if response.status_code == 200:
            alerts = response.json()['alerts']
            budget_alerts = [a for a in alerts if a['type'] == 'budget_warning']
            print(f"Found {len(budget_alerts)} budget alerts")

    return True

def test_goal_achievement_alerts():
    """Test goal achievement alert generation"""
    print("Testing Goal Achievement Alerts...")

    # Create a goal
    goal_data = {
        "user_id": TEST_USER_ID,
        "name": "Emergency Fund",
        "target_amount": 1000.00,
        "initial_amount": 750.00,  # 75% progress
        "target_date": (datetime.now(timezone.utc) + timedelta(days=90)).date().isoformat()
    }

    response = requests.post(f"{BASE_URL}/pfm/goals", json=goal_data)
    print(f"Create goal response: {response.status_code}")
    if response.status_code == 201:
        goal = response.json()['goal']
        goal_id = goal['goal_id']

        # Check alerts (should trigger 75% progress alert)
        response = requests.get(f"{BASE_URL}/pfm/alerts/check?user_id={TEST_USER_ID}")
        print(f"Check alerts response: {response.status_code}")
        if response.status_code == 200:
            alerts = response.json()['alerts']
            goal_alerts = [a for a in alerts if a['type'] in ['goal_progress', 'goal_achieved']]
            print(f"Found {len(goal_alerts)} goal alerts")

    return True

def test_bill_payment_reminders():
    """Test bill payment reminder alerts"""
    print("Testing Bill Payment Reminders...")

    # Create a bill due soon
    bill_data = {
        "user_id": TEST_USER_ID,
        "name": "Internet Bill",
        "amount": 80.00,
        "due_date": (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat(),  # Due in 2 days
        "category": "utilities",
        "frequency": "monthly",
        "reminder_days": 3
    }

    response = requests.post(f"{BASE_URL}/pfm/bills", json=bill_data)
    print(f"Create bill response: {response.status_code}")

    # Check alerts (should trigger bill due soon alert)
    response = requests.get(f"{BASE_URL}/pfm/alerts/check?user_id={TEST_USER_ID}")
    print(f"Check alerts response: {response.status_code}")
    if response.status_code == 200:
        alerts = response.json()['alerts']
        bill_alerts = [a for a in alerts if a['type'] in ['bill_due_soon', 'bill_overdue']]
        print(f"Found {len(bill_alerts)} bill alerts")

    return True

def test_account_balance_alerts():
    """Test account balance alert generation"""
    print("Testing Account Balance Alerts...")

    # Create an account with low balance
    account_data = {
        "user_id": TEST_USER_ID,
        "institution_id": "test_bank",
        "account_type": "checking",
        "account_name": "Test Checking"
    }

    response = requests.post(f"{BASE_URL}/pfm/accounts/link", json=account_data)
    print(f"Link account response: {response.status_code}")
    if response.status_code == 200:
        account = response.json()['account']

        # Set up balance monitoring with low threshold
        monitor_data = {
            "user_id": TEST_USER_ID,
            "account_id": account['account_id'],
            "alert_thresholds": {
                "low_balance": 200.00  # Current balance is 2450, so no alert
            }
        }

        response = requests.post(f"{BASE_URL}/pfm/accounts/monitor", json=monitor_data)
        print(f"Setup monitoring response: {response.status_code}")

        # Check alerts (should not trigger since balance is above threshold)
        response = requests.get(f"{BASE_URL}/pfm/alerts/check?user_id={TEST_USER_ID}")
        print(f"Check alerts response: {response.status_code}")
        if response.status_code == 200:
            alerts = response.json()['alerts']
            balance_alerts = [a for a in alerts if a['type'] == 'low_balance']
            print(f"Found {len(balance_alerts)} balance alerts")

    return True

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("Testing Error Handling...")

    # Test missing user_id
    response = requests.post(f"{BASE_URL}/pfm/bills", json={"name": "Test Bill"})
    print(f"Missing user_id response: {response.status_code}")

    # Test invalid bill data
    response = requests.post(f"{BASE_URL}/pfm/bills", json={
        "user_id": TEST_USER_ID,
        "amount": -100.00  # Invalid negative amount
    })
    print(f"Invalid amount response: {response.status_code}")

    return True

def main():
    """Run all Phase 5 tests"""
    print("Starting Phase 5: Notifications and Alerts Testing")
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

        # Test bill management
        test_bill_management()
        print()

        # Test budget alerts
        test_budget_alerts()
        print()

        # Test goal achievement alerts
        test_goal_achievement_alerts()
        print()

        # Test bill payment reminders
        test_bill_payment_reminders()
        print()

        # Test account balance alerts
        test_account_balance_alerts()
        print()

        # Test error handling
        test_error_handling()
        print()

        print("Phase 5 testing completed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        if started_here:
            stop_local_test_server()
            print("Stopped local in-process PFM test server")

if __name__ == "__main__":
    main()
