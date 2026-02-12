"""
Personal Finance Management (PFM) Blueprint for JPMorgan Financial APIs
Provides comprehensive PFM features including account linking, budgeting, financial goals,
spending insights, and consumer-facing financial management capabilities.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, List
import json

# Import services and utilities
from src.logger import telemetry_logger

# Import authentication and rate limiting decorators
try:
    from src.auth import token_auth_required
    from src.rate_limiting import conditional_limit
except ImportError:
    # Fallback if not found - these would need to be implemented
    def token_auth_required(f):
        return f
    def conditional_limit(rate):
        def decorator(f):
            return f
        return decorator

pfm_bp = Blueprint('pfm', __name__)

# Mock data storage for PFM features (in real implementation, this would be a database)
_mock_accounts = {}
_mock_budgets = {}
_mock_goals = {}
_mock_notifications = {}
_mock_financial_health = {}
_mock_transactions = {}
_mock_balance_alerts = {}

# Sample mock transactions for testing
_mock_transactions = {
    'user123': [
        {'transaction_id': 'txn1', 'account_id': 'acc1', 'amount': -45.67, 'description': 'Grocery Store Purchase', 'date': '2024-01-15', 'category': None},
        {'transaction_id': 'txn2', 'account_id': 'acc1', 'amount': -25.00, 'description': 'Starbucks Coffee', 'date': '2024-01-16', 'category': None},
        {'transaction_id': 'txn3', 'account_id': 'acc1', 'amount': -120.50, 'description': 'Gas Station Fill-up', 'date': '2024-01-17', 'category': None},
        {'transaction_id': 'txn4', 'account_id': 'acc2', 'amount': -89.99, 'description': 'Amazon Purchase', 'date': '2024-01-18', 'category': None},
        {'transaction_id': 'txn5', 'account_id': 'acc1', 'amount': 1500.00, 'description': 'Salary Deposit', 'date': '2024-01-01', 'category': 'income'}
    ]
}

def categorize_transaction(description: str) -> str:
    """
    Simple transaction categorization based on keywords
    """
    description_lower = description.lower()

    if any(word in description_lower for word in ['grocery', 'supermarket', 'food lion', 'walmart', 'target', 'safeway']):
        return 'groceries'
    if any(word in description_lower for word in ['restaurant', 'dining', 'mcdonald', 'starbucks', 'pizza', 'burger king']):
        return 'dining'
    if any(word in description_lower for word in ['gas', 'fuel', 'shell', 'bp', 'exxon', 'mobil']):
        return 'transportation'
    if any(word in description_lower for word in ['netflix', 'spotify', 'amazon prime', 'hulu', 'subscription', 'entertainment']):
        return 'entertainment'
    if any(word in description_lower for word in ['electric', 'water', 'gas bill', 'utility', 'comcast', 'verizon']):
        return 'utilities'
    if any(word in description_lower for word in ['amazon', 'best buy', 'shopping', 'mall', 'store']):
        return 'shopping'
    if any(word in description_lower for word in ['salary', 'deposit', 'income', 'payroll']):
        return 'income'

    return 'other'

def calculate_budget_spent(user_id: str, budget_category: str, start_date: str = None) -> float:
    """
    Calculate total spent amount for a budget category from transactions
    """
    transactions = _mock_transactions.get(user_id, [])
    total_spent = 0.0

    for txn in transactions:
        if txn.get('category') == budget_category:
            amount = txn.get('amount', 0)
            if amount < 0:  # Only count expenses (negative amounts)
                total_spent += abs(amount)

    return total_spent


# =============================================================================
# ACCOUNT LINKING AND AGGREGATION ENDPOINTS
# =============================================================================

@pfm_bp.route('/pfm/accounts/link', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def link_financial_account():
    """
    Link a financial account (Plaid-like account linking simulation)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No account data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        institution_id = data.get('institution_id')
        account_type = data.get('account_type', 'checking')  # checking, savings, credit_card, investment
        account_name = data.get('account_name')
        public_token = data.get('public_token')  # Mock Plaid public token

        if not all([user_id, institution_id, account_name]):
            return jsonify({'error': 'User ID, institution ID, and account name are required', 'status': 'error'}), 400

        # Mock account linking process
        account_id = str(uuid.uuid4())

        # Simulate different account types with realistic balances
        if account_type == 'checking':
            balance = 2450.75
            available_balance = 2450.75
        elif account_type == 'savings':
            balance = 15750.00
            available_balance = 15750.00
        elif account_type == 'credit_card':
            balance = -1250.50  # Negative for credit cards (amount owed)
            available_balance = 2249.50  # Credit limit minus balance
        elif account_type == 'investment':
            balance = 45680.25
            available_balance = 45680.25
        else:
            balance = 1000.00
            available_balance = 1000.00

        linked_account = {
            'account_id': account_id,
            'user_id': user_id,
            'institution_id': institution_id,
            'account_type': account_type,
            'account_name': account_name,
            'account_number': f"****{str(uuid.uuid4().hex[:4]).upper()}",  # Masked account number
            'routing_number': '123456789',
            'balance': balance,
            'available_balance': available_balance,
            'currency': 'USD',
            'status': 'active',
            'linked_at': datetime.now(timezone.utc).isoformat(),
            'last_synced': datetime.now(timezone.utc).isoformat()
        }

        if user_id not in _mock_accounts:
            _mock_accounts[user_id] = []
        _mock_accounts[user_id].append(linked_account)

        telemetry_logger.log_info(f"Account linked successfully: {account_id}")

        return jsonify({
            'status': 'success',
            'message': 'Account linked successfully',
            'account': {
                'account_id': account_id,
                'account_name': account_name,
                'account_type': account_type,
                'balance': balance,
                'status': 'active'
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'link_financial_account'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/accounts', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_linked_accounts():
    """
    Get all linked financial accounts for a user
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        accounts = _mock_accounts.get(user_id, [])

        # Calculate totals
        total_balance = sum(acc['balance'] for acc in accounts if acc['account_type'] != 'credit_card')
        total_debt = abs(sum(acc['balance'] for acc in accounts if acc['account_type'] == 'credit_card' and acc['balance'] < 0))
        net_worth = total_balance - total_debt

        return jsonify({
            'status': 'success',
            'accounts': accounts,
            'summary': {
                'total_accounts': len(accounts),
                'total_balance': total_balance,
                'total_debt': total_debt,
                'net_worth': net_worth,
                'currency': 'USD'
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_linked_accounts'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/accounts/<account_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_account_details(account_id):
    """
    Get detailed information for a specific account
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        accounts = _mock_accounts.get(user_id, [])
        account = next((acc for acc in accounts if acc['account_id'] == account_id), None)

        if not account:
            return jsonify({'error': 'Account not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'account': account,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_account_details'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/accounts/sync', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def sync_accounts():
    """
    Sync account balances and transactions from linked institutions
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        accounts = _mock_accounts.get(user_id, [])
        synced_accounts = []

        for account in accounts:
            # Mock balance updates
            if account['account_type'] == 'checking':
                account['balance'] += 150.25  # Mock new deposits
            elif account['account_type'] == 'credit_card':
                account['balance'] -= 75.50  # Mock new charges

            account['last_synced'] = datetime.now(timezone.utc).isoformat()
            synced_accounts.append(account)

        telemetry_logger.log_info(f"Accounts synced for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': f'Synced {len(synced_accounts)} accounts',
            'synced_accounts': len(synced_accounts),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'sync_accounts'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# BUDGETING ENDPOINTS
# =============================================================================

@pfm_bp.route('/pfm/budgets', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_budget():
    """
    Create a new budget
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No budget data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        budget_name = data.get('name')
        category = data.get('category')
        amount = data.get('amount')
        period = data.get('period', 'monthly')  # monthly, weekly, yearly
        start_date = data.get('start_date', datetime.now(timezone.utc).date().isoformat())

        if not all([user_id, budget_name, category, amount]):
            return jsonify({'error': 'User ID, name, category, and amount are required', 'status': 'error'}), 400

        if amount <= 0:
            return jsonify({'error': 'Budget amount must be positive', 'status': 'error'}), 400

        budget_id = str(uuid.uuid4())

        budget = {
            'budget_id': budget_id,
            'user_id': user_id,
            'name': budget_name,
            'category': category,
            'amount': amount,
            'period': period,
            'start_date': start_date,
            'spent': 0.00,
            'remaining': amount,
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'alerts_enabled': True
        }

        if user_id not in _mock_budgets:
            _mock_budgets[user_id] = []
        _mock_budgets[user_id].append(budget)

        telemetry_logger.log_info(f"Budget created: {budget_id}")

        return jsonify({
            'status': 'success',
            'message': 'Budget created successfully',
            'budget': budget,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_budget'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/budgets', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_budgets():
    """
    Get all budgets for a user
    """
    try:
        user_id = request.args.get('user_id')
        category = request.args.get('category')
        status_filter = request.args.get('status', 'active')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        budgets = _mock_budgets.get(user_id, [])

        # Apply filters
        if category:
            budgets = [b for b in budgets if b['category'].lower() == category.lower()]
        if status_filter:
            budgets = [b for b in budgets if b['status'] == status_filter]

        # Calculate summary
        total_budgeted = sum(b['amount'] for b in budgets)
        total_spent = sum(b['spent'] for b in budgets)
        total_remaining = sum(b['remaining'] for b in budgets)

        return jsonify({
            'status': 'success',
            'budgets': budgets,
            'summary': {
                'total_budgets': len(budgets),
                'total_budgeted': total_budgeted,
                'total_spent': total_spent,
                'total_remaining': total_remaining,
                'average_utilization': (total_spent / total_budgeted * 100) if total_budgeted > 0 else 0
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_budgets'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/budgets/<budget_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_budget(budget_id):
    """
    Update an existing budget
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No update data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        budgets = _mock_budgets.get(user_id, [])
        budget = next((b for b in budgets if b['budget_id'] == budget_id), None)

        if not budget:
            return jsonify({'error': 'Budget not found', 'status': 'error'}), 404

        # Update allowed fields
        updatable_fields = ['name', 'amount', 'period', 'alerts_enabled']
        for field in updatable_fields:
            if field in data:
                budget[field] = data[field]

        # Recalculate remaining if amount changed
        if 'amount' in data:
            budget['remaining'] = data['amount'] - budget['spent']

        budget['updated_at'] = datetime.now(timezone.utc).isoformat()

        telemetry_logger.log_info(f"Budget updated: {budget_id}")

        return jsonify({
            'status': 'success',
            'message': 'Budget updated successfully',
            'budget': budget,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_budget'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/budgets/<budget_id>/progress', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_budget_progress(budget_id):
    """
    Get detailed progress for a specific budget
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        budgets = _mock_budgets.get(user_id, [])
        budget = next((b for b in budgets if b['budget_id'] == budget_id), None)

        if not budget:
            return jsonify({'error': 'Budget not found', 'status': 'error'}), 404

        # Calculate progress metrics
        spent_percentage = (budget['spent'] / budget['amount']) * 100
        remaining_percentage = 100 - spent_percentage

        # Determine status
        if spent_percentage >= 100:
            status = 'exceeded'
        elif spent_percentage >= 80:
            status = 'warning'
        else:
            status = 'on_track'

        progress = {
            'budget_id': budget_id,
            'budget_name': budget['name'],
            'category': budget['category'],
            'budgeted_amount': budget['amount'],
            'spent_amount': budget['spent'],
            'remaining_amount': budget['remaining'],
            'spent_percentage': round(spent_percentage, 2),
            'remaining_percentage': round(remaining_percentage, 2),
            'status': status,
            'period': budget['period'],
            'days_remaining': 15,  # Mock remaining days in period
            'alerts_enabled': budget['alerts_enabled']
        }

        return jsonify({
            'status': 'success',
            'progress': progress,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_budget_progress'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# FINANCIAL GOALS ENDPOINTS
# =============================================================================

@pfm_bp.route('/pfm/goals', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_financial_goal():
    """
    Create a new financial goal
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No goal data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        goal_name = data.get('name')
        goal_type = data.get('type', 'savings')  # savings, debt_payoff, investment, emergency_fund
        target_amount = data.get('target_amount')
        target_date = data.get('target_date')
        initial_amount = data.get('initial_amount', 0.00)

        if not all([user_id, goal_name, target_amount]):
            return jsonify({'error': 'User ID, name, and target amount are required', 'status': 'error'}), 400

        if target_amount <= 0:
            return jsonify({'error': 'Target amount must be positive', 'status': 'error'}), 400

        goal_id = str(uuid.uuid4())

        goal = {
            'goal_id': goal_id,
            'user_id': user_id,
            'name': goal_name,
            'type': goal_type,
            'target_amount': target_amount,
            'current_amount': initial_amount,
            'target_date': target_date,
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'contributions': [],
            'notifications_enabled': True
        }

        # Calculate progress
        goal['progress_percentage'] = (initial_amount / target_amount) * 100
        goal['remaining_amount'] = target_amount - initial_amount

        if user_id not in _mock_goals:
            _mock_goals[user_id] = []
        _mock_goals[user_id].append(goal)

        telemetry_logger.log_info(f"Financial goal created: {goal_id}")

        return jsonify({
            'status': 'success',
            'message': 'Financial goal created successfully',
            'goal': goal,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_financial_goal'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/goals', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_financial_goals():
    """
    Get all financial goals for a user
    """
    try:
        user_id = request.args.get('user_id')
        goal_type = request.args.get('type')
        status_filter = request.args.get('status', 'active')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        goals = _mock_goals.get(user_id, [])

        # Apply filters
        if goal_type:
            goals = [g for g in goals if g['type'] == goal_type]
        if status_filter:
            goals = [g for g in goals if g['status'] == status_filter]

        # Calculate summary
        total_target = sum(g['target_amount'] for g in goals)
        total_current = sum(g['current_amount'] for g in goals)
        average_progress = sum(g['progress_percentage'] for g in goals) / len(goals) if goals else 0

        return jsonify({
            'status': 'success',
            'goals': goals,
            'summary': {
                'total_goals': len(goals),
                'total_target_amount': total_target,
                'total_current_amount': total_current,
                'average_progress': round(average_progress, 2)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_financial_goals'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/goals/<goal_id>/contribute', methods=['POST'])
@token_auth_required
@conditional_limit("15 per minute")
def contribute_to_goal(goal_id):
    """
    Add a contribution to a financial goal
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No contribution data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        amount = data.get('amount')
        source = data.get('source', 'manual')

        if not all([user_id, amount]):
            return jsonify({'error': 'User ID and amount are required', 'status': 'error'}), 400

        if amount <= 0:
            return jsonify({'error': 'Contribution amount must be positive', 'status': 'error'}), 400

        goals = _mock_goals.get(user_id, [])
        goal = next((g for g in goals if g['goal_id'] == goal_id), None)

        if not goal:
            return jsonify({'error': 'Goal not found', 'status': 'error'}), 404

        # Add contribution
        contribution = {
            'contribution_id': str(uuid.uuid4()),
            'amount': amount,
            'source': source,
            'date': datetime.now(timezone.utc).isoformat()
        }

        goal['contributions'].append(contribution)
        goal['current_amount'] += amount
        goal['progress_percentage'] = (goal['current_amount'] / goal['target_amount']) * 100
        goal['remaining_amount'] = goal['target_amount'] - goal['current_amount']

        # Check if goal is completed
        if goal['current_amount'] >= goal['target_amount']:
            goal['status'] = 'completed'
            goal['completed_at'] = datetime.now(timezone.utc).isoformat()

        telemetry_logger.log_info(f"Contribution added to goal {goal_id}: ${amount}")

        return jsonify({
            'status': 'success',
            'message': f'Contribution of ${amount} added successfully',
            'goal': goal,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'contribute_to_goal'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# SPENDING INSIGHTS AND ANALYTICS ENDPOINTS
# =============================================================================

@pfm_bp.route('/pfm/insights/spending', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_spending_insights():
    """
    Get personalized spending insights and trends
    """
    try:
        user_id = request.args.get('user_id')
        period = request.args.get('period', 'current_month')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        # Mock spending data and insights
        insights = {
            'user_id': user_id,
            'period': period,
            'total_spending': 2847.50,
            'transaction_count': 67,
            'avg_transaction': 42.50,
            'categories': {
                'groceries': {'amount': 450.25, 'percentage': 15.8, 'trend': 'stable', 'transactions': 12},
                'dining': {'amount': 380.75, 'percentage': 13.4, 'trend': 'increasing', 'transactions': 18},
                'entertainment': {'amount': 295.50, 'percentage': 10.4, 'trend': 'decreasing', 'transactions': 8},
                'utilities': {'amount': 180.00, 'percentage': 6.3, 'trend': 'stable', 'transactions': 3},
                'transportation': {'amount': 220.30, 'percentage': 7.7, 'trend': 'increasing', 'transactions': 15},
                'shopping': {'amount': 520.40, 'percentage': 18.3, 'trend': 'stable', 'transactions': 11}
            },
            'insights': [
                'Your dining expenses increased by 15% compared to last month',
                'Entertainment spending is trending down - great job staying within budget!',
                'Transportation costs are rising - consider carpooling or public transit',
                'You have 3 recurring subscriptions totaling $45/month'
            ],
            'recommendations': [
                'Consider meal planning to reduce dining out expenses',
                'Your grocery spending is within the recommended range',
                'Set up automatic savings transfers to build better habits'
            ],
            'comparison': {
                'vs_last_period': '+8.5%',
                'vs_budget': '95.2%',
                'vs_income_percentage': '68.3%'
            }
        }

        return jsonify({
            'status': 'success',
            'insights': insights,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_spending_insights'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/insights/trends', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_spending_trends():
    """
    Get spending trends over time
    """
    try:
        user_id = request.args.get('user_id')
        timeframe = request.args.get('timeframe', '6_months')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        # Mock trend data
        trends = {
            'user_id': user_id,
            'timeframe': timeframe,
            'monthly_spending': [
                {'month': '2023-08', 'amount': 2650.00, 'change': '-2.1%'},
                {'month': '2023-09', 'amount': 2720.00, 'change': '+2.6%'},
                {'month': '2023-10', 'amount': 2580.00, 'change': '-5.1%'},
                {'month': '2023-11', 'amount': 2890.00, 'change': '+12.0%'},
                {'month': '2023-12', 'amount': 3120.00, 'change': '+7.9%'},
                {'month': '2024-01', 'amount': 2847.50, 'change': '-8.8%'}
            ],
            'category_trends': {
                'groceries': {'trend': 'stable', 'avg_monthly': 445.00, 'change_6m': '+3.2%'},
                'dining': {'trend': 'increasing', 'avg_monthly': 365.00, 'change_6m': '+12.5%'},
                'entertainment': {'trend': 'decreasing', 'avg_monthly': 310.00, 'change_6m': '-8.7%'},
                'utilities': {'trend': 'stable', 'avg_monthly': 185.00, 'change_6m': '+1.8%'},
                'transportation': {'trend': 'increasing', 'avg_monthly': 215.00, 'change_6m': '+15.3%'}
            },
            'seasonal_patterns': {
                'highest_month': 'December',
                'lowest_month': 'October',
                'peak_categories': ['shopping', 'dining'],
                'seasonal_increase': '15-20% in Q4'
            },
            'predictions': {
                'next_month_estimate': 2950.00,
                'confidence': 0.78,
                'based_on': '6-month historical data'
            }
        }

        return jsonify({
            'status': 'success',
            'trends': trends,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_spending_trends'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# FINANCIAL HEALTH AND RECOMMENDATIONS ENDPOINTS
# =============================================================================

@pfm_bp.route('/pfm/health/score', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_financial_health_score():
    """
    Calculate and return financial health score
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        # Mock financial health calculation
        health_score = {
            'user_id': user_id,
            'overall_score': 78,
            'grade': 'B+',
            'components': {
                'savings_rate': {'score': 85, 'value': '12.5%', 'status': 'excellent'},
                'debt_to_income': {'score': 70, 'value': '0.35', 'status': 'good'},
                'emergency_fund': {'score': 90, 'value': '6 months', 'status': 'excellent'},
                'budget_adherence': {'score': 65, 'value': '78%', 'status': 'fair'},
                'credit_utilization': {'score': 75, 'value': '25%', 'status': 'good'},
                'investment_diversity': {'score': 80, 'value': 'moderate', 'status': 'good'}
            },
            'strengths': [
                'Strong emergency fund coverage',
                'Good savings rate',
                'Low debt-to-income ratio'
            ],
            'improvement_areas': [
                'Budget adherence could be improved',
                'Consider increasing retirement contributions',
                'Review recurring subscription costs'
            ],
            'recommendations': [
                'Increase retirement savings by 2% of income',
                'Set up automatic bill payments to avoid late fees',
                'Review and optimize insurance coverage',
                'Consider consolidating high-interest debt'
            ],
            'calculated_at': datetime.now(timezone.utc).isoformat()
        }

        # Store for tracking
        _mock_financial_health[user_id] = health_score

        return jsonify({
            'status': 'success',
            'health_score': health_score,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_financial_health_score'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/recommendations', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_personalized_recommendations():
    """
    Get personalized financial recommendations
    """
    try:
        user_id = request.args.get('user_id')
        category = request.args.get('category')  # savings, debt, budget, investment

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        # Mock personalized recommendations
        recommendations = {
            'user_id': user_id,
            'category': category or 'all',
            'recommendations': [
                {
                    'id': str(uuid.uuid4()),
                    'type': 'savings',
                    'title': 'Increase Emergency Fund',
                    'description': 'Your emergency fund covers 6 months of expenses. Consider increasing to 9-12 months.',
                    'priority': 'high',
                    'potential_impact': '$2,400 annual savings',
                    'timeframe': '3-6 months',
                    'action_items': [
                        'Set up automatic monthly transfers of $500',
                        'Cut discretionary spending by $200/month',
                        'Look for high-yield savings accounts'
                    ]
                },
                {
                    'id': str(uuid.uuid4()),
                    'type': 'budget',
                    'title': 'Optimize Dining Budget',
                    'description': 'Dining expenses increased 15% this month. Consider meal planning to reduce costs.',
                    'priority': 'medium',
                    'potential_impact': '$180 annual savings',
                    'timeframe': '1 month',
                    'action_items': [
                        'Plan meals for the week ahead',
                        'Use grocery delivery for convenience',
                        'Try restaurant loyalty programs for discounts'
                    ]
                },
                {
                    'id': str(uuid.uuid4()),
                    'type': 'debt',
                    'title': 'Credit Card Optimization',
                    'description': 'Your credit utilization is at 25%. Consider paying down balances to improve credit score.',
                    'priority': 'medium',
                    'potential_impact': 'Improve credit score by 20-30 points',
                    'timeframe': '2-3 months',
                    'action_items': [
                        'Set up bi-weekly payments',
                        'Consider balance transfer to 0% APR card',
                        'Avoid new credit applications for 6 months'
                    ]
                },
                {
                    'id': str(uuid.uuid4()),
                    'type': 'investment',
                    'title': 'Diversify Investment Portfolio',
                    'description': 'Your portfolio is moderately diversified. Consider adding international funds.',
                    'priority': 'low',
                    'potential_impact': 'Reduce portfolio volatility',
                    'timeframe': 'Ongoing',
                    'action_items': [
                        'Review current asset allocation',
                        'Consider target-date funds for simplicity',
                        'Consult with financial advisor'
                    ]
                }
            ],
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

        # Filter by category if specified
        if category and category != 'all':
            recommendations['recommendations'] = [
                r for r in recommendations['recommendations']
                if r['type'] == category
            ]

        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_personalized_recommendations'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# NOTIFICATIONS AND ALERTS ENDPOINTS
# =============================================================================

@pfm_bp.route('/pfm/notifications', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_notification():
    """
    Create a custom financial notification
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No notification data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        notification_type = data.get('type', 'budget_alert')  # budget_alert, goal_reminder, bill_due, unusual_spending
        title = data.get('title')
        message = data.get('message')
        trigger_condition = data.get('trigger_condition', {})
        channels = data.get('channels', ['email'])  # email, sms, push

        if not all([user_id, title, message]):
            return jsonify({'error': 'User ID, title, and message are required', 'status': 'error'}), 400

        notification_id = str(uuid.uuid4())

        notification = {
            'notification_id': notification_id,
            'user_id': user_id,
            'type': notification_type,
            'title': title,
            'message': message,
            'trigger_condition': trigger_condition,
            'channels': channels,
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_triggered': None
        }

        if user_id not in _mock_notifications:
            _mock_notifications[user_id] = []
        _mock_notifications[user_id].append(notification)

        telemetry_logger.log_info(f"Notification created: {notification_id}")

        return jsonify({
            'status': 'success',
            'message': 'Notification created successfully',
            'notification': notification,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_notification'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/notifications', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_notifications():
    """
    Get all notifications for a user
    """
    try:
        user_id = request.args.get('user_id')
        status_filter = request.args.get('status', 'active')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        notifications = _mock_notifications.get(user_id, [])

        # Apply filters
        if status_filter:
            notifications = [n for n in notifications if n['status'] == status_filter]

        return jsonify({
            'status': 'success',
            'notifications': notifications,
            'count': len(notifications),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_notifications'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/alerts/trigger', methods=['POST'])
@token_auth_required
@conditional_limit("20 per minute")
def trigger_alert():
    """
    Manually trigger a financial alert (for testing)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No alert data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        alert_type = data.get('alert_type', 'budget_warning')
        message = data.get('message', f'Alert triggered: {alert_type}')

        # Create alert notification
        alert = {
            'alert_id': str(uuid.uuid4()),
            'user_id': user_id,
            'type': alert_type,
            'message': message,
            'triggered_at': datetime.now(timezone.utc).isoformat(),
            'status': 'sent'
        }

        if user_id not in _mock_notifications:
            _mock_notifications[user_id] = []
        _mock_notifications[user_id].append(alert)

        telemetry_logger.log_info(f"Alert triggered for user {user_id}: {alert_type}")

        return jsonify({
            'status': 'success',
            'message': 'Alert triggered successfully',
            'alert': alert,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'trigger_alert'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# PHASE 2: ACCOUNT MANAGEMENT FEATURES - TRANSACTION CATEGORIZATION
# =============================================================================

@pfm_bp.route('/pfm/transactions/categorize', methods=['POST'])
@token_auth_required
@conditional_limit("30 per minute")
def categorize_transactions():
    """
    Categorize transactions based on description and merchant data
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No transaction data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        transactions = data.get('transactions', [])

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        if not transactions:
            return jsonify({'error': 'No transactions provided for categorization', 'status': 'error'}), 400

        categorized_transactions = []

        for transaction in transactions:
            description = transaction.get('description', '')
            amount = transaction.get('amount', 0)
            transaction_id = transaction.get('transaction_id', str(uuid.uuid4()))

            # Categorize the transaction
            category = categorize_transaction(description)

            # Create categorized transaction record
            categorized_txn = {
                'transaction_id': transaction_id,
                'user_id': user_id,
                'description': description,
                'amount': amount,
                'category': category,
                'categorized_at': datetime.now(timezone.utc).isoformat(),
                'confidence': 0.85  # Mock confidence score
            }

            categorized_transactions.append(categorized_txn)

            # Store in mock data
            if user_id not in _mock_transactions:
                _mock_transactions[user_id] = []
            _mock_transactions[user_id].append(categorized_txn)

        telemetry_logger.log_info(f"Categorized {len(categorized_transactions)} transactions for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': f'Successfully categorized {len(categorized_transactions)} transactions',
            'categorized_transactions': categorized_transactions,
            'summary': {
                'total_categorized': len(categorized_transactions),
                'categories_used': list(set(txn['category'] for txn in categorized_transactions)),
                'average_confidence': 0.85
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'categorize_transactions'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/transactions', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_transactions():
    """
    Get categorized transactions for a user
    """
    try:
        user_id = request.args.get('user_id')
        category = request.args.get('category')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        transactions = _mock_transactions.get(user_id, [])

        # Apply category filter
        if category:
            transactions = [t for t in transactions if t.get('category') == category]

        # Apply pagination
        total_count = len(transactions)
        transactions = transactions[offset:offset + limit]

        # Calculate category summary
        category_summary = {}
        for txn in _mock_transactions.get(user_id, []):
            cat = txn.get('category', 'uncategorized')
            if cat not in category_summary:
                category_summary[cat] = {'count': 0, 'total_amount': 0}
            category_summary[cat]['count'] += 1
            category_summary[cat]['total_amount'] += txn.get('amount', 0)

        return jsonify({
            'status': 'success',
            'transactions': transactions,
            'pagination': {
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            },
            'summary': {
                'total_transactions': total_count,
                'category_breakdown': category_summary
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_transactions'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# PHASE 2: ACCOUNT BALANCE MONITORING WITH ALERTS
# =============================================================================

@pfm_bp.route('/pfm/accounts/monitor', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def setup_balance_monitoring():
    """
    Set up balance monitoring and alerts for accounts
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No monitoring data provided', 'status': 'error'}), 400

        user_id = data.get('user_id')
        account_id = data.get('account_id')
        alert_thresholds = data.get('alert_thresholds', {})
        alert_channels = data.get('alert_channels', ['email'])

        if not all([user_id, account_id]):
            return jsonify({'error': 'User ID and account ID are required', 'status': 'error'}), 400

        # Verify account exists
        accounts = _mock_accounts.get(user_id, [])
        account = next((acc for acc in accounts if acc['account_id'] == account_id), None)

        if not account:
            return jsonify({'error': 'Account not found', 'status': 'error'}), 404

        # Create balance monitoring configuration
        monitor_config = {
            'monitor_id': str(uuid.uuid4()),
            'user_id': user_id,
            'account_id': account_id,
            'alert_thresholds': {
                'low_balance': alert_thresholds.get('low_balance', 100.00),
                'high_balance': alert_thresholds.get('high_balance', 10000.00),
                'unusual_spending': alert_thresholds.get('unusual_spending', 500.00)
            },
            'alert_channels': alert_channels,
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_checked': datetime.now(timezone.utc).isoformat()
        }

        if user_id not in _mock_balance_alerts:
            _mock_balance_alerts[user_id] = []
        _mock_balance_alerts[user_id].append(monitor_config)

        telemetry_logger.log_info(f"Balance monitoring set up for account {account_id}")

        return jsonify({
            'status': 'success',
            'message': 'Balance monitoring configured successfully',
            'monitor_config': monitor_config,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'setup_balance_monitoring'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/accounts/<account_id>/alerts', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_account_alerts(account_id):
    """
    Get balance alerts and monitoring status for an account
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        # Get account details
        accounts = _mock_accounts.get(user_id, [])
        account = next((acc for acc in accounts if acc['account_id'] == account_id), None)

        if not account:
            return jsonify({'error': 'Account not found', 'status': 'error'}), 404

        # Get monitoring configuration
        monitors = _mock_balance_alerts.get(user_id, [])
        monitor = next((m for m in monitors if m['account_id'] == account_id), None)

        # Check for potential alerts
        alerts = []
        if monitor:
            balance = account.get('balance', 0)
            thresholds = monitor.get('alert_thresholds', {})

            if balance <= thresholds.get('low_balance', 100):
                alerts.append({
                    'alert_id': str(uuid.uuid4()),
                    'type': 'low_balance',
                    'message': f'Account balance is low: ${balance:.2f}',
                    'severity': 'warning',
                    'triggered_at': datetime.now(timezone.utc).isoformat()
                })

            if balance >= thresholds.get('high_balance', 10000):
                alerts.append({
                    'alert_id': str(uuid.uuid4()),
                    'type': 'high_balance',
                    'message': f'Account balance is high: ${balance:.2f}',
                    'severity': 'info',
                    'triggered_at': datetime.now(timezone.utc).isoformat()
                })

        return jsonify({
            'status': 'success',
            'account_id': account_id,
            'current_balance': account.get('balance', 0),
            'monitoring_config': monitor,
            'active_alerts': alerts,
            'alert_count': len(alerts),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_account_alerts'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# PHASE 2: MULTI-INSTITUTION ACCOUNT AGGREGATION
# =============================================================================

@pfm_bp.route('/pfm/accounts/aggregate', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_account_aggregation():
    """
    Get aggregated view of accounts across multiple institutions
    """
    try:
        user_id = request.args.get('user_id')
        institution_filter = request.args.get('institution')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        accounts = _mock_accounts.get(user_id, [])

        # Filter by institution if specified
        if institution_filter:
            accounts = [acc for acc in accounts if acc.get('institution_id') == institution_filter]

        # Aggregate by institution
        institution_summary = {}
        for account in accounts:
            inst_id = account.get('institution_id', 'unknown')
            if inst_id not in institution_summary:
                institution_summary[inst_id] = {
                    'institution_id': inst_id,
                    'accounts': [],
                    'total_balance': 0,
                    'total_debt': 0,
                    'net_worth': 0,
                    'account_types': set()
                }

            institution_summary[inst_id]['accounts'].append(account)
            institution_summary[inst_id]['account_types'].add(account.get('account_type', 'unknown'))

            balance = account.get('balance', 0)
            if account.get('account_type') == 'credit_card' and balance < 0:
                institution_summary[inst_id]['total_debt'] += abs(balance)
            else:
                institution_summary[inst_id]['total_balance'] += balance

        # Calculate net worth per institution
        for inst in institution_summary.values():
            inst['net_worth'] = inst['total_balance'] - inst['total_debt']
            inst['account_types'] = list(inst['account_types'])

        # Overall aggregation
        total_balance = sum(inst['total_balance'] for inst in institution_summary.values())
        total_debt = sum(inst['total_debt'] for inst in institution_summary.values())
        total_net_worth = total_balance - total_debt

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'institution_breakdown': list(institution_summary.values()),
            'overall_aggregation': {
                'total_institutions': len(institution_summary),
                'total_accounts': len(accounts),
                'total_balance': total_balance,
                'total_debt': total_debt,
                'net_worth': total_net_worth,
                'currency': 'USD'
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_account_aggregation'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@pfm_bp.route('/pfm/transactions/insights', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_transaction_insights():
    """
    Get insights based on categorized transactions
    """
    try:
        user_id = request.args.get('user_id')
        period = request.args.get('period', '30_days')

        if not user_id:
            return jsonify({'error': 'User ID is required', 'status': 'error'}), 400

        transactions = _mock_transactions.get(user_id, [])

        # Calculate insights
        category_spending = {}
        total_spending = 0
        transaction_count = len(transactions)

        for txn in transactions:
            category = txn.get('category', 'uncategorized')
            amount = txn.get('amount', 0)

            if amount < 0:  # Only count expenses
                if category not in category_spending:
                    category_spending[category] = {'total': 0, 'count': 0, 'avg_transaction': 0}

                category_spending[category]['total'] += abs(amount)
                category_spending[category]['count'] += 1
                total_spending += abs(amount)

        # Calculate averages and percentages
        for cat_data in category_spending.values():
            cat_data['avg_transaction'] = cat_data['total'] / cat_data['count'] if cat_data['count'] > 0 else 0
            cat_data['percentage'] = (cat_data['total'] / total_spending * 100) if total_spending > 0 else 0

        # Generate insights
        insights = []
        if category_spending:
            top_category = max(category_spending.items(), key=lambda x: x[1]['total'])
            insights.append(f"Your highest spending category is {top_category[0]} with ${top_category[1]['total']:.2f}")

            # Check for unusual spending patterns
            avg_transaction = total_spending / transaction_count if transaction_count > 0 else 0
            if avg_transaction > 100:
                insights.append("Your average transaction amount is relatively high")

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'period': period,
            'insights': insights,
            'spending_analysis': {
                'total_spending': total_spending,
                'transaction_count': transaction_count,
                'avg_transaction': total_spending / transaction_count if transaction_count > 0 else 0,
                'category_breakdown': category_spending
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_transaction_insights'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
