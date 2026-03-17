"""
Financial Data Blueprint for JPMorgan Dashboard
Provides JPMorgan-scale financial data for dashboard endpoints called by dashboard.js
Realistic mock data scaled to JPMorgan levels ($3.2T assets, $125B revenue etc.)
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone, timedelta
import random
import uuid

try:
    from src.logger import telemetry_logger
except ImportError:
    class MockLogger:
        def log_info(self, *args): pass
        def log_error(self, *args): pass
    telemetry_logger = MockLogger()

try:
    from src.auth import token_auth_required
except ImportError:
    def token_auth_required(f): return f

try:
    from src.rate_limiting import conditional_limit
except ImportError:
    def conditional_limit(limit): 
        def decorator(f): return f
        return decorator

financial_bp = Blueprint('financial', __name__, url_prefix='/financial')

@financial_bp.route('/summary', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def financial_summary():
    """Summary data for dashboard top cards - total balance, accounts, transactions"""
    try:
        # JPMorgan-scale enterprise data
        summary = {
            'totalBalance': 3200000000000.0,  # $3.2T total assets
            'currency': 'USD',
            'accountsCount': 12500000,  # Millions of accounts
            'recentTransactionsCount': 45000000,  # 45M transactions last 30 days
            'revenue': 125000000000,  # $125B annual revenue
            'netIncome': 48000000000,  # $48B net income
            'marketCap': 450000000000,  # $450B market cap
            'recentTransactions': [
                {
                    'id': str(uuid.uuid4()),
                    'categorized_at': (datetime.now(timezone.utc) - timedelta(days=random.randint(1,30))).isoformat(),
                    'description': f'Transaction {random.randint(1,1000000)}',
                    'amount': random.uniform(-1000000, 5000000000),
                    'currency': 'USD',
                    'category': random.choice(['income', 'payment', 'transfer', 'investment'])
                } for _ in range(10)
            ]
        }
        return jsonify(summary), 200
    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'financial_summary'})
        return jsonify({'error': 'Internal server error'}), 500

@financial_bp.route('/assets', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def financial_assets():
    """Assets breakdown table data"""
    assets_by_account = [
        {
            'accountName': 'Corporate Banking Division',
            'accountType': 'Loans & Deposits',
            'balance': 1100000000000,
            'currency': 'USD'
        },
        {
            'accountName': 'Investment Banking',
            'accountType': 'Trading Assets',
            'balance': 450000000000,
            'currency': 'USD'
        },
        {
            'accountName': 'Wealth Management',
            'accountType': 'Client Assets',
            'balance': 2800000000000,
            'currency': 'USD'
        },
        {
            'accountName': 'Cash & Equivalents',
            'accountType': 'Liquidity',
            'balance': 850000000000,
            'currency': 'USD'
        },
        {
            'accountName': 'Securities Portfolio',
            'accountType': 'Investments',
            'balance': 650000000000,
            'currency': 'USD'
        }
    ]
    return jsonify({'assetsByAccount': assets_by_account}), 200

@financial_bp.route('/stocks', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def financial_stocks():
    """Stock holdings table data"""
    stocks = [
        {
            'accountId': 'JPM-Investments',
            'accountName': 'JPMorgan Treasury Portfolio',
            'totalValue': 125000000000,
            'currency': 'USD'
        },
        {
            'accountId': 'Client-Wealth',
            'accountName': 'Wealth Management Holdings',
            'totalValue': 95000000000,
            'currency': 'USD'
        },
        {
            'accountId': 'Trading-Desk',
            'accountName': 'Proprietary Trading',
            'totalValue': 75000000000,
            'currency': 'USD'
        }
    ]
    return jsonify({'stocks': stocks}), 200

@financial_bp.route('/performance', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def financial_performance():
    """Performance trends chart data"""
    trends = [
        {'period': '2024-Q1', 'balance': 3100000000000},
        {'period': '2024-Q2', 'balance': 3150000000000},
        {'period': '2024-Q3', 'balance': 3200000000000},
        {'period': '2024-Q4', 'balance': 3250000000000},
        {'period': '2025-Q1', 'balance': 3300000000000}
    ]
    return jsonify({'trends': trends}), 200

