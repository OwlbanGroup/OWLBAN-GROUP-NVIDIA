#!/usr/bin/env python3
"""Banking Blueprint for personal bank account management and transactions."""

from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
from src.banking_service import banking_service
from src.auth import token_auth_required
from src.rate_limiting import conditional_limit
from src.logger import telemetry_logger
import uuid

banking_bp = Blueprint('banking', __name__)

@banking_bp.route('/accounts', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def list_accounts():
 """List user's bank accounts."""
    user_id = getattr(request, 'user_id', 'demo_user')
    try:
        accounts = banking_service.get_accounts(user_id)
        return jsonify({
            'status': 'success',
            'accounts': [acc.to_dict() for acc in accounts],
            'count': len(accounts)
        })
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

@banking_bp.route('/accounts', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def create_account():
 """Create new bank account."""
    user_id = getattr(request, 'user_id', 'demo_user')
    data = request.get_json() or {}
    data['user_id'] = user_id
    data.setdefault('account_type', 'checking')
    data.setdefault('initial_balance', 0.0)
    try:
        account = banking_service.create_account(data)
        return jsonify({
            'status': 'success',
            'account': account.to_dict(),
            'message': 'Account created successfully'
        }), 201
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

@banking_bp.route('/accounts/<int:account_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_account(account_id: int):
    \"\"\"Get specific account details.\"\"\"
    user_id = getattr(request, 'user_id', 'demo_user')
    try:
        account = banking_service.get_account(account_id, user_id)
        if not account:
            return jsonify({'error': 'Account not found', 'status': 'error'}), 404
        return jsonify({'status': 'success', 'account': account.to_dict()})
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

@banking_bp.route('/accounts/<int:account_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("5 per minute")
def update_account(account_id: int):
 """Update account details (interest_rate, overdraft_limit, etc.)."""
    user_id = getattr(request, 'user_id', 'demo_user')
    data = request.get_json() or {}
    try:
        account = banking_service.update_account(account_id, user_id, data)
        if not account:
            return jsonify({'error': 'Account not found', 'status': 'error'}), 404
        return jsonify({'status': 'success', 'account': account.to_dict()})
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

@banking_bp.route('/accounts/<int:account_id>/validate', methods=['POST'])
@token_auth_required
@conditional_limit("20 per minute")
def validate_account(account_id: int):
 """Validate account status and balance for transactions."""
    user_id = getattr(request, 'user_id', 'demo_user')
    data = request.get_json() or {}
    min_balance = data.get('min_balance', 0.0)
    try:
        validation = banking_service.validate_account(account_id, user_id, min_balance)
        return jsonify({'status': 'success', 'validation': validation})
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

@banking_bp.route('/accounts/<int:account_id>/transactions', methods=['GET'])
@token_auth_required
@conditional_limit("30 per minute")
def list_transactions(account_id: int):
    \"\"\"List account transactions.\"\"\"
    user_id = getattr(request, 'user_id', 'demo_user')
    limit = min(int(request.args.get('limit', 50)), 100)
    try:
        txs = banking_service.get_account_transactions(account_id, user_id, limit)
        return jsonify({
            'status': 'success',
            'transactions': [tx.to_dict() for tx in txs],
            'count': len(txs)
        })
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

@banking_bp.route('/accounts/<int:account_id>/transactions', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_transaction(account_id: int):
 """Create deposit/withdrawal/transfer transaction."""
    user_id = getattr(request, 'user_id', 'demo_user')
    data = request.get_json() or {}
    tx_type = data.get('type', 'deposit')  # deposit|withdrawal|transfer
    amount = float(data.get('amount', 0))
    description = data.get('description', '')
    if amount <= 0:
        return jsonify({'error': 'Amount must be positive', 'status': 'error'}), 400
    try:
        tx = banking_service.create_transaction(account_id, user_id, tx_type, amount, description)
        return jsonify({
            'status': 'success',
            'transaction': tx.to_dict(),
            'message': f'{tx_type.title()} processed successfully'
        }), 201
    except Exception as e:
        telemetry_logger.log_error(e)
        return jsonify({'error': str(e), 'status': 'error'}), 400

