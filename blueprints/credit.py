"""
Credit Card Blueprint for JPMorgan Financial APIs
Provides endpoints for credit card management.
"""

import secrets
from flask import Blueprint, request, jsonify, g
from datetime import datetime, timezone
from typing import Dict, Any

try:
    from src.auth import token_auth_required
except ImportError:
    def token_auth_required(f):
        return f

try:
    from src.logger import telemetry_logger
except ImportError:
    class FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg}")
    telemetry_logger = FallbackLogger()


# Create blueprint
credit_bp = Blueprint('credit', __name__)


# =============================================================================
# IN-MEMORY STORE (Replace with database in production)
# =============================================================================

class CreditCardStore:
    """In-memory storage for credit cards"""
    
    def __init__(self):
        self.cards = {}
        self.transactions = {}
    
    def create_card(self, card: Dict[str, Any]) -> Dict[str, Any]:
        card_id = card['card_number']
        self.cards[card_id] = card
        return card
    
    def get_card(self, card_id: str) -> Dict[str, Any]:
        return self.cards.get(card_id)
    
    def get_cards_by_user(self, user_id: str) -> list:
        return [c for c in self.cards.values() if c.get('user_id') == user_id]
    
    def update_card(self, card_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if card_id in self.cards:
            self.cards[card_id].update(updates)
            return self.cards[card_id]
        return None
    
    def delete_card(self, card_id: str) -> bool:
        if card_id in self.cards:
            del self.cards[card_id]
            return True
        return False
    
    def create_transaction(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        txn_id = txn['transaction_id']
        self.transactions[txn_id] = txn
        return txn
    
    def get_transaction(self, txn_id: str) -> Dict[str, Any]:
        return self.transactions.get(txn_id)
    
    def get_transactions_by_card(self, card_id: str) -> list:
        return [t for t in self.transactions.values() if t.get('card_id') == card_id]


credit_store = CreditCardStore()


# =============================================================================
# CREDIT CARD ENDPOINTS
# =============================================================================

@credit_bp.route('/cards', methods=['POST'])
@token_auth_required
def create_card():
    """
    Apply for a new credit card
    """
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        required = ['card_type', 'credit_limit']
        for field in required:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
        
        card_number = secrets.token_hex(8).upper()
        
        card = {
            'card_number': card_number,
            'user_id': user_id,
            'card_type': data['card_type'],
            'card_brand': data.get('card_brand', data['card_type'].upper()),
            'expiry_month': data.get('expiry_month', datetime.now().month),
            'expiry_year': data.get('expiry_year', datetime.now().year + 3),
            'cvv': secrets.token_hex(2),
            'status': 'application',
            'credit_limit': float(data['credit_limit']),
            'available_credit': float(data['credit_limit']),
            'current_balance': 0,
            'interest_rate': float(data.get('interest_rate', 19.99)),
            'annual_fee': float(data.get('annual_fee', 0)),
            'reward_points': 0,
            'cash_back_balance': 0,
            'cardholder_name': data.get('cardholder_name', ''),
            'issue_date': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        credit_store.create_card(card)
        
        telemetry_logger.log_info(f"Credit card created: {card_number[-4:]}", {'context': 'credit'})
        
        return jsonify({'status': 'success', 'card': card}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_card'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/cards', methods=['GET'])
@token_auth_required
def list_cards():
    """List all credit cards for the user"""
    try:
        user_id = g.get('user_id', 'test_user')
        cards = credit_store.get_cards_by_user(user_id)
        
        for card in cards:
            card['card_number'] = card['card_number'][-4:].rjust(len(card['card_number']), '*')
        
        return jsonify({'status': 'success', 'cards': cards, 'count': len(cards)}), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_cards'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/cards/<card_number>', methods=['GET'])
@token_auth_required
def get_card(card_number):
    """Get credit card details"""
    try:
        card = credit_store.get_card(card_number)
        
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'}), 404
        
        transactions = credit_store.get_transactions_by_card(card_number)
        
        card_display = card.copy()
        card_display['card_number'] = card['card_number'][-4:].rjust(len(card['card_number']), '*')
        
        return jsonify({'status': 'success', 'card': card_display, 'transactions': transactions}), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_card'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/cards/<card_number>', methods=['PUT'])
@token_auth_required
def update_card(card_number):
    """Update credit card (activate, block, etc.)"""
    try:
        data = request.get_json()
        
        card = credit_store.get_card(card_number)
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'}), 404
        
        allowed_fields = ['status', 'credit_limit']
        updates = {k: v for k, v in data.items() if k in allowed_fields}
        
        if 'status' in updates:
            if updates['status'] == 'active' and card['status'] == 'application':
                updates['activation_date'] = datetime.now(timezone.utc).isoformat()
        
        credit_store.update_card(card_number, updates)
        
        telemetry_logger.log_info(f"Credit card updated: {card_number[-4:]}", {'context': 'credit'})
        
        card = credit_store.get_card(card_number)
        card['card_number'] = card['card_number'][-4:].rjust(len(card['card_number']), '*')
        
        return jsonify({'status': 'success', 'card': card}), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'update_card'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/cards/<card_number>/transactions', methods=['POST'])
@token_auth_required
def create_transaction(card_number):
    """Create a credit card transaction (charge)"""
    try:
        data = request.get_json()
        
        card = credit_store.get_card(card_number)
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'}), 404
        
        if card['status'] != 'active':
            return jsonify({'status': 'error', 'message': 'Card is not active'}), 400
        
        amount = float(data.get('amount', 0))
        if amount <= 0:
            return jsonify({'status': 'error', 'message': 'Invalid amount'}), 400
        
        new_balance = card['current_balance'] + amount
        if new_balance > card['credit_limit']:
            return jsonify({'status': 'error', 'message': 'Credit limit exceeded'}), 400
        
        transaction_id = f"CC-{secrets.token_hex(4).upper()}"
        
        transaction = {
            'transaction_id': transaction_id,
            'card_id': card_number,
            'user_id': card['user_id'],
            'amount': amount,
            'description': data.get('description', ''),
            'merchant_name': data.get('merchant_name', ''),
            'merchant_category': data.get('merchant_category', ''),
            'transaction_date': datetime.now(timezone.utc).isoformat(),
            'status': 'pending',
            'is_credit': False,
            'reward_points_earned': int(amount),
            'cash_back_amount': amount * 0.01 if amount >= 10 else 0,
            'billing_cycle': datetime.now().month,
            'is_posted': False,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        credit_store.update_card(card_number, {
            'current_balance': new_balance,
            'available_credit': card['credit_limit'] - new_balance,
            'reward_points': card['reward_points'] + transaction['reward_points_earned'],
            'cash_back_balance': card['cash_back_balance'] + transaction['cash_back_amount']
        })
        
        credit_store.create_transaction(transaction)
        
        telemetry_logger.log_info(f"Credit card transaction: {transaction_id}", {'context': 'credit'})
        
        return jsonify({'status': 'success', 'transaction': transaction}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_transaction'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/cards/<card_number>/transactions', methods=['GET'])
@token_auth_required
def list_transactions(card_number):
    """List all transactions for a credit card"""
    try:
        card = credit_store.get_card(card_number)
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'}), 404
        
        transactions = credit_store.get_transactions_by_card(card_number)
        
        return jsonify({'status': 'success', 'transactions': transactions, 'count': len(transactions)}), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_transactions'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/cards/<card_number>/payment', methods=['POST'])
@token_auth_required
def make_payment(card_number):
    """Make a credit card payment"""
    try:
        data = request.get_json()
        
        card = credit_store.get_card(card_number)
        if not card:
            return jsonify({'status': 'error', 'message': 'Card not found'}), 404
        
        amount = float(data.get('amount', 0))
        if amount <= 0:
            return jsonify({'status': 'error', 'message': 'Invalid amount'}), 400
        
        transaction_id = f"CCP-{secrets.token_hex(4).upper()}"
        
        transaction = {
            'transaction_id': transaction_id,
            'card_id': card_number,
            'user_id': card['user_id'],
            'amount': amount,
            'description': 'Payment',
            'merchant_name': 'Payment',
            'transaction_date': datetime.now(timezone.utc).isoformat(),
            'status': 'completed',
            'is_credit': True,
            'reward_points_earned': 0,
            'cash_back_amount': 0,
            'billing_cycle': datetime.now().month,
            'is_posted': True,
            'posting_date': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        new_balance = max(0, card['current_balance'] - amount)
        credit_store.update_card(card_number, {
            'current_balance': new_balance,
            'available_credit': card['credit_limit'] - new_balance
        })
        
        credit_store.create_transaction(transaction)
        
        telemetry_logger.log_info(f"Credit card payment: {transaction_id}", {'context': 'credit'})
        
        return jsonify({'status': 'success', 'transaction': transaction}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'make_payment'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@credit_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for credit service"""
    return jsonify({
        'status': 'healthy',
        'service': 'credit',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


__all__ = ['credit_bp']
