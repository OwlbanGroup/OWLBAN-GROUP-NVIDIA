"""
Loans Blueprint for JPMorgan Financial APIs
Provides endpoints for loan management.
"""

import secrets
from flask import Blueprint, request, jsonify, g
from datetime import datetime, timezone, timedelta
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
loans_bp = Blueprint('loans', __name__)


# =============================================================================
# IN-MEMORY STORE (Replace with database in production)
# =============================================================================

class LoanStore:
    """In-memory storage for loans"""
    
    def __init__(self):
        self.loans = {}
        self.payments = {}
    
    def create_loan(self, loan: Dict[str, Any]) -> Dict[str, Any]:
        loan_id = loan['loan_number']
        self.loans[loan_id] = loan
        return loan
    
    def get_loan(self, loan_id: str) -> Dict[str, Any]:
        return self.loans.get(loan_id)
    
    def get_loans_by_user(self, user_id: str) -> list:
        return [l for l in self.loans.values() if l.get('user_id') == user_id]
    
    def update_loan(self, loan_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if loan_id in self.loans:
            self.loans[loan_id].update(updates)
            return self.loans[loan_id]
        return None
    
    def delete_loan(self, loan_id: str) -> bool:
        if loan_id in self.loans:
            del self.loans[loan_id]
            return True
        return False
    
    def create_payment(self, payment: Dict[str, Any]) -> Dict[str, Any]:
        payment_id = payment['payment_id']
        self.payments[payment_id] = payment
        return payment
    
    def get_payment(self, payment_id: str) -> Dict[str, Any]:
        return self.payments.get(payment_id)
    
    def get_payments_by_loan(self, loan_id: str) -> list:
        return [p for p in self.payments.values() if p.get('loan_id') == loan_id]


loan_store = LoanStore()


# =============================================================================
# LOAN ENDPOINTS
# =============================================================================

@loans_bp.route('/loans', methods=['POST'])
@token_auth_required
def create_loan():
    """
    Create a new loan application
    """
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        # Validate required fields
        required = ['loan_type', 'principal_amount', 'interest_rate', 'term_months']
        for field in required:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
        
        loan_number = f"LN-{secrets.token_hex(4).upper()}"
        
        loan = {
            'loan_number': loan_number,
            'user_id': user_id,
            'loan_type': data['loan_type'],
            'principal_amount': float(data['principal_amount']),
            'interest_rate': float(data['interest_rate']),
            'term_months': int(data['term_months']),
            'status': 'application',
            'monthly_payment': 0,
            'total_interest': 0,
            'total_amount': 0,
            'remaining_balance': float(data['principal_amount']),
            'collateral_description': data.get('collateral_description'),
            'collateral_value': float(data.get('collateral_value', 0)),
            'application_date': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Calculate monthly payment
        principal = loan['principal_amount']
        rate = loan['interest_rate'] / 100 / 12
        months = loan['term_months']
        
        if rate > 0:
            monthly_payment = principal * (rate * (1 + rate) ** months) / ((1 + rate) ** months - 1)
        else:
            monthly_payment = principal / months
        
        loan['monthly_payment'] = round(monthly_payment, 2)
        loan['total_amount'] = round(monthly_payment * months, 2)
        loan['total_interest'] = round(loan['total_amount'] - principal, 2)
        
        # Calculate next payment date
        next_payment = datetime.now(timezone.utc) + timedelta(days=30)
        loan['next_payment_date'] = next_payment.isoformat()
        loan['next_payment_amount'] = loan['monthly_payment']
        
        loan_store.create_loan(loan)
        
        telemetry_logger.log_info(f"Loan created: {loan_number}", {'context': 'loans'})
        
        return jsonify({'status': 'success', 'loan': loan}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_loan'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@loans_bp.route('/loans', methods=['GET'])
@token_auth_required
def list_loans():
    """
    List all loans for the user
    """
    try:
        user_id = g.get('user_id', 'test_user')
        loans = loan_store.get_loans_by_user(user_id)
        
        return jsonify({
            'status': 'success',
            'loans': loans,
            'count': len(loans)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_loans'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@loans_bp.route('/loans/<loan_number>', methods=['GET'])
@token_auth_required
def get_loan(loan_number):
    """
    Get loan details
    """
    try:
        loan = loan_store.get_loan(loan_number)
        
        if not loan:
            return jsonify({'status': 'error', 'message': 'Loan not found'}), 404
        
        # Get payments for this loan
        payments = loan_store.get_payments_by_loan(loan_number)
        
        return jsonify({
            'status': 'success',
            'loan': loan,
            'payments': payments
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_loan'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@loans_bp.route('/loans/<loan_number>', methods=['PUT'])
@token_auth_required
def update_loan(loan_number):
    """
    Update loan details (approve, reject, etc.)
    """
    try:
        data = request.get_json()
        
        loan = loan_store.get_loan(loan_number)
        if not loan:
            return jsonify({'status': 'error', 'message': 'Loan not found'}), 404
        
        # Update allowed fields
        allowed_fields = ['status', 'approval_date', 'disbursement_date', 'maturity_date']
        updates = {k: v for k, v in data.items() if k in allowed_fields}
        
        if 'status' in updates:
            if updates['status'] == 'approved':
                updates['approval_date'] = datetime.now(timezone.utc).isoformat()
            elif updates['status'] == 'active':
                updates['disbursement_date'] = datetime.now(timezone.utc).isoformat()
        
        loan_store.update_loan(loan_number, updates)
        
        telemetry_logger.log_info(f"Loan updated: {loan_number}", {'context': 'loans'})
        
        return jsonify({
            'status': 'success',
            'loan': loan_store.get_loan(loan_number)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'update_loan'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@loans_bp.route('/loans/<loan_number>/payments', methods=['POST'])
@token_auth_required
def make_payment(loan_number):
    """
    Make a loan payment
    """
    try:
        data = request.get_json()
        
        loan = loan_store.get_loan(loan_number)
        if not loan:
            return jsonify({'status': 'error', 'message': 'Loan not found'}), 404
        
        amount = float(data.get('amount', 0))
        if amount <= 0:
            return jsonify({'status': 'error', 'message': 'Invalid payment amount'}), 400
        
        payment_id = f"LP-{secrets.token_hex(4).upper()}"
        
        payment = {
            'payment_id': payment_id,
            'loan_id': loan_number,
            'payment_number': len(loan_store.get_payments_by_loan(loan_number)) + 1,
            'payment_date': datetime.now(timezone.utc).isoformat(),
            'amount': amount,
            'principal_amount': amount,
            'interest_amount': 0,
            'balance_after': loan['remaining_balance'] - amount,
            'status': 'completed',
            'payment_method': data.get('payment_method', 'direct_debit'),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Update loan balance
        new_balance = loan['remaining_balance'] - amount
        loan_store.update_loan(loan_number, {
            'remaining_balance': max(0, new_balance)
        })
        
        # Check if loan is paid off
        if new_balance <= 0:
            loan_store.update_loan(loan_number, {
                'status': 'paid_off',
                'closed_date': datetime.now(timezone.utc).isoformat()
            })
        
        loan_store.create_payment(payment)
        
        telemetry_logger.log_info(f"Loan payment made: {payment_id}", {'context': 'loans'})
        
        return jsonify({'status': 'success', 'payment': payment}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'make_payment'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@loans_bp.route('/loans/<loan_number>/payments', methods=['GET'])
@token_auth_required
def list_payments(loan_number):
    """
    List all payments for a loan
    """
    try:
        loan = loan_store.get_loan(loan_number)
        if not loan:
            return jsonify({'status': 'error', 'message': 'Loan not found'}), 404
        
        payments = loan_store.get_payments_by_loan(loan_number)
        
        return jsonify({
            'status': 'success',
            'payments': payments,
            'count': len(payments)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_payments'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@loans_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for loans service"""
    return jsonify({
        'status': 'healthy',
        'service': 'loans',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


__all__ = ['loans_bp']
