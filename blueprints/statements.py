"""
Statements Blueprint for JPMorgan Financial APIs
Provides endpoints for account statements.
"""

import secrets
import json
from flask import Blueprint, request, jsonify, g
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from calendar import monthrange

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
statements_bp = Blueprint('statements', __name__)


# =============================================================================
# IN-MEMORY STORE (Replace with database in production)
# =============================================================================

class StatementStore:
    """In-memory storage for statements"""
    
    def __init__(self):
        self.statements = {}
    
    def create_statement(self, statement: Dict[str, Any]) -> Dict[str, Any]:
        statement_id = statement['statement_id']
        self.statements[statement_id] = statement
        return statement
    
    def get_statement(self, statement_id: str) -> Dict[str, Any]:
        return self.statements.get(statement_id)
    
    def get_statements_by_user(self, user_id: str) -> list:
        return [s for s in self.statements.values() if s.get('user_id') == user_id]
    
    def get_statements_by_account(self, account_id: str) -> list:
        return [s for s in self.statements.values() if s.get('account_id') == account_id]
    
    def update_statement(self, statement_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if statement_id in self.statements:
            self.statements[statement_id].update(updates)
            return self.statements[statement_id]
        return None


statement_store = StatementStore()


# Mock transaction store for generating statements
class MockTransactionStore:
    """Mock store for generating sample transactions"""
    
    @staticmethod
    def get_transactions_for_period(user_id: str, start_date: datetime, end_date: datetime) -> list:
        """Generate mock transactions for a period"""
        import random
        
        transactions = []
        categories = ['Food', 'Shopping', 'Transportation', 'Utilities', 'Entertainment', 'Healthcare', 'Travel']
        merchants = ['Amazon', 'Walmart', 'Target', 'Costco', 'Starbucks', 'Shell', 'Netflix', 'Spotify', 'Uber', 'Lyft']
        
        current = start_date
        while current <= end_date:
            # Generate 0-5 transactions per day
            num_transactions = random.randint(0, 5)
            for _ in range(num_transactions):
                amount = round(random.uniform(5, 500), 2)
                transactions.append({
                    'transaction_id': f"TXN-{secrets.token_hex(4).upper()}",
                    'date': current.isoformat(),
                    'description': f"{random.choice(merchants)} purchase",
                    'category': random.choice(categories),
                    'amount': -amount if random.random() > 0.3 else amount,
                    'type': 'debit' if amount > 0 else 'credit'
                })
            current += timedelta(days=1)
        
        return transactions


mock_transaction_store = MockTransactionStore()


# =============================================================================
# STATEMENT ENDPOINTS
# =============================================================================

@statements_bp.route('/statements', methods=['POST'])
@token_auth_required
def create_statement():
    """
    Generate a new statement
    """
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        required = ['account_id', 'statement_type', 'period_start', 'period_end']
        for field in required:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
        
        statement_id = f"STM-{secrets.token_hex(4).upper()}"
        
        # Parse dates
        period_start = datetime.fromisoformat(data['period_start'].replace('Z', '+00:00'))
        period_end = datetime.fromisoformat(data['period_end'].replace('Z', '+00:00'))
        
        # Get transactions for the period (mock for now)
        transactions = mock_transaction_store.get_transactions_for_period(user_id, period_start, period_end)
        
        # Calculate totals
        total_credits = sum(t['amount'] for t in transactions if t['amount'] > 0)
        total_debits = sum(abs(t['amount']) for t in transactions if t['amount'] < 0)
        
        # Calculate opening and closing balance (mock)
        opening_balance = round(10000 + total_credits - total_debits, 2)
        closing_balance = round(10000, 2)
        
        statement = {
            'statement_id': statement_id,
            'user_id': user_id,
            'account_id': data['account_id'],
            'statement_type': data['statement_type'],  # monthly, quarterly, annual
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'opening_balance': opening_balance,
            'closing_balance': closing_balance,
            'total_credits': round(total_credits, 2),
            'total_debits': round(total_debits, 2),
            'total_fees': 0,
            'total_interest': 0,
            'transaction_count': len(transactions),
            'transactions': transactions[:100],  # Limit to 100 transactions
            'status': 'generated',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        statement_store.create_statement(statement)
        
        telemetry_logger.log_info(f"Statement generated: {statement_id}", {'context': 'statements'})
        
        return jsonify({'status': 'success', 'statement': statement}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_statement'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@statements_bp.route('/statements', methods=['GET'])
@token_auth_required
def list_statements():
    """List all statements for the user"""
    try:
        user_id = g.get('user_id', 'test_user')
        account_id = request.args.get('account_id')
        
        if account_id:
            statements = statement_store.get_statements_by_account(account_id)
        else:
            statements = statement_store.get_statements_by_user(user_id)
        
        return jsonify({
            'status': 'success',
            'statements': statements,
            'count': len(statements)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_statements'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@statements_bp.route('/statements/<statement_id>', methods=['GET'])
@token_auth_required
def get_statement(statement_id):
    """Get statement details"""
    try:
        statement = statement_store.get_statement(statement_id)
        
        if not statement:
            return jsonify({'status': 'error', 'message': 'Statement not found'}), 404
        
        return jsonify({
            'status': 'success',
            'statement': statement
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_statement'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@statements_bp.route('/statements/<statement_id>/download', methods=['GET'])
@token_auth_required
def download_statement(statement_id):
    """Download statement in specified format"""
    try:
        statement = statement_store.get_statement(statement_id)
        
        if not statement:
            return jsonify({'status': 'error', 'message': 'Statement not found'}), 404
        
        format_type = request.args.get('format', 'json').lower()
        
        if format_type == 'json':
            return jsonify({
                'status': 'success',
                'statement': statement
            }), 200
        elif format_type == 'csv':
            # Generate CSV
            csv_lines = ['Date,Description,Category,Amount,Type']
            for txn in statement.get('transactions', []):
                csv_lines.append(f"{txn['date']},{txn['description']},{txn['category']},{txn['amount']},{txn['type']}")
            
            csv_data = '\n'.join(csv_lines)
            return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': f'attachment; filename={statement_id}.csv'}
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported format'}), 400
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'download_statement'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@statements_bp.route('/statements/monthly', methods=['POST'])
@token_auth_required
def generate_monthly_statement():
    """Generate monthly statement for current month"""
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        account_id = data.get('account_id', 'ACC-001')
        
        # Get current month
        now = datetime.now(timezone.utc)
        _, days_in_month = monthrange(now.year, now.month)
        
        period_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        period_end = datetime(now.year, now.month, days_in_month, 23, 59, 59, tzinfo=timezone.utc)
        
        # Create statement
        data = {
            'account_id': account_id,
            'statement_type': 'monthly',
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat()
        }
        
        # Reuse create_statement logic
        statement_id = f"STM-{secrets.token_hex(4).upper()}"
        
        transactions = mock_transaction_store.get_transactions_for_period(user_id, period_start, period_end)
        
        total_credits = sum(t['amount'] for t in transactions if t['amount'] > 0)
        total_debits = sum(abs(t['amount']) for t in transactions if t['amount'] < 0)
        
        opening_balance = round(10000 + total_credits - total_debits, 2)
        closing_balance = round(10000, 2)
        
        statement = {
            'statement_id': statement_id,
            'user_id': user_id,
            'account_id': account_id,
            'statement_type': 'monthly',
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'opening_balance': opening_balance,
            'closing_balance': closing_balance,
            'total_credits': round(total_credits, 2),
            'total_debits': round(total_debits, 2),
            'total_fees': 0,
            'total_interest': 0,
            'transaction_count': len(transactions),
            'transactions': transactions[:100],
            'status': 'generated',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        statement_store.create_statement(statement)
        
        telemetry_logger.log_info(f"Monthly statement generated: {statement_id}", {'context': 'statements'})
        
        return jsonify({'status': 'success', 'statement': statement}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'generate_monthly_statement'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@statements_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for statements service"""
    return jsonify({
        'status': 'healthy',
        'service': 'statements',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


__all__ = ['statements_bp']
