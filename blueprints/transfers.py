"""
Transfers Blueprint for JPMorgan Financial APIs
Provides endpoints for wire/ACH transfers.
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
transfers_bp = Blueprint('transfers', __name__)


# =============================================================================
# IN-MEMORY STORE (Replace with database in production)
# =============================================================================

class TransferStore:
    """In-memory storage for transfers"""
    
    def __init__(self):
        self.transfers = {}
    
    def create_transfer(self, transfer: Dict[str, Any]) -> Dict[str, Any]:
        transfer_id = transfer['transfer_id']
        self.transfers[transfer_id] = transfer
        return transfer
    
    def get_transfer(self, transfer_id: str) -> Dict[str, Any]:
        return self.transfers.get(transfer_id)
    
    def get_transfers_by_user(self, user_id: str) -> list:
        return [t for t in self.transfers.values() if t.get('user_id') == user_id]
    
    def update_transfer(self, transfer_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if transfer_id in self.transfers:
            self.transfers[transfer_id].update(updates)
            return self.transfers[transfer_id]
        return None


transfer_store = TransferStore()


# =============================================================================
# WIRE TRANSFER FEES
# =============================================================================

WIRE_FEES = {
    'internal': 0,
    'ach': 0,
    'domestic_wire': 25,
    'international_wire': 50,
    'rtp': 0
}

TRANSFER_LIMITS = {
    'ach': {'min': 0.01, 'max': 100000, 'daily': 250000},
    'wire': {'min': 1, 'max': 1000000, 'daily': 5000000},
    'rtp': {'min': 0.01, 'max': 100000, 'daily': 250000}
}


# =============================================================================
# TRANSFER ENDPOINTS
# =============================================================================

@transfers_bp.route('/transfers', methods=['POST'])
@token_auth_required
def create_transfer():
    """
    Create a new transfer (wire/ACH/RTP)
    """
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        required = ['transfer_type', 'direction', 'amount', 'to_account_number']
        for field in required:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing required field: {field}'}), 400
        
        transfer_type = data['transfer_type'].lower()
        direction = data['direction'].lower()
        amount = float(data['amount'])
        
        # Validate transfer type
        valid_types = ['ach', 'wire', 'rtp', 'internal']
        if transfer_type not in valid_types:
            return jsonify({'status': 'error', 'message': f'Invalid transfer type. Must be one of: {", ".join(valid_types)}'}), 400
        
        # Validate direction
        if direction not in ['incoming', 'outgoing']:
            return jsonify({'status': 'error', 'message': 'Direction must be incoming or outgoing'}), 400
        
        # Validate amount
        limits = TRANSFER_LIMITS.get(transfer_type, TRANSFER_LIMITS['ach'])
        if amount < limits['min']:
            return jsonify({'status': 'error', 'message': f'Amount must be at least {limits["min"]}'}), 400
        if amount > limits['max']:
            return jsonify({'status': 'error', 'message': f'Amount cannot exceed {limits["max"]}'}), 400
        
        # Calculate fee
        fee_key = f"{direction}_wire" if transfer_type == 'wire' else transfer_type
        fee = WIRE_FEES.get(fee_key, WIRE_FEES['ach'])
        
        # Generate transfer ID
        prefix = 'ACH' if transfer_type == 'ach' else ('WRT' if transfer_type == 'wire' else 'RTP')
        transfer_id = f"{prefix}-{secrets.token_hex(4).upper()}"
        
        transfer = {
            'transfer_id': transfer_id,
            'user_id': user_id,
            'transfer_type': transfer_type,
            'direction': direction,
            'from_account_number': data.get('from_account_number', ''),
            'to_account_number': data['to_account_number'],
            'from_routing_number': data.get('from_routing_number', ''),
            'to_routing_number': data.get('to_routing_number', ''),
            'amount': amount,
            'currency': data.get('currency', 'USD'),
            'fee': fee,
            'exchange_rate': 1.0,
            'status': 'pending',
            'description': data.get('description', ''),
            'reference': data.get('reference', ''),
            'initiated_at': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        transfer_store.create_transfer(transfer)
        
        telemetry_logger.log_info(f"Transfer created: {transfer_id}", {'context': 'transfers'})
        
        return jsonify({'status': 'success', 'transfer': transfer}), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_transfer'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@transfers_bp.route('/transfers', methods=['GET'])
@token_auth_required
def list_transfers():
    """List all transfers for the user"""
    try:
        user_id = g.get('user_id', 'test_user')
        transfers = transfer_store.get_transfers_by_user(user_id)
        
        return jsonify({
            'status': 'success',
            'transfers': transfers,
            'count': len(transfers)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_transfers'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@transfers_bp.route('/transfers/<transfer_id>', methods=['GET'])
@token_auth_required
def get_transfer(transfer_id):
    """Get transfer details"""
    try:
        transfer = transfer_store.get_transfer(transfer_id)
        
        if not transfer:
            return jsonify({'status': 'error', 'message': 'Transfer not found'}), 404
        
        return jsonify({
            'status': 'success',
            'transfer': transfer
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_transfer'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@transfers_bp.route('/transfers/<transfer_id>/cancel', methods=['POST'])
@token_auth_required
def cancel_transfer(transfer_id):
    """Cancel a pending transfer"""
    try:
        transfer = transfer_store.get_transfer(transfer_id)
        
        if not transfer:
            return jsonify({'status': 'error', 'message': 'Transfer not found'}), 404
        
        if transfer['status'] != 'pending':
            return jsonify({'status': 'error', 'message': f'Cannot cancel transfer with status: {transfer["status"]}'}), 400
        
        transfer_store.update_transfer(transfer_id, {
            'status': 'cancelled',
            'cancelled_at': datetime.now(timezone.utc).isoformat()
        })
        
        telemetry_logger.log_info(f"Transfer cancelled: {transfer_id}", {'context': 'transfers'})
        
        return jsonify({
            'status': 'success',
            'transfer': transfer_store.get_transfer(transfer_id)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'cancel_transfer'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@transfers_bp.route('/transfers/<transfer_id>/complete', methods=['POST'])
@token_auth_required
def complete_transfer(transfer_id):
    """Mark a transfer as completed (for testing/simulation)"""
    try:
        transfer = transfer_store.get_transfer(transfer_id)
        
        if not transfer:
            return jsonify({'status': 'error', 'message': 'Transfer not found'}), 404
        
        if transfer['status'] != 'pending':
            return jsonify({'status': 'error', 'message': f'Cannot complete transfer with status: {transfer["status"]}'}), 400
        
        transfer_store.update_transfer(transfer_id, {
            'status': 'completed',
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'completed_at': datetime.now(timezone.utc).isoformat()
        })
        
        telemetry_logger.log_info(f"Transfer completed: {transfer_id}", {'context': 'transfers'})
        
        return jsonify({
            'status': 'success',
            'transfer': transfer_store.get_transfer(transfer_id)
        }), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'complete_transfer'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@transfers_bp.route('/fees', methods=['GET'])
def get_fees():
    """Get transfer fee schedule"""
    return jsonify({
        'status': 'success',
        'fees': WIRE_FEES,
        'limits': TRANSFER_LIMITS,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


@transfers_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for transfers service"""
    return jsonify({
        'status': 'healthy',
        'service': 'transfers',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


__all__ = ['transfers_bp']
