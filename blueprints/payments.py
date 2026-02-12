"""
Payments Blueprint for JPMorgan Financial APIs
Provides comprehensive banking suite with card loading, transactions, and instant pay functionality.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, List

# Import services and utilities
from src.payments_service import payments_service
from src.logger import telemetry_logger
from src.models.payments import PaymentType, PaymentStatus, PaymentMethod

# Import authentication and rate limiting decorators
# These need to be imported from wherever they are defined in the project
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

payments_bp = Blueprint('payments', __name__)


# =============================================================================
# PAYMENT METHOD MANAGEMENT ENDPOINTS
# =============================================================================

@payments_bp.route('/payments/methods', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def add_payment_method():
    """
    Add a new payment method for the authenticated user
    Supports cards, bank accounts, and digital wallets
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Extract required fields
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        method_type = data.get('type')
        provider = data.get('provider', '')
        last_four = data.get('last_four', '')
        is_default = data.get('is_default', False)

        if not method_type:
            return jsonify({'error': 'Payment method type is required', 'status': 'error'}), 400

        # Validate method type
        valid_types = ['card', 'bank_account', 'wallet']
        if method_type not in valid_types:
            return jsonify({'error': f'Invalid payment method type. Must be one of: {valid_types}', 'status': 'error'}), 400

        # Create payment method (in a real implementation, this would be stored securely)
        method_id = str(uuid.uuid4())
        payment_method = {
            'id': method_id,
            'user_id': user_id,
            'type': method_type,
            'provider': provider,
            'last_four': last_four,
            'is_default': is_default,
            'is_active': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        telemetry_logger.log_info(f"Payment method added for user {user_id}: {method_type}", {'method_id': method_id})

        return jsonify({
            'status': 'success',
            'message': 'Payment method added successfully',
            'payment_method': payment_method
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'add_payment_method'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/methods', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_payment_methods():
    """
    Get all payment methods for the authenticated user
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # In a real implementation, this would query the database
        # For demo purposes, return mock data
        payment_methods = [
            {
                'id': 'pm_123',
                'user_id': user_id,
                'type': 'card',
                'provider': 'visa',
                'last_four': '4242',
                'is_default': True,
                'is_active': True,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        ]

        return jsonify({
            'status': 'success',
            'payment_methods': payment_methods,
            'count': len(payment_methods)
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_payment_methods'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/methods/<method_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_payment_method(method_id):
    """
    Delete a payment method
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # In a real implementation, this would delete from database
        telemetry_logger.log_info(f"Payment method {method_id} deleted for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': 'Payment method deleted successfully'
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_payment_method'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# CARD LOADING ENDPOINTS
# =============================================================================

@payments_bp.route('/payments/load', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def load_card():
    """
    Load funds onto a card or payment method
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        method_id = data.get('method_id')
        amount = data.get('amount')
        currency = data.get('currency', 'USD')

        if not method_id or not amount:
            return jsonify({'error': 'method_id and amount are required', 'status': 'error'}), 400

        if amount <= 0:
            return jsonify({'error': 'Amount must be positive', 'status': 'error'}), 400

        # Create a payment record for the load operation
        payment = payments_service.create_payment(
            amount=amount,
            payment_type=PaymentType.CARD,
            user_id=user_id,
            description=f"Card load - {method_id}",
            currency=currency,
            metadata={'operation': 'card_load', 'method_id': method_id}
        )

        # Process the payment immediately
        success = payments_service.process_payment(payment.id)

        if success:
            telemetry_logger.log_info(f"Card loaded successfully: {amount} {currency} for user {user_id}")
            return jsonify({
                'status': 'success',
                'message': 'Card loaded successfully',
                'payment_id': payment.id,
                'amount': amount,
                'currency': currency,
                'new_balance': amount  # In real implementation, would calculate actual balance
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Card loading failed',
                'payment_id': payment.id
            }), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'load_card'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/cards/<card_id>/balance', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_card_balance(card_id):
    """
    Get balance for a specific card
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # In a real implementation, this would query the card balance
        # For demo purposes, return mock balance
        balance = {
            'card_id': card_id,
            'available_balance': 1250.75,
            'pending_balance': 0.0,
            'currency': 'USD',
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'status': 'success',
            'balance': balance
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_card_balance'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# TRANSACTION PROCESSING ENDPOINTS
# =============================================================================

@payments_bp.route('/payments/process', methods=['POST'])
@token_auth_required
@conditional_limit("20 per minute")
def process_payment():
    """
    Process a payment transaction
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        amount = data.get('amount')
        payment_type = data.get('payment_type', 'card')
        description = data.get('description', '')
        currency = data.get('currency', 'USD')
        method_id = data.get('method_id')

        if not amount or not method_id:
            return jsonify({'error': 'amount and method_id are required', 'status': 'error'}), 400

        if amount <= 0:
            return jsonify({'error': 'Amount must be positive', 'status': 'error'}), 400

        # Validate payment type
        try:
            payment_type_enum = PaymentType(payment_type)
        except ValueError:
            return jsonify({'error': f'Invalid payment type: {payment_type}', 'status': 'error'}), 400

        # Create payment
        payment = payments_service.create_payment(
            amount=amount,
            payment_type=payment_type_enum,
            user_id=user_id,
            description=description,
            currency=currency,
            metadata={'method_id': method_id}
        )

        # Process payment
        success = payments_service.process_payment(payment.id)

        if success:
            payment_data = payment.to_dict()
            telemetry_logger.log_info(f"Payment processed: {payment.id} for user {user_id}")
            return jsonify({
                'status': 'success',
                'message': 'Payment processed successfully',
                'payment': payment_data
            }), 200
        else:
            payment_data = payment.to_dict()
            return jsonify({
                'status': 'error',
                'message': 'Payment processing failed',
                'payment': payment_data
            }), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'process_payment'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/transactions', methods=['GET'])
@token_auth_required
@conditional_limit("30 per minute")
def get_transactions():
    """
    Get transaction history for the authenticated user
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100
        offset = int(request.args.get('offset', 0))
        status = request.args.get('status')
        payment_type = request.args.get('type')

        # Get user payments
        payments = payments_service.get_user_payments(user_id, limit=limit, offset=offset)

        # Filter by status if provided
        if status:
            try:
                status_enum = PaymentStatus(status)
                payments = [p for p in payments if p.status == status_enum]
            except ValueError:
                return jsonify({'error': f'Invalid status: {status}', 'status': 'error'}), 400

        # Filter by payment type if provided
        if payment_type:
            try:
                type_enum = PaymentType(payment_type)
                payments = [p for p in payments if p.payment_type == type_enum.value]
            except ValueError:
                return jsonify({'error': f'Invalid payment type: {payment_type}', 'status': 'error'}), 400

        transactions = [p.to_dict() for p in payments]

        return jsonify({
            'status': 'success',
            'transactions': transactions,
            'count': len(transactions),
            'total_count': len(payments_service.get_user_payments(user_id, limit=1000))  # Rough total
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_transactions'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/transactions/<transaction_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_transaction_details(transaction_id):
    """
    Get detailed information about a specific transaction
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get payment details
        payment = payments_service.get_payment(transaction_id)

        if not payment:
            return jsonify({'error': 'Transaction not found', 'status': 'error'}), 404

        # Check if payment belongs to user (in real implementation)
        if payment.user_id != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        return jsonify({
            'status': 'success',
            'transaction': payment.to_dict()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_transaction_details'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# INSTANT PAY ENDPOINTS
# =============================================================================

@payments_bp.route('/payments/quick-pay', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def quick_pay():
    """
    Process an instant/quick payment
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        recipient_id = data.get('recipient_id')
        amount = data.get('amount')
        description = data.get('description', 'Quick Pay')
        currency = data.get('currency', 'USD')

        if not recipient_id or not amount:
            return jsonify({'error': 'recipient_id and amount are required', 'status': 'error'}), 400

        if amount <= 0:
            return jsonify({'error': 'Amount must be positive', 'status': 'error'}), 400

        # Create instant payment
        payment = payments_service.create_payment(
            amount=amount,
            payment_type=PaymentType.WALLET,
            user_id=user_id,
            description=f"{description} - To: {recipient_id}",
            currency=currency,
            metadata={'operation': 'quick_pay', 'recipient_id': recipient_id, 'instant': True}
        )

        # Process immediately for instant pay
        success = payments_service.process_payment(payment.id)

        if success:
            telemetry_logger.log_info(f"Quick pay processed: {amount} {currency} from {user_id} to {recipient_id}")
            return jsonify({
                'status': 'success',
                'message': 'Quick pay processed instantly',
                'payment_id': payment.id,
                'amount': amount,
                'recipient_id': recipient_id,
                'processed_at': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Quick pay failed',
                'payment_id': payment.id
            }), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'quick_pay'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/transfer', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def instant_transfer():
    """
    Process an instant transfer between accounts
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        to_account = data.get('to_account')
        amount = data.get('amount')
        transfer_type = data.get('transfer_type', 'instant')
        description = data.get('description', 'Instant Transfer')

        if not to_account or not amount:
            return jsonify({'error': 'to_account and amount are required', 'status': 'error'}), 400

        if amount <= 0:
            return jsonify({'error': 'Amount must be positive', 'status': 'error'}), 400

        # Create transfer payment
        payment = payments_service.create_payment(
            amount=amount,
            payment_type=PaymentType.WIRE if transfer_type == 'wire' else PaymentType.ACH,
            user_id=user_id,
            description=f"{description} - To: {to_account}",
            metadata={'operation': 'transfer', 'to_account': to_account, 'transfer_type': transfer_type}
        )

        # Process transfer
        success = payments_service.process_payment(payment.id)

        if success:
            telemetry_logger.log_info(f"Instant transfer processed: {amount} from {user_id} to {to_account}")
            return jsonify({
                'status': 'success',
                'message': f'{transfer_type.title()} transfer processed successfully',
                'payment_id': payment.id,
                'amount': amount,
                'to_account': to_account,
                'transfer_type': transfer_type,
                'estimated_completion': 'instant' if transfer_type == 'instant' else '1-2 business days'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Transfer failed',
                'payment_id': payment.id
            }), 400

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'instant_transfer'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/status/<payment_id>', methods=['GET'])
@token_auth_required
@conditional_limit("30 per minute")
def get_payment_status(payment_id):
    """
    Get real-time status of a payment
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        payment = payments_service.get_payment(payment_id)

        if not payment:
            return jsonify({'error': 'Payment not found', 'status': 'error'}), 404

        # Check ownership (in real implementation)
        if payment.user_id != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        status_info = {
            'payment_id': payment.id,
            'status': payment.status,
            'amount': payment.amount,
            'currency': payment.currency,
            'created_at': payment.created_at.isoformat() if payment.created_at else None,
            'processed_at': payment.processed_at.isoformat() if payment.processed_at else None,
            'processing_time_ms': payment.processing_time_ms,
            'error_code': payment.error_code,
            'error_message': payment.error_message
        }

        return jsonify({
            'status': 'success',
            'payment_status': status_info
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_payment_status'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# DASHBOARD AND ANALYTICS ENDPOINTS
# =============================================================================

@payments_bp.route('/payments/dashboard', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_payments_dashboard():
    """
    Get payments dashboard data including total payments, recent transactions, and payment status summary
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get total payments count and amount
        total_payments = len(payments_service.get_user_payments(user_id, limit=1000))
        user_payments = payments_service.get_user_payments(user_id, limit=1000)
        total_amount = sum(p.amount for p in user_payments if p.amount > 0)

        # Get recent transactions (last 10)
        recent_transactions = payments_service.get_user_payments(user_id, limit=10)

        # Get payment status summary
        status_counts = {}
        for payment in user_payments:
            status = payment.status
            status_counts[status] = status_counts.get(status, 0) + 1

        status_summary = [
            {'status': status, 'count': count}
            for status, count in status_counts.items()
        ]

        return jsonify({
            'status': 'success',
            'dashboard': {
                'total_payments': total_payments,
                'total_amount': total_amount,
                'recent_transactions': [t.to_dict() for t in recent_transactions],
                'status_summary': status_summary
            }
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_payments_dashboard'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/alerts', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_payment_alerts():
    """
    Get payment processing alerts status
    """
    try:
        # Get all alerts
        all_alerts = payments_service.get_all_alerts()

        # Get only active alerts
        active_alerts = payments_service.get_active_alerts()

        return jsonify({
            'status': 'success',
            'alerts': {
                'all': all_alerts,
                'active': active_alerts,
                'active_count': len(active_alerts)
            }
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_payment_alerts'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@payments_bp.route('/payments/stats', methods=['GET'])
@token_auth_required
@conditional_limit("5 per minute")
def get_payment_stats():
    """
    Get comprehensive payment statistics
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get user-specific stats
        user_payments = payments_service.get_user_payments(user_id, limit=1000)
        user_stats = {
            'total_payments': len(user_payments),
            'total_amount': sum(p.amount for p in user_payments if p.amount > 0),
            'successful_payments': len([p for p in user_payments if p.status == PaymentStatus.COMPLETED]),
            'failed_payments': len([p for p in user_payments if p.status == PaymentStatus.FAILED])
        }

        # Get global stats (admin only in real implementation)
        global_stats = payments_service.get_payment_stats()

        return jsonify({
            'status': 'success',
            'stats': {
                'user': user_stats,
                'global': global_stats
            }
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_payment_stats'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# BATCH PROCESSING ENDPOINTS (for future expansion)
# =============================================================================

@payments_bp.route('/batch/start', methods=['POST'])
@token_auth_required
@conditional_limit("2 per minute")
def start_batch_processing():
    """
    Start batch payment processing (placeholder for future implementation)
    """
    return jsonify({
        'status': 'success',
        'message': 'Batch processing started',
        'batch_id': str(uuid.uuid4())
    }), 200


@payments_bp.route('/batch/stop', methods=['POST'])
@token_auth_required
@conditional_limit("2 per minute")
def stop_batch_processing():
    """
    Stop batch payment processing (placeholder for future implementation)
    """
    return jsonify({
        'status': 'success',
        'message': 'Batch processing stopped'
    }), 200


@payments_bp.route('/batch/status', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_batch_status():
    """
    Get batch processing status (placeholder for future implementation)
    """
    return jsonify({
        'status': 'success',
        'batch_status': {
            'active': False,
            'processed_count': 0,
            'total_count': 0,
            'success_rate': 0.0
        }
    }), 200
