"""
Payments Service Module for JPMorgan Financial APIs
Handles payment processing, transaction management, and payment-related operations.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid
from src.models.payments import Payment, PaymentStatus, PaymentType
from src.structured_logger import app_logger


class PaymentsService:
    """
    Service class for handling payment operations
    """

    def __init__(self):
        self.logger = app_logger.get_logger()
        self._payments = {}  # In-memory storage for demo purposes

    def create_payment(self, amount: float, payment_type: PaymentType,
                      user_id: str, description: str = "",
                      currency: str = "USD", metadata: Optional[Dict[str, Any]] = None) -> Payment:
        """
        Create a new payment transaction

        Args:
            amount: Payment amount
            payment_type: Type of payment (ACH, Card, Wallet, etc.)
            user_id: User initiating the payment
            description: Payment description
            currency: Currency code
            metadata: Additional payment metadata

        Returns:
            Payment: Created payment object
        """
        payment_id = str(uuid.uuid4())

        payment = Payment(
            id=payment_id,
            amount=amount,
            currency=currency,
            payment_type=payment_type,
            status=PaymentStatus.PENDING,
            user_id=user_id,
            description=description,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )

        self._payments[payment_id] = payment

        self.logger.info(f"Payment created: {payment_id} for user {user_id} amount {amount} {currency}")

        return payment

    def get_payment(self, payment_id: str) -> Optional[Payment]:
        """
        Get payment by ID

        Args:
            payment_id: Payment identifier

        Returns:
            Payment or None if not found
        """
        return self._payments.get(payment_id)

    def update_payment_status(self, payment_id: str, status: PaymentStatus,
                            transaction_id: Optional[str] = None) -> bool:
        """
        Update payment status

        Args:
            payment_id: Payment identifier
            status: New payment status
            transaction_id: External transaction ID

        Returns:
            bool: True if updated successfully
        """
        payment = self._payments.get(payment_id)
        if not payment:
            return False

        payment.status = status
        payment.updated_at = datetime.now(timezone.utc)

        if transaction_id:
            payment.metadata['transaction_id'] = transaction_id

        self.logger.info(f"Payment {payment_id} status updated to {status.value}")

        return True

    def process_payment(self, payment_id: str) -> bool:
        """
        Process a pending payment

        Args:
            payment_id: Payment identifier

        Returns:
            bool: True if processed successfully
        """
        payment = self._payments.get(payment_id)
        if not payment or payment.status != PaymentStatus.PENDING:
            return False

        # Simulate payment processing
        payment.status = PaymentStatus.PROCESSING
        payment.updated_at = datetime.now(timezone.utc)

        # In a real implementation, this would integrate with payment processors
        # For demo purposes, we'll simulate success
        payment.status = PaymentStatus.COMPLETED
        payment.metadata['processed_at'] = datetime.now(timezone.utc).isoformat()
        payment.metadata['transaction_id'] = f"txn_{uuid.uuid4().hex[:8]}"

        self.logger.info(f"Payment {payment_id} processed successfully")

        return True

    def get_user_payments(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Payment]:
        """
        Get payments for a specific user

        Args:
            user_id: User identifier
            limit: Maximum number of payments to return
            offset: Number of payments to skip

        Returns:
            List of Payment objects
        """
        user_payments = [p for p in self._payments.values() if p.user_id == user_id]
        user_payments.sort(key=lambda x: x.created_at, reverse=True)

        return user_payments[offset:offset + limit]

    def get_payments_by_status(self, status: PaymentStatus, limit: int = 100) -> List[Payment]:
        """
        Get payments by status

        Args:
            status: Payment status to filter by
            limit: Maximum number of payments to return

        Returns:
            List of Payment objects
        """
        status_payments = [p for p in self._payments.values() if p.status == status]
        status_payments.sort(key=lambda x: x.created_at, reverse=True)

        return status_payments[:limit]

    def cancel_payment(self, payment_id: str) -> bool:
        """
        Cancel a pending payment

        Args:
            payment_id: Payment identifier

        Returns:
            bool: True if cancelled successfully
        """
        payment = self._payments.get(payment_id)
        if not payment or payment.status not in [PaymentStatus.PENDING, PaymentStatus.PROCESSING]:
            return False

        payment.status = PaymentStatus.CANCELLED
        payment.updated_at = datetime.now(timezone.utc)

        self.logger.info(f"Payment {payment_id} cancelled")

        return True

    def refund_payment(self, payment_id: str, refund_amount: Optional[float] = None) -> bool:
        """
        Process a refund for a completed payment

        Args:
            payment_id: Payment identifier
            refund_amount: Amount to refund (full amount if not specified)

        Returns:
            bool: True if refund processed successfully
        """
        payment = self._payments.get(payment_id)
        if not payment or payment.status != PaymentStatus.COMPLETED:
            return False

        refund_amount = refund_amount or payment.amount

        if refund_amount > payment.amount:
            return False

        # Create refund payment record
        refund_payment = Payment(
            id=str(uuid.uuid4()),
            amount=-refund_amount,  # Negative amount for refund
            currency=payment.currency,
            payment_type=payment.payment_type,
            status=PaymentStatus.COMPLETED,
            user_id=payment.user_id,
            description=f"Refund for payment {payment_id}",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={
                'original_payment_id': payment_id,
                'refund_type': 'full' if refund_amount == payment.amount else 'partial'
            }
        )

        self._payments[refund_payment.id] = refund_payment

        self.logger.info(f"Refund processed for payment {payment_id}: {refund_amount} {payment.currency}")

        return True

    def get_payment_stats(self) -> Dict[str, Any]:
        """
        Get payment statistics

        Returns:
            Dict containing payment statistics
        """
        total_payments = len(self._payments)
        total_amount = sum(p.amount for p in self._payments.values() if p.amount > 0)
        completed_payments = len([p for p in self._payments.values() if p.status == PaymentStatus.COMPLETED])
        pending_payments = len([p for p in self._payments.values() if p.status == PaymentStatus.PENDING])

        return {
            'total_payments': total_payments,
            'total_amount': total_amount,
            'completed_payments': completed_payments,
            'pending_payments': pending_payments,
            'completion_rate': (completed_payments / total_payments) if total_payments > 0 else 0
        }


# Global payments service instance
payments_service = PaymentsService()
