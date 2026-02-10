

"""
Payments Service Module for JPMorgan Financial APIs
Handles payment processing, transaction management, and payment-related operations.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid
import stripe
from src.models.payments import Payment, PaymentStatus, PaymentType
from src.structured_logger import app_logger
from config import config


class PaymentsService:
    """
    Service class for handling payment operations
    """

    # Expose enums for external access
    PaymentType = PaymentType
    PaymentStatus = PaymentStatus

    def __init__(self):
        self.logger = app_logger
        self._payments = {}  # In-memory storage for demo purposes

        # Initialize Stripe
        if config.STRIPE_SECRET_KEY:
            stripe.api_key = config.STRIPE_SECRET_KEY
            stripe.api_version = config.STRIPE_API_VERSION
            self.logger.info("Stripe payment service initialized")
        else:
            self.logger.warning("Stripe secret key not configured - Stripe payments disabled")

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
            extra_metadata=metadata or {}
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
            payment.metadata_json['transaction_id'] = transaction_id

        self.logger.info(f"Payment {payment_id} status updated to {status}")

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


        # For demo purposes, we'll simulate success/failure randomly
        import random
        processed_at = datetime.now(timezone.utc)
        payment.processed_at = processed_at
        payment.processing_time_ms = (processed_at - payment.created_at).total_seconds() * 1000

        # Simulate occasional failures for demo
        if random.random() < 0.1:  # 10% failure rate
            payment.status = PaymentStatus.FAILED
            payment.error_code = random.choice(['INSUFFICIENT_FUNDS', 'CARD_DECLINED', 'NETWORK_ERROR', 'TIMEOUT'])
            payment.error_message = {
                'INSUFFICIENT_FUNDS': 'Account balance insufficient for transaction',
                'CARD_DECLINED': 'Card issuer declined the transaction',
                'NETWORK_ERROR': 'Network connectivity issue during processing',
                'TIMEOUT': 'Transaction timed out during processing'
            }.get(payment.error_code, 'Unknown error occurred')
            self.logger.error(f"Payment {payment_id} failed with error: {payment.error_code} - {payment.error_message}")
            return False
        else:
            payment.status = PaymentStatus.COMPLETED
            payment.metadata_json['transaction_id'] = f"txn_{uuid.uuid4().hex[:8]}"
            self.logger.info(f"Payment {payment_id} processed successfully in {payment.processing_time_ms:.2f}ms")
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
            extra_metadata={
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

        # Processing time metrics
        processed_payments = [p for p in self._payments.values() if p.processed_at is not None and p.processing_time_ms is not None]
        avg_processing_time = sum(p.processing_time_ms for p in processed_payments) / len(processed_payments) if processed_payments else 0
        max_processing_time = max((p.processing_time_ms for p in processed_payments), default=0)

        return {
            'total_payments': total_payments,
            'total_amount': total_amount,
            'completed_payments': completed_payments,
            'pending_payments': pending_payments,
            'completion_rate': (completed_payments / total_payments) if total_payments > 0 else 0,
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'processed_payments_count': len(processed_payments)
        }

    def get_payment_throughput_by_minute(self, hours: int = 2) -> List[Dict[str, Any]]:
        """
        Get payment processing throughput by minute

        Args:
            hours: Number of hours to look back

        Returns:
            List of dicts with timestamp and count
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        processed_payments = [p for p in self._payments.values()
                            if p.processed_at is not None and p.processed_at > cutoff_time]

        # Group by minute
        throughput = {}
        for payment in processed_payments:
            minute_key = payment.processed_at.replace(second=0, microsecond=0)
            throughput[minute_key] = throughput.get(minute_key, 0) + 1

        # Convert to list and sort
        result = [{'timestamp': ts.isoformat(), 'count': count}
                 for ts, count in throughput.items()]
        result.sort(key=lambda x: x['timestamp'])

        return result

    def get_failed_payments_count(self, minutes: int = 5) -> int:
        """
        Get count of failed payments within the specified time window

        Args:
            minutes: Number of minutes to look back

        Returns:
            Count of failed payments
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        failed_payments = [p for p in self._payments.values()
                          if p.status == PaymentStatus.FAILED and
                          p.processed_at is not None and
                          p.processed_at > cutoff_time]

        return len(failed_payments)

    def get_failed_payments(self, limit: int = 50) -> List[Payment]:
        """
        Get failed payments

        Args:
            limit: Maximum number of payments to return

        Returns:
            List of failed Payment objects
        """
        failed_payments = [p for p in self._payments.values() if p.status == PaymentStatus.FAILED]
        failed_payments.sort(key=lambda x: x.created_at, reverse=True)

        return failed_payments[:limit]

    def get_error_code_distribution(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Get error code distribution for failed payments within the specified time window

        Args:
            hours: Number of hours to look back

        Returns:
            List of dicts with error_code and count
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        failed_payments = [p for p in self._payments.values()
                          if p.status == PaymentStatus.FAILED and
                          p.processed_at is not None and
                          p.processed_at > cutoff_time and
                          p.error_code is not None]

        # Group by error code
        error_distribution = {}
        for payment in failed_payments:
            error_distribution[payment.error_code] = error_distribution.get(payment.error_code, 0) + 1

        # Convert to list and sort by count descending
        result = [{'error_code': error_code, 'count': count}
                 for error_code, count in error_distribution.items()]
        result.sort(key=lambda x: x['count'], reverse=True)

        return result

    def get_queue_depth(self) -> int:
        """
        Get the current queue depth (number of pending payments)

        Returns:
            Number of payments in pending/processing status (ApproximateNumberOfMessagesVisible equivalent)
        """
        pending_payments = len([p for p in self._payments.values()
                               if p.status in [PaymentStatus.PENDING, PaymentStatus.PROCESSING]])

        return pending_payments

    def get_api_latency_percentile(self, percentile: float = 0.95, minutes: int = 5) -> float:
        """
        Get the specified percentile of API latency over the given time window

        Args:
            percentile: Percentile to calculate (0.95 for 95th percentile)
            minutes: Number of minutes to look back

        Returns:
            Latency value at the specified percentile in seconds
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        processed_payments = [p for p in self._payments.values()
                            if p.processed_at is not None and
                            p.processing_time_ms is not None and
                            p.processed_at > cutoff_time]

        if not processed_payments:
            return 0.0

        # Convert processing time from milliseconds to seconds
        latencies = [p.processing_time_ms / 1000 for p in processed_payments]
        latencies.sort()

        # Calculate percentile index
        index = int(percentile * (len(latencies) - 1))
        return latencies[index]

    def get_database_connection_rate(self, minutes: int = 1) -> float:
        """
        Get the rate of database connections over the specified time window
        (Simulated for demo - in real implementation would query pg_stat_activity)

        Args:
            minutes: Number of minutes to look back

        Returns:
            Connection rate per minute
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        # Simulate database connection activity based on payment processing
        # In a real implementation, this would query PostgreSQL's pg_stat_activity
        recent_payments = len([p for p in self._payments.values()
                             if p.created_at > cutoff_time])

        # Simulate connection rate based on payment activity
        # Assuming each payment operation requires database connections
        connection_rate = recent_payments / minutes

        return connection_rate

    def get_recent_payments_with_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent payments with error information (equivalent to Loki + SQL query)
        Filters for payments with errors and returns specified fields

        Args:
            limit: Maximum number of payments to return

        Returns:
            List of dicts with payment_id, amount, status, processing_time_ms, processed_at
        """
        # Get payments that have errors (failed status or error_code present)
        error_payments = [p for p in self._payments.values()
                         if p.status == PaymentStatus.FAILED or p.error_code is not None]

        # Sort by processed_at descending (most recent first)
        error_payments.sort(key=lambda x: x.processed_at or x.created_at, reverse=True)

        # Convert to dict format with specified fields
        result = []
        for payment in error_payments[:limit]:
            result.append({
                'payment_id': payment.id,
                'amount': payment.amount,
                'status': payment.status,
                'processing_time_ms': payment.processing_time_ms,
                'processed_at': payment.processed_at.isoformat() if payment.processed_at else None,
                'error_code': payment.error_code
            })

        return result

    def get_failed_payments_sql_style(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get failed payments with error details (equivalent to SQL query)
        SELECT payment_id, amount, error_code, error_message, processed_at
        FROM payments WHERE status = 'failed' ORDER BY processed_at DESC LIMIT 100

        Args:
            limit: Maximum number of payments to return

        Returns:
            List of dicts with payment_id, amount, error_code, error_message, processed_at
        """
        # Get only failed payments
        failed_payments = [p for p in self._payments.values() if p.status == PaymentStatus.FAILED]

        # Sort by processed_at descending (most recent first)
        failed_payments.sort(key=lambda x: x.processed_at or x.created_at, reverse=True)

        # Convert to dict format with specified fields
        result = []
        for payment in failed_payments[:limit]:
            result.append({
                'payment_id': payment.id,
                'amount': payment.amount,
                'error_code': payment.error_code,
                'error_message': payment.error_message,
                'processed_at': payment.processed_at.isoformat() if payment.processed_at else None
            })

        return result

    def check_processing_time_alert(self, threshold_seconds: float = 2.0, window_minutes: int = 10) -> Dict[str, Any]:
        """
        Check if average processing time exceeds threshold for the specified time window

        Args:
            threshold_seconds: Processing time threshold in seconds
            window_minutes: Time window to check in minutes

        Returns:
            Dict with alert status and details
        """
        # Get 95th percentile latency (most representative of worst-case performance)
        percentile_latency = self.get_api_latency_percentile(percentile=0.95, minutes=window_minutes)

        is_alert = percentile_latency > threshold_seconds

        return {
            'alert_type': 'processing_time',
            'alert_triggered': is_alert,
            'current_value': percentile_latency,
            'threshold': threshold_seconds,
            'window_minutes': window_minutes,
            'message': f'95th percentile processing time is {percentile_latency:.2f}s (threshold: {threshold_seconds}s)' if is_alert else f'Processing time OK: {percentile_latency:.2f}s'
        }

    def check_error_rate_alert(self, threshold_percent: float = 5.0, window_minutes: int = 5) -> Dict[str, Any]:
        """
        Check if error rate exceeds threshold for the specified time window

        Args:
            threshold_percent: Error rate threshold as percentage
            window_minutes: Time window to check in minutes

        Returns:
            Dict with alert status and details
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        # Count total processed payments in window
        total_processed = len([p for p in self._payments.values()
                              if p.processed_at is not None and p.processed_at > cutoff_time])

        # Count failed payments in window
        failed_count = self.get_failed_payments_count(minutes=window_minutes)

        # Calculate error rate
        error_rate = (failed_count / total_processed * 100) if total_processed > 0 else 0.0

        is_alert = error_rate > threshold_percent

        return {
            'alert_type': 'error_rate',
            'alert_triggered': is_alert,
            'current_value': error_rate,
            'threshold': threshold_percent,
            'window_minutes': window_minutes,
            'failed_count': failed_count,
            'total_processed': total_processed,
            'message': f'Error rate is {error_rate:.1f}% (threshold: {threshold_percent}%)' if is_alert else f'Error rate OK: {error_rate:.1f}%'
        }

    def check_queue_depth_alert(self, threshold: int = 1000) -> Dict[str, Any]:
        """
        Check if queue depth exceeds threshold

        Args:
            threshold: Queue depth threshold

        Returns:
            Dict with alert status and details
        """
        current_depth = self.get_queue_depth()

        is_alert = current_depth > threshold

        return {
            'alert_type': 'queue_depth',
            'alert_triggered': is_alert,
            'current_value': current_depth,
            'threshold': threshold,
            'message': f'Queue depth is {current_depth} (threshold: {threshold})' if is_alert else f'Queue depth OK: {current_depth}'
        }

    def get_all_alerts(self) -> List[Dict[str, Any]]:
        """
        Get status of all configured alerts

        Returns:
            List of alert status dictionaries
        """
        alerts = []

        # Processing Time Alert: Avg processing time > 2 seconds for 10 minutes
        alerts.append(self.check_processing_time_alert(threshold_seconds=2.0, window_minutes=10))

        # Error Rate Alert: Error rate > 5%
        alerts.append(self.check_error_rate_alert(threshold_percent=5.0, window_minutes=5))

        # Queue Depth Alert: Queue depth > 1000 messages
        alerts.append(self.check_queue_depth_alert(threshold=1000))

        return alerts

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get only alerts that are currently triggered

        Returns:
            List of active alert dictionaries
        """
        all_alerts = self.get_all_alerts()
        return [alert for alert in all_alerts if alert['alert_triggered']]

    # Stripe-specific methods
    def create_stripe_payment_intent(self, amount: float, currency: str = None,
                                    description: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a Stripe payment intent

        Args:
            amount: Payment amount in cents
            currency: Currency code (defaults to config)
            description: Payment description
            metadata: Additional metadata

        Returns:
            Dict with payment intent details
        """
        try:
            if not config.STRIPE_SECRET_KEY:
                return {
                    'status': 'error',
                    'error': 'Stripe not configured'
                }

            currency = currency or config.STRIPE_CURRENCY

            # Convert amount to cents if it's in dollars
            if currency.lower() == 'usd' and amount < 100:
                amount_cents = int(amount * 100)
            else:
                amount_cents = int(amount)

            intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency=currency,
                description=description,
                metadata=metadata or {},
                automatic_payment_methods={'enabled': True}
            )

            self.logger.info(f"Stripe payment intent created: {intent.id}")

            return {
                'status': 'success',
                'payment_intent_id': intent.id,
                'client_secret': intent.client_secret,
                'amount': amount_cents,
                'currency': currency,
                'description': description
            }

        except stripe.error.StripeError as e:
            self.logger.error(f"Stripe payment intent creation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        except Exception as e:
            self.logger.error(f"Unexpected error creating Stripe payment intent: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def confirm_stripe_payment(self, payment_intent_id: str) -> Dict[str, Any]:
        """
        Confirm a Stripe payment intent

        Args:
            payment_intent_id: Stripe payment intent ID

        Returns:
            Dict with confirmation details
        """
        try:
            if not config.STRIPE_SECRET_KEY:
                return {
                    'status': 'error',
                    'error': 'Stripe not configured'
                }

            intent = stripe.PaymentIntent.retrieve(payment_intent_id)

            if intent.status == 'succeeded':
                self.logger.info(f"Stripe payment confirmed: {payment_intent_id}")
                return {
                    'status': 'success',
                    'payment_intent_id': payment_intent_id,
                    'amount': intent.amount,
                    'currency': intent.currency,
                    'status': intent.status
                }
            else:
                return {
                    'status': 'pending',
                    'payment_intent_id': payment_intent_id,
                    'status': intent.status
                }

        except stripe.error.StripeError as e:
            self.logger.error(f"Stripe payment confirmation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        except Exception as e:
            self.logger.error(f"Unexpected error confirming Stripe payment: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def process_stripe_webhook(self, payload: str, signature: str) -> Dict[str, Any]:
        """
        Process Stripe webhook events

        Args:
            payload: Raw webhook payload
            signature: Stripe signature header

        Returns:
            Dict with processing result
        """
        try:
            if not config.STRIPE_WEBHOOK_SECRET:
                return {
                    'status': 'error',
                    'error': 'Stripe webhook secret not configured'
                }

            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, signature, config.STRIPE_WEBHOOK_SECRET
            )

            self.logger.info(f"Stripe webhook received: {event.type}")

            # Handle different event types
            if event.type == 'payment_intent.succeeded':
                payment_intent = event.data.object
                self.logger.info(f"Payment succeeded: {payment_intent.id}")

                # Update local payment record if it exists
                # This would typically update the payment status in your database

            elif event.type == 'payment_intent.payment_failed':
                payment_intent = event.data.object
                self.logger.error(f"Payment failed: {payment_intent.id}")

            return {
                'status': 'success',
                'event_type': event.type,
                'event_id': event.id
            }

        except stripe.error.SignatureVerificationError as e:
            self.logger.error(f"Stripe webhook signature verification failed: {str(e)}")
            return {
                'status': 'error',
                'error': 'Invalid signature'
            }
        except Exception as e:
            self.logger.error(f"Error processing Stripe webhook: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def create_stripe_refund(self, payment_intent_id: str, amount: Optional[int] = None,
                           reason: str = 'requested_by_customer') -> Dict[str, Any]:
        """
        Create a Stripe refund

        Args:
            payment_intent_id: Original payment intent ID
            amount: Refund amount in cents (full refund if not specified)
            reason: Refund reason

        Returns:
            Dict with refund details
        """
        try:
            if not config.STRIPE_SECRET_KEY:
                return {
                    'status': 'error',
                    'error': 'Stripe not configured'
                }

            refund = stripe.Refund.create(
                payment_intent=payment_intent_id,
                amount=amount,
                reason=reason
            )

            self.logger.info(f"Stripe refund created: {refund.id} for payment {payment_intent_id}")

            return {
                'status': 'success',
                'refund_id': refund.id,
                'amount': refund.amount,
                'currency': refund.currency,
                'status': refund.status
            }

        except stripe.error.StripeError as e:
            self.logger.error(f"Stripe refund creation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        except Exception as e:
            self.logger.error(f"Unexpected error creating Stripe refund: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }


# Global payments service instance
payments_service = PaymentsService()
