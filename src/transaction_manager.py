#!/usr/bin/env python3
"""
Transaction Manager for JPMorgan Financial APIs
Ensures ACID compliance and proper rollback mechanisms for database operations
"""
from contextlib import contextmanager
from typing import Generator, Any, Callable, Optional, List
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.orm import Session
import logging
from datetime import datetime, timezone

from src.database_fixed import db_manager, DBAssetModel, DBBusinessModel
from src.logger import telemetry_logger
from src.models.payments import PaymentStatus

class TransactionManager:
    """Manages database transactions with ACID compliance and rollback support"""

    def __init__(self):
        self.logger = telemetry_logger.get_logger()
        self._active_transactions = {}

    @contextmanager
    def transaction(self, session: Optional[Session] = None, isolation_level: str = "READ_COMMITTED") -> Generator[Session, None, None]:
        """
        Context manager for database transactions with automatic rollback on failure

        Args:
            session: Optional existing session to use
            isolation_level: Transaction isolation level

        Yields:
            Database session within transaction context
        """
        session_created = False
        if session is None:
            session_context = db_manager.get_session()
            transaction_session = session_context.__enter__()
            session_created = True
        else:
            transaction_session = session

        transaction_id = f"txn_{datetime.now(timezone.utc).timestamp()}_{id(transaction_session)}"

        try:
            # Set transaction isolation level if specified (skip for SQLite)
            if isolation_level != "READ_COMMITTED" and "sqlite" not in str(self.engine.url).lower():
                transaction_session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))

            # Begin transaction
            transaction_session.begin()
            self._active_transactions[transaction_id] = {
                'session': transaction_session,
                'start_time': datetime.now(timezone.utc),
                'operations': []
            }

            self.logger.debug(f"Transaction {transaction_id} started")

            yield transaction_session

            # Commit transaction
            transaction_session.commit()
            self.logger.info(f"Transaction {transaction_id} committed successfully")

        except IntegrityError as e:
            transaction_session.rollback()
            self.logger.error(f"Transaction {transaction_id} rolled back due to integrity error: {e}")
            raise ValueError(f"Data integrity violation: {str(e)}") from e

        except OperationalError as e:
            transaction_session.rollback()
            self.logger.error(f"Transaction {transaction_id} rolled back due to operational error: {e}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e

        except SQLAlchemyError as e:
            transaction_session.rollback()
            self.logger.error(f"Transaction {transaction_id} rolled back due to SQLAlchemy error: {e}")
            raise RuntimeError(f"Database error: {str(e)}") from e

        except Exception as e:
            transaction_session.rollback()
            self.logger.error(f"Transaction {transaction_id} rolled back due to unexpected error: {e}")
            raise

        finally:
            # Clean up transaction tracking
            if transaction_id in self._active_transactions:
                del self._active_transactions[transaction_id]

            # Close session if we created it
            if session_created:
                session_context.__exit__(None, None, None)

    def execute_with_retry(self, operation: Callable[[Session], Any],
                          max_retries: int = 3,
                          isolation_level: str = "READ_COMMITTED") -> Any:
        """
        Execute database operation with automatic retry on transient failures

        Args:
            operation: Function that takes a session and performs database operations
            max_retries: Maximum number of retry attempts
            isolation_level: Transaction isolation level

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                with self.transaction(isolation_level=isolation_level) as session:
                    result = operation(session)
                    return result

            except (OperationalError, RuntimeError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    import time
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Operation failed after {max_retries} attempts: {e}")
                    break

        raise RuntimeError(f"Operation failed after {max_retries} retries: {last_error}") from last_error

    def execute_batch(self, operations: List[Callable[[Session], Any]],
                     isolation_level: str = "READ_COMMITTED") -> List[Any]:
        """
        Execute multiple operations in a single transaction (all-or-nothing)

        Args:
            operations: List of functions that take a session and perform operations
            isolation_level: Transaction isolation level

        Returns:
            List of operation results

        Raises:
            RuntimeError: If any operation fails (all operations rolled back)
        """
        results = []

        def batch_operation(session: Session) -> List[Any]:
            for i, operation in enumerate(operations):
                try:
                    result = operation(session)
                    results.append(result)
                    self.logger.debug(f"Batch operation {i + 1}/{len(operations)} completed")
                except Exception as e:
                    self.logger.error(f"Batch operation {i + 1} failed: {e}")
                    raise
            return results

        return self.execute_with_retry(batch_operation, isolation_level=isolation_level)

    def health_check(self) -> dict:
        """
        Perform transaction health check

        Returns:
            Dictionary with health check results
        """
        try:
            with self.transaction() as session:
                # Test basic transaction operations
                session.execute(text("SELECT 1"))
                return {
                    'status': 'healthy',
                    'active_transactions': len(self._active_transactions),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            self.logger.error(f"Transaction health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'active_transactions': len(self._active_transactions),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_transaction_stats(self) -> dict:
        """
        Get transaction statistics

        Returns:
            Dictionary with transaction statistics
        """
        return {
            'active_transactions': len(self._active_transactions),
            'transaction_details': [
                {
                    'transaction_id': txn_id,
                    'start_time': details['start_time'].isoformat(),
                    'duration_seconds': (datetime.now(timezone.utc) - details['start_time']).total_seconds(),
                    'operations_count': len(details['operations'])
                }
                for txn_id, details in self._active_transactions.items()
            ]
        }

class BusinessTransactionManager(TransactionManager):
    """Transaction manager specifically for business operations"""

    def create_business_with_assets(self, business_data: dict, assets_data: List[dict]) -> tuple:
        """
        Create business with associated assets in a single transaction

        Args:
            business_data: Business creation data
            assets_data: List of asset creation data

        Returns:
            Tuple of (business, assets) objects
        """
        def operation(session: Session):
            # Create business
            business = db_manager.create_business(business_data)

            # Create associated assets
            assets = []
            for asset_data in assets_data:
                asset_data['business_id'] = business.id
                asset = db_manager.create_asset(asset_data)
                assets.append(asset)

            return business, assets

        # Use READ_COMMITTED for SQLite compatibility
        return self.execute_with_retry(operation, isolation_level="READ_COMMITTED")

    def transfer_asset_ownership(self, asset_id: int, new_business_id: int, transfer_value: float) -> bool:
        """
        Transfer asset ownership between businesses with value update

        Args:
            asset_id: Asset to transfer
            new_business_id: New owning business
            transfer_value: New asset value after transfer

        Returns:
            True if transfer successful
        """
        def operation(session: Session):
            # Verify asset exists and get current business
            asset = session.query(DBAssetModel).filter(DBAssetModel.id == asset_id).first()
            if not asset:
                raise ValueError(f"Asset {asset_id} not found")

            old_business_id = asset.business_id

            # Verify new business exists
            new_business = session.query(DBBusinessModel).filter(DBBusinessModel.id == new_business_id).first()
            if not new_business:
                raise ValueError(f"Business {new_business_id} not found")

            # Update asset ownership and value
            asset.business_id = new_business_id
            asset.current_value = transfer_value
            asset.updated_at = datetime.now(timezone.utc)

            # Ensure the asset is added to the session for tracking
            session.add(asset)

            self.logger.info(f"Asset {asset_id} transferred from business {old_business_id} to {new_business_id}")
            return True

        # Use READ_COMMITTED for SQLite compatibility
        return self.execute_with_retry(operation, isolation_level="READ_COMMITTED")

class PaymentTransactionManager(TransactionManager):
    """Transaction manager specifically for payment operations"""

    def process_payment_with_fee(self, payment_data: dict, fee_data: dict) -> tuple:
        """
        Process payment with associated fee calculation in a single transaction

        Args:
            payment_data: Payment creation data
            fee_data: Fee calculation data

        Returns:
            Tuple of (payment, fee) objects
        """
        def operation(session: Session):
            # Create payment record
            from src.payments_service import payments_service
            payment = payments_service.create_payment(**payment_data)

            # Calculate and create fee record
            fee_amount = payment_data['amount'] * fee_data.get('fee_percentage', 0.029)  # 2.9% default
            fee = {
                'id': f"fee_{payment.id}",
                'payment_id': payment.id,
                'fee_type': fee_data.get('fee_type', 'processing_fee'),
                'amount': fee_amount,
                'currency': payment_data.get('currency', 'USD'),
                'description': f"Processing fee for payment {payment.id}"
            }

            # In a real implementation, this would create a TransactionFee record
            # For now, we'll just return the fee data
            return payment, fee

        return self.execute_with_retry(operation, isolation_level="READ_COMMITTED")

    def refund_payment_with_adjustment(self, payment_id: str, refund_amount: float) -> tuple:
        """
        Process refund with balance adjustment in a single transaction

        Args:
            payment_id: Payment to refund
            refund_amount: Amount to refund

        Returns:
            Tuple of (original_payment, refund_payment)
        """
        def operation(session: Session):
            from src.payments_service import payments_service

            # Get original payment
            original_payment = payments_service.get_payment(payment_id)
            if not original_payment:
                raise ValueError(f"Payment {payment_id} not found")

            if original_payment.status != PaymentStatus.COMPLETED:
                raise ValueError(f"Cannot refund payment with status {original_payment.status}")

            # Process refund
            refund_success = payments_service.refund_payment(payment_id, refund_amount)
            if not refund_success:
                raise RuntimeError("Refund processing failed")

            # Get the refund payment (last payment in the list)
            user_payments = payments_service.get_user_payments(original_payment.user_id, limit=1)
            refund_payment = user_payments[0] if user_payments else None

            return original_payment, refund_payment

        # Use READ_COMMITTED for SQLite compatibility
        return self.execute_with_retry(operation, isolation_level="READ_COMMITTED")

# Global transaction managers
transaction_manager = TransactionManager()
business_transaction_manager = BusinessTransactionManager()
payment_transaction_manager = PaymentTransactionManager()
