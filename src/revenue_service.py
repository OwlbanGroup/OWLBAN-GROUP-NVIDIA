"""
Revenue service for processing financial transactions and tracking revenue
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from src.models.revenue import RevenueTransaction, RevenueMetrics, RevenueType, TransactionStatus
from src.database_fixed import db_manager
from src.logger import telemetry_logger

class RevenueService:
    """Service for managing revenue transactions and metrics"""

    def __init__(self):
        self.logger = telemetry_logger.get_logger()

    def create_transaction(self, user_id: str, revenue_type: RevenueType,
                          amount: float, currency: str = 'USD',
                          description: str = None, merchant_name: str = None,
                          category: str = None, payment_method: str = None,
                          business_id: int = None, external_reference: str = None,
                          additional_metadata: str = None) -> RevenueTransaction:
        """
        Create a new revenue transaction

        Args:
            user_id: User identifier
            revenue_type: Type of revenue transaction
            amount: Transaction amount
            currency: Currency code (default USD)
            description: Transaction description
            merchant_name: Merchant/business name
            category: Transaction category
            payment_method: Payment method used
            business_id: Associated business ID
            external_reference: External system reference
            metadata: Additional metadata

        Returns:
            Created RevenueTransaction object
        """
        try:
            # Generate unique transaction ID
            transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"

            # Calculate fees and taxes based on revenue type
            fee_amount, tax_amount = self._calculate_fees_and_taxes(revenue_type, amount)
            net_amount = amount - fee_amount - tax_amount

            # Create transaction
            transaction = RevenueTransaction(
                transaction_id=transaction_id,
                user_id=user_id,
                business_id=business_id,
                revenue_type=revenue_type,
                amount=amount,
                currency=currency,
                status=TransactionStatus.PENDING,
                description=description,
                merchant_name=merchant_name,
                category=category,
                fee_amount=fee_amount,
                tax_amount=tax_amount,
                net_amount=net_amount,
                payment_method=payment_method,
                external_reference=external_reference,
                additional_data=additional_metadata
            )

            # Save to database
            with db_manager.get_session() as session:
                session.add(transaction)
                session.commit()
                session.refresh(transaction)

            self.logger.info(f"Created revenue transaction: {transaction_id} for user {user_id}")
            return transaction

        except Exception as e:
            self.logger.error(f"Failed to create revenue transaction: {e}")
            raise

    def process_transaction(self, transaction_id: str, success: bool = True,
                           settlement_date: datetime = None) -> bool:
        """
        Process a pending transaction (mark as completed or failed)

        Args:
            transaction_id: Transaction ID to process
            success: Whether the transaction was successful
            settlement_date: When the transaction settled

        Returns:
            True if processed successfully
        """
        try:
            with db_manager.get_session() as session:
                transaction = session.query(RevenueTransaction).filter_by(
                    transaction_id=transaction_id
                ).first()

                if not transaction:
                    self.logger.warning(f"Transaction not found: {transaction_id}")
                    return False

                if transaction.status != TransactionStatus.PENDING:
                    self.logger.warning(f"Transaction {transaction_id} is not pending")
                    return False

                # Update transaction status
                transaction.status = TransactionStatus.COMPLETED if success else TransactionStatus.FAILED
                transaction.settlement_date = settlement_date or datetime.now(timezone.utc)
                transaction.updated_at = datetime.now(timezone.utc)

                session.commit()

                self.logger.info(f"Processed transaction {transaction_id}: {'success' if success else 'failed'}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to process transaction {transaction_id}: {e}")
            return False

    def get_transaction(self, transaction_id: str) -> Optional[RevenueTransaction]:
        """Get transaction by ID"""
        try:
            with db_manager.get_session() as session:
                return session.query(RevenueTransaction).filter_by(
                    transaction_id=transaction_id
                ).first()
        except Exception as e:
            self.logger.error(f"Failed to get transaction {transaction_id}: {e}")
            return None

    def get_user_transactions(self, user_id: str, limit: int = 50,
                            offset: int = 0) -> List[RevenueTransaction]:
        """Get transactions for a user"""
        try:
            with db_manager.get_session() as session:
                return session.query(RevenueTransaction).filter_by(
                    user_id=user_id
                ).order_by(desc(RevenueTransaction.created_at)).limit(limit).offset(offset).all()
        except Exception as e:
            self.logger.error(f"Failed to get transactions for user {user_id}: {e}")
            return []

    def get_revenue_metrics(self, start_date: datetime, end_date: datetime,
                           revenue_type: RevenueType = None) -> Dict:
        """Get revenue metrics for a date range"""
        try:
            with db_manager.get_session() as session:
                query = session.query(
                    func.sum(RevenueTransaction.amount).label('total_amount'),
                    func.sum(RevenueTransaction.fee_amount).label('total_fees'),
                    func.sum(RevenueTransaction.tax_amount).label('total_taxes'),
                    func.sum(RevenueTransaction.net_amount).label('net_revenue'),
                    func.count(RevenueTransaction.id).label('transaction_count'),
                    func.avg(RevenueTransaction.amount).label('avg_transaction')
                ).filter(
                    and_(
                        RevenueTransaction.processing_date >= start_date,
                        RevenueTransaction.processing_date <= end_date,
                        RevenueTransaction.status == TransactionStatus.COMPLETED
                    )
                )

                if revenue_type:
                    query = query.filter(RevenueTransaction.revenue_type == revenue_type)

                result = query.first()

                return {
                    'total_amount': float(result.total_amount or 0),
                    'total_fees': float(result.total_fees or 0),
                    'total_taxes': float(result.total_taxes or 0),
                    'net_revenue': float(result.net_revenue or 0),
                    'transaction_count': int(result.transaction_count or 0),
                    'average_transaction_value': float(result.avg_transaction or 0),
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }

        except Exception as e:
            self.logger.error(f"Failed to get revenue metrics: {e}")
            return {}

    def update_daily_metrics(self, date: datetime = None) -> bool:
        """Update daily revenue metrics aggregation"""
        try:
            target_date = date or datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

            with db_manager.get_session() as session:
                # Calculate metrics for each revenue type
                for revenue_type in RevenueType:
                    # Get daily transactions
                    transactions = session.query(RevenueTransaction).filter(
                        and_(
                            func.date(RevenueTransaction.processing_date) == target_date.date(),
                            RevenueTransaction.revenue_type == revenue_type,
                            RevenueTransaction.status == TransactionStatus.COMPLETED
                        )
                    ).all()

                    if not transactions:
                        continue

                    # Calculate aggregated metrics
                    total_amount = sum(t.amount for t in transactions)
                    total_fees = sum(t.fee_amount for t in transactions)
                    total_taxes = sum(t.tax_amount for t in transactions)
                    net_revenue = sum(t.net_amount for t in transactions)
                    transaction_count = len(transactions)
                    successful_transactions = len([t for t in transactions if t.status == TransactionStatus.COMPLETED])
                    failed_transactions = transaction_count - successful_transactions
                    avg_transaction_value = total_amount / transaction_count if transaction_count > 0 else 0

                    # Create or update metrics record
                    existing_metric = session.query(RevenueMetrics).filter_by(
                        date=target_date,
                        revenue_type=revenue_type
                    ).first()

                    if existing_metric:
                        existing_metric.total_amount = total_amount
                        existing_metric.total_fees = total_fees
                        existing_metric.total_taxes = total_taxes
                        existing_metric.net_revenue = net_revenue
                        existing_metric.transaction_count = transaction_count
                        existing_metric.successful_transactions = successful_transactions
                        existing_metric.failed_transactions = failed_transactions
                        existing_metric.average_transaction_value = avg_transaction_value
                        existing_metric.updated_at = datetime.now(timezone.utc)
                    else:
                        metric = RevenueMetrics(
                            date=target_date,
                            revenue_type=revenue_type,
                            total_amount=total_amount,
                            total_fees=total_fees,
                            total_taxes=total_taxes,
                            net_revenue=net_revenue,
                            transaction_count=transaction_count,
                            successful_transactions=successful_transactions,
                            failed_transactions=failed_transactions,
                            average_transaction_value=avg_transaction_value
                        )
                        session.add(metric)

                session.commit()

            self.logger.info(f"Updated daily revenue metrics for {target_date.date()}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update daily metrics: {e}")
            return False

    def _calculate_fees_and_taxes(self, revenue_type: RevenueType, amount: float) -> Tuple[float, float]:
        """Calculate fees and taxes based on revenue type"""
        # Fee structure (simplified for demo)
        fee_rates = {
            RevenueType.PURCHASE: 0.029,  # 2.9%
            RevenueType.BILL_PAY: 0.015,  # 1.5%
            RevenueType.SUBSCRIPTION: 0.025,  # 2.5%
            RevenueType.INVESTMENT: 0.005,  # 0.5%
            RevenueType.LOAN: 0.010,  # 1.0%
            RevenueType.INSURANCE: 0.020,  # 2.0%
            RevenueType.OTHER: 0.030  # 3.0%
        }

        # Tax rates (simplified)
        tax_rates = {
            RevenueType.PURCHASE: 0.08,  # 8% sales tax
            RevenueType.BILL_PAY: 0.00,  # No tax
            RevenueType.SUBSCRIPTION: 0.08,  # 8% tax
            RevenueType.INVESTMENT: 0.00,  # No tax
            RevenueType.LOAN: 0.00,  # No tax
            RevenueType.INSURANCE: 0.08,  # 8% tax
            RevenueType.OTHER: 0.08  # 8% tax
        }

        fee_rate = fee_rates.get(revenue_type, 0.03)
        tax_rate = tax_rates.get(revenue_type, 0.08)

        fee_amount = amount * fee_rate
        tax_amount = amount * tax_rate

        return round(fee_amount, 2), round(tax_amount, 2)

# Global revenue service instance
revenue_service = RevenueService()
