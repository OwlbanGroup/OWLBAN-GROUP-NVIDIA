"""
Revenue tracking models for JPMorgan Financial APIs
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class RevenueType(enum.Enum):
    """Types of revenue transactions"""
    PURCHASE = "purchase"
    BILL_PAY = "bill_pay"
    SUBSCRIPTION = "subscription"
    INVESTMENT = "investment"
    LOAN = "loan"
    INSURANCE = "insurance"
    OTHER = "other"

class TransactionStatus(enum.Enum):
    """Transaction status"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class RevenueTransaction(Base):
    """Revenue transaction model"""
    __tablename__ = 'revenue_transactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    business_id = Column(Integer, ForeignKey('businesses.id'), nullable=True)

    revenue_type = Column(Enum(RevenueType), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default='USD')
    status = Column(Enum(TransactionStatus), default=TransactionStatus.PENDING)

    # Transaction details
    description = Column(Text)
    merchant_name = Column(String(200))
    category = Column(String(100))

    # Financial details
    fee_amount = Column(Float, default=0.0)
    tax_amount = Column(Float, default=0.0)
    net_amount = Column(Float, nullable=False)  # amount - fees - taxes

    # Processing details
    processing_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    settlement_date = Column(DateTime(timezone=True), nullable=True)
    payment_method = Column(String(50))

    # Metadata
    source_system = Column(String(100), default='api')
    external_reference = Column(String(200))
    additional_data = Column(Text)  # JSON string for additional data

    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))



    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'user_id': self.user_id,
            'business_id': self.business_id,
            'revenue_type': self.revenue_type.value,
            'amount': self.amount,
            'currency': self.currency,
            'status': self.status.value,
            'description': self.description,
            'merchant_name': self.merchant_name,
            'category': self.category,
            'fee_amount': self.fee_amount,
            'tax_amount': self.tax_amount,
            'net_amount': self.net_amount,
            'processing_date': self.processing_date.isoformat() if self.processing_date else None,
            'settlement_date': self.settlement_date.isoformat() if self.settlement_date else None,
            'payment_method': self.payment_method,
            'source_system': self.source_system,
            'external_reference': self.external_reference,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class RevenueMetrics(Base):
    """Daily revenue metrics aggregation"""
    __tablename__ = 'revenue_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    revenue_type = Column(Enum(RevenueType), nullable=False)

    # Aggregated amounts
    total_amount = Column(Float, default=0.0)
    total_fees = Column(Float, default=0.0)
    total_taxes = Column(Float, default=0.0)
    net_revenue = Column(Float, default=0.0)

    # Transaction counts
    transaction_count = Column(Integer, default=0)
    successful_transactions = Column(Integer, default=0)
    failed_transactions = Column(Integer, default=0)

    # Performance metrics
    average_transaction_value = Column(Float, default=0.0)
    processing_time_avg = Column(Float, default=0.0)  # in milliseconds

    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'revenue_type': self.revenue_type.value,
            'total_amount': self.total_amount,
            'total_fees': self.total_fees,
            'total_taxes': self.total_taxes,
            'net_revenue': self.net_revenue,
            'transaction_count': self.transaction_count,
            'successful_transactions': self.successful_transactions,
            'failed_transactions': self.failed_transactions,
            'average_transaction_value': self.average_transaction_value,
            'processing_time_avg': self.processing_time_avg,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
