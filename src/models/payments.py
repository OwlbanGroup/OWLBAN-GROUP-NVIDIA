"""
Payment Models for JPMorgan Financial APIs
Defines data models for payment transactions and related entities.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class PaymentStatus(str, Enum):
    """Payment status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentType(str, Enum):
    """Payment type enumeration"""
    ACH = "ach"
    CARD = "card"
    WALLET = "wallet"
    WIRE = "wire"
    CHECK = "check"


class Payment(Base):
    """
    Payment transaction model
    """
    __tablename__ = 'payments'

    id = Column(String(36), primary_key=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    payment_type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default=PaymentStatus.PENDING.value)
    user_id = Column(String(36), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime(timezone=True))
    processing_time_ms = Column(Float)
    error_code = Column(String(50))
    payment_metadata = Column(JSON, default=dict)

    def __init__(self, id: str, amount: float, currency: str, payment_type: PaymentType,
                 status: PaymentStatus, user_id: str, description: str = "",
                 created_at: Optional[datetime] = None, updated_at: Optional[datetime] = None,
                 extra_metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.amount = amount
        self.currency = currency
        self.payment_type = payment_type.value
        self.status = status.value
        self.user_id = user_id
        self.description = description
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.payment_metadata = extra_metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert payment object to dictionary

        Returns:
            Dict representation of the payment
        """
        return {
            'id': self.id,
            'amount': self.amount,
            'currency': self.currency,
            'payment_type': self.payment_type,
            'status': self.status,
            'user_id': self.user_id,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processing_time_ms': self.processing_time_ms,
            'error_code': self.error_code,
            'metadata': self.payment_metadata
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Payment':
        """
        Create payment object from dictionary

        Args:
            data: Dictionary containing payment data

        Returns:
            Payment object
        """
        return cls(
            id=data['id'],
            amount=data['amount'],
            currency=data.get('currency', 'USD'),
            payment_type=PaymentType(data['payment_type']),
            status=PaymentStatus(data['status']),
            user_id=data['user_id'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            extra_metadata=data.get('metadata', {})
        )


class PaymentMethod(Base):
    """
    Payment method model for storing user payment methods
    """
    __tablename__ = 'payment_methods'

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    type = Column(String(20), nullable=False)  # card, bank_account, wallet
    provider = Column(String(50))  # visa, mastercard, paypal, etc.
    last_four = Column(String(4))  # Last 4 digits for display
    is_default = Column(Integer, default=0)  # 1 for default, 0 otherwise
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    extra_metadata = Column(JSON, default=dict)  # Encrypted sensitive data

    def __init__(self, id: str, user_id: str, type: str, provider: str = "",
                 last_four: str = "", is_default: bool = False, is_active: bool = True,
                 extra_metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.user_id = user_id
        self.type = type
        self.provider = provider
        self.last_four = last_four
        self.is_default = 1 if is_default else 0
        self.is_active = 1 if is_active else 0
        self.extra_metadata = extra_metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert payment method object to dictionary

        Returns:
            Dict representation of the payment method
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'type': self.type,
            'provider': self.provider,
            'last_four': self.last_four,
            'is_default': bool(self.is_default),
            'is_active': bool(self.is_active),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.extra_metadata
        }


class TransactionFee(Base):
    """
    Transaction fee model for tracking fees associated with payments
    """
    __tablename__ = 'transaction_fees'

    id = Column(String(36), primary_key=True)
    payment_id = Column(String(36), ForeignKey('payments.id'), nullable=False)
    fee_type = Column(String(50), nullable=False)  # processing_fee, interchange_fee, etc.
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    extra_metadata = Column(JSON, default=dict)

    def __init__(self, id: str, payment_id: str, fee_type: str, amount: float,
                 currency: str = "USD", description: str = "", extra_metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.payment_id = payment_id
        self.fee_type = fee_type
        self.amount = amount
        self.currency = currency
        self.description = description
        self.extra_metadata = extra_metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transaction fee object to dictionary

        Returns:
            Dict representation of the transaction fee
        """
        return {
            'id': self.id,
            'payment_id': self.payment_id,
            'fee_type': self.fee_type,
            'amount': self.amount,
            'currency': self.currency,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.extra_metadata
        }
