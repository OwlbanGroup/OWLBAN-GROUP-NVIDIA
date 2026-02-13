"""
Banking Data Models for JPMorgan Financial APIs
Provides SQLAlchemy models for banking entities.
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

try:
    from src.models.base import Base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()


# =============================================================================
# ENUMS
# =============================================================================

class AccountType(enum.Enum):
    """Enum for account types"""
    CHECKING = "checking"
    SAVINGS = "savings"
    INVESTMENT = "investment"
    CREDIT = "credit"
    LOAN = "loan"
    MONEY_MARKET = "money_market"
    CERTIFICATE_OF_DEPOSIT = "certificate_of_deposit"


class TransactionType(enum.Enum):
    """Enum for transaction types"""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    REFUND = "refund"
    FEE = "fee"
    INTEREST = "interest"
    DIVIDEND = "dividend"


class TransactionStatus(enum.Enum):
    """Enum for transaction status"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REVIEW = "review"


class LoanStatus(enum.Enum):
    """Enum for loan status"""
    APPLICATION = "application"
    APPROVED = "approved"
    ACTIVE = "active"
    PAID_OFF = "paid_off"
    DEFAULTED = "defaulted"
    CANCELLED = "cancelled"


class CreditCardType(enum.Enum):
    """Enum for credit card types"""
    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"
    DISCOVER = "discover"


class CreditCardStatus(enum.Enum):
    """Enum for credit card status"""
    APPLICATION = "application"
    ACTIVE = "active"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PayrollStatus(enum.Enum):
    """Enum for payroll status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# BANKING MODELS
# =============================================================================

class BankAccountModel(Base):
    """SQLAlchemy model for bank accounts"""
    __tablename__ = 'bank_accounts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_number = Column(String(50), unique=True, nullable=False, index=True)
    account_type = Column(String(50), nullable=False)
    user_id = Column(String(100), nullable=False, index=True)
    balance = Column(Float, default=0.0)
    available_balance = Column(Float, default=0.0)
    currency = Column(String(3), default='USD')
    status = Column(String(20), default='active')
    
    # Account details
    branch_code = Column(String(20))
    routing_number = Column(String(20))
    swift_code = Column(String(20))
    iban = Column(String(50))
    
    # Interest and fees
    interest_rate = Column(Float, default=0.0)
    overdraft_limit = Column(Float, default=0.0)
    monthly_fee = Column(Float, default=0.0)
    
    # Timestamps
    opened_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime)
    last_transaction_at = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    transactions = relationship("TransactionModel", back_populates="account", lazy="dynamic")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'account_number': self.account_number,
            'account_type': self.account_type,
            'user_id': self.user_id,
            'balance': self.balance,
            'available_balance': self.available_balance,
            'currency': self.currency,
            'status': self.status,
            'interest_rate': self.interest_rate,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TransactionModel(Base):
    """SQLAlchemy model for transactions"""
    __tablename__ = 'transactions'
    __table_args__ = (
        Index('idx_transaction_account_date', 'account_id', 'created_at'),
        Index('idx_transaction_user_date', 'user_id', 'created_at'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('bank_accounts.id'), nullable=False)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Transaction details
    transaction_type = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default='USD')
    balance_after = Column(Float)
    description = Column(Text)
    category = Column(String(50))
    merchant_name = Column(String(100))
    merchant_category = Column(String(50))
    
    # Status and flags
    status = Column(String(20), default='completed')
    is_credit = Column(Boolean, default=True)
    is_recurring = Column(Boolean, default=False)
    
    # Reference numbers
    reference_number = Column(String(100))
    check_number = Column(String(50))
    reconciliation_id = Column(String(100))
    
    # timestamps
    transaction_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    posted_date = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    account = relationship("BankAccountModel", back_populates="transactions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'account_id': self.account_id,
            'user_id': self.user_id,
            'transaction_type': self.transaction_type,
            'amount': self.amount,
            'currency': self.currency,
            'balance_after': self.balance_after,
            'description': self.description,
            'category': self.category,
            'status': self.status,
            'is_credit': self.is_credit,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class LoanModel(Base):
    """SQLAlchemy model for loans"""
    __tablename__ = 'loans'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    loan_number = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('bank_accounts.id'))
    
    # Loan details
    loan_type = Column(String(50), nullable=False)  # personal, mortgage, auto, student
    principal_amount = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    term_months = Column(Integer, nullable=False)
    
    # Status
    status = Column(String(20), default='application')
    
    # Payment details
    monthly_payment = Column(Float)
    total_interest = Column(Float)
    total_amount = Column(Float)
    remaining_balance = Column(Float)
    next_payment_date = Column(DateTime)
    next_payment_amount = Column(Float)
    
    # Collateral
    collateral_description = Column(Text)
    collateral_value = Column(Float)
    
    # Dates
    application_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    approval_date = Column(DateTime)
    disbursement_date = Column(DateTime)
    maturity_date = Column(DateTime)
    closed_date = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    payments = relationship("LoanPaymentModel", back_populates="loan", lazy="dynamic")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'loan_number': self.loan_number,
            'user_id': self.user_id,
            'loan_type': self.loan_type,
            'principal_amount': self.principal_amount,
            'interest_rate': self.interest_rate,
            'term_months': self.term_months,
            'status': self.status,
            'monthly_payment': self.monthly_payment,
            'remaining_balance': self.remaining_balance,
            'next_payment_date': self.next_payment_date.isoformat() if self.next_payment_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class LoanPaymentModel(Base):
    """SQLAlchemy model for loan payments"""
    __tablename__ = 'loan_payments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String(50), unique=True, nullable=False, index=True)
    loan_id = Column(Integer, ForeignKey('loans.id'), nullable=False)
    
    # Payment details
    payment_number = Column(Integer)
    payment_date = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    principal_amount = Column(Float)
    interest_amount = Column(Float)
    late_fee = Column(Float, default=0.0)
    balance_after = Column(Float)
    
    # Status
    status = Column(String(20), default='completed')
    payment_method = Column(String(20))
    
    # Reference
    transaction_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    loan = relationship("LoanModel", back_populates="payments")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'payment_id': self.payment_id,
            'loan_id': self.loan_id,
            'payment_number': self.payment_number,
            'payment_date': self.payment_date.isoformat() if self.payment_date else None,
            'amount': self.amount,
            'principal_amount': self.principal_amount,
            'interest_amount': self.interest_amount,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class CreditCardModel(Base):
    """SQLAlchemy model for credit cards"""
    __tablename__ = 'credit_cards'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    card_number = Column(String(20), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('bank_accounts.id'))
    
    # Card details
    card_type = Column(String(20), nullable=False)
    card_brand = Column(String(20))
    expiry_month = Column(Integer, nullable=False)
    expiry_year = Column(Integer, nullable=False)
    cvv = Column(String(4))
    
    # Status and limits
    status = Column(String(20), default='application')
    credit_limit = Column(Float, default=0.0)
    available_credit = Column(Float, default=0.0)
    current_balance = Column(Float, default=0.0)
    
    # Interest and fees
    interest_rate = Column(Float, default=0.0)
    annual_fee = Column(Float, default=0.0)
    
    # Rewards
    reward_points = Column(Integer, default=0)
    cash_back_balance = Column(Float, default=0.0)
    
    # Cardholder info
    cardholder_name = Column(String(100))
    
    # Dates
    issue_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    activation_date = Column(DateTime)
    expiration_date = Column(DateTime)
    closed_date = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    transactions = relationship("CreditCardTransactionModel", back_populates="card", lazy="dynamic")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'card_number': self.card_number[-4:].rjust(len(self.card_number), '*'),  # Mask card number
            'user_id': self.user_id,
            'card_type': self.card_type,
            'expiry_month': self.expiry_month,
            'expiry_year': self.expiry_year,
            'status': self.status,
            'credit_limit': self.credit_limit,
            'available_credit': self.available_credit,
            'current_balance': self.current_balance,
            'reward_points': self.reward_points,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class CreditCardTransactionModel(Base):
    """SQLAlchemy model for credit card transactions"""
    __tablename__ = 'credit_card_transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), unique=True, nullable=False, index=True)
    card_id = Column(Integer, ForeignKey('credit_cards.id'), nullable=False)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Transaction details
    amount = Column(Float, nullable=False)
    description = Column(Text)
    merchant_name = Column(String(100))
    merchant_category = Column(String(50))
    transaction_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Status
    status = Column(String(20), default='pending')
    is_credit = Column(Boolean, default=False)
    
    # Rewards
    reward_points_earned = Column(Integer, default=0)
    cash_back_amount = Column(Float, default=0.0)
    
    # Billing
    billing_cycle = Column(Integer)  # Month
    is_posted = Column(Boolean, default=False)
    posting_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    card = relationship("CreditCardModel", back_populates="transactions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'card_id': self.card_id,
            'user_id': self.user_id,
            'amount': self.amount,
            'description': self.description,
            'merchant_name': self.merchant_name,
            'status': self.status,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TransferModel(Base):
    """SQLAlchemy model for transfers (wire/ACH)"""
    __tablename__ = 'transfers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transfer_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Transfer type
    transfer_type = Column(String(20), nullable=False)  # wire, ach, rtp, internal
    direction = Column(String(10), nullable=False)  # incoming, outgoing
    
    # Accounts
    from_account_id = Column(Integer, ForeignKey('bank_accounts.id'))
    to_account_id = Column(Integer, ForeignKey('bank_accounts.id'))
    from_account_number = Column(String(50))
    to_account_number = Column(String(50))
    from_routing_number = Column(String(20))
    to_routing_number = Column(String(20))
    
    # Amount and fees
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default='USD')
    fee = Column(Float, default=0.0)
    exchange_rate = Column(Float, default=1.0)
    
    # Status
    status = Column(String(20), default='pending')
    
    # Details
    description = Column(Text)
    reference = Column(String(100))
    
    # Processing
    initiated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime)
    completed_at = Column(DateTime)
    failed_at = Column(DateTime)
    failure_reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'transfer_id': self.transfer_id,
            'user_id': self.user_id,
            'transfer_type': self.transfer_type,
            'direction': self.direction,
            'amount': self.amount,
            'currency': self.currency,
            'fee': self.fee,
            'status': self.status,
            'description': self.description,
            'initiated_at': self.initiated_at.isoformat() if self.initiated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class StatementModel(Base):
    """SQLAlchemy model for account statements"""
    __tablename__ = 'statements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    statement_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey('bank_accounts.id'))
    
    # Statement details
    statement_type = Column(String(20), nullable=False)  # monthly, quarterly, annual
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Summary
    opening_balance = Column(Float)
    closing_balance = Column(Float)
    total_credits = Column(Float, default=0.0)
    total_debits = Column(Float, default=0.0)
    total_fees = Column(Float, default=0.0)
    total_interest = Column(Float, default=0.0)
    
    # File
    file_path = Column(String(500))
    file_format = Column(String(10))  # pdf, csv, html
    file_size = Column(Integer)
    
    # Status
    status = Column(String(20), default='generated')
    
    # Timestamps
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'statement_id': self.statement_id,
            'user_id': self.user_id,
            'account_id': self.account_id,
            'statement_type': self.statement_type,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'opening_balance': self.opening_balance,
            'closing_balance': self.closing_balance,
            'total_credits': self.total_credits,
            'total_debits': self.total_debits,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PayrollEmployeeModel(Base):
    """SQLAlchemy model for payroll employees"""
    __tablename__ = 'payroll_employees'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Personal info
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(200), nullable=False)
    phone = Column(String(20))
    
    # Employment details
    department = Column(String(100))
    position = Column(String(100))
    hire_date = Column(DateTime)
    employment_type = Column(String(20))  # full_time, part_time, contractor
    
    # Compensation
    salary = Column(Float)
    hourly_rate = Column(Float)
    pay_frequency = Column(String(20))  # weekly, biweekly, monthly
    
    # Tax info
    tax_filing_status = Column(String(20))
    tax_withholding_rate = Column(Float, default=0.0)
    
    # Direct deposit
    bank_account_id = Column(Integer, ForeignKey('bank_accounts.id'))
    routing_number = Column(String(20))
    account_number = Column(String(50))
    
    # Status
    status = Column(String(20), default='active')
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    payments = relationship("PayrollPaymentModel", back_populates="employee", lazy="dynamic")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'user_id': self.user_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'department': self.department,
            'position': self.position,
            'salary': self.salary,
            'hourly_rate': self.hourly_rate,
            'pay_frequency': self.pay_frequency,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PayrollPaymentModel(Base):
    """SQLAlchemy model for payroll payments"""
    __tablename__ = 'payroll_payments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String(50), unique=True, nullable=False, index=True)
    employee_id = Column(Integer, ForeignKey('payroll_employees.id'), nullable=False)
    payroll_run_id = Column(Integer, ForeignKey('payroll_runs.id'))
    
    # Payment details
    pay_period_start = Column(DateTime, nullable=False)
    pay_period_end = Column(DateTime, nullable=False)
    payment_date = Column(DateTime, nullable=False)
    
    # Compensation
    gross_pay = Column(Float, nullable=False)
    net_pay = Column(Float, nullable=False)
    regular_hours = Column(Float)
    overtime_hours = Column(Float)
    overtime_pay = Column(Float)
    
    # Deductions
    federal_tax = Column(Float, default=0.0)
    state_tax = Column(Float, default=0.0)
    social_security = Column(Float, default=0.0)
    medicare = Column(Float, default=0.0)
    health_insurance = Column(Float, default=0.0)
    retirement_contribution = Column(Float, default=0.0)
    other_deductions = Column(Float, default=0.0)
    
    # Status
    status = Column(String(20), default='pending')
    
    # Reference
    transaction_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    employee = relationship("PayrollEmployeeModel", back_populates="payments")
    payroll_run = relationship("PayrollRunModel", back_populates="payments")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'payment_id': self.payment_id,
            'employee_id': self.employee_id,
            'pay_period_start': self.pay_period_start.isoformat() if self.pay_period_start else None,
            'pay_period_end': self.pay_period_end.isoformat() if self.pay_period_end else None,
            'payment_date': self.payment_date.isoformat() if self.payment_date else None,
            'gross_pay': self.gross_pay,
            'net_pay': self.net_pay,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PayrollRunModel(Base):
    """SQLAlchemy model for payroll runs"""
    __tablename__ = 'payroll_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)  # Employer
    
    # Run details
    pay_period_start = Column(DateTime, nullable=False)
    pay_period_end = Column(DateTime, nullable=False)
    payment_date = Column(DateTime, nullable=False)
    
    # Summary
    total_gross_pay = Column(Float, default=0.0)
    total_net_pay = Column(Float, default=0.0)
    total_deductions = Column(Float, default=0.0)
    total_taxes = Column(Float, default=0.0)
    employee_count = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), default='pending')
    
    # Processing
    initiated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    payments = relationship("PayrollPaymentModel", back_populates="payroll_run", lazy="dynamic")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'user_id': self.user_id,
            'pay_period_start': self.pay_period_start.isoformat() if self.pay_period_start else None,
            'pay_period_end': self.pay_period_end.isoformat() if self.pay_period_end else None,
            'payment_date': self.payment_date.isoformat() if self.payment_date else None,
            'total_gross_pay': self.total_gross_pay,
            'total_net_pay': self.total_net_pay,
            'employee_count': self.employee_count,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# Add index imports at the end
from sqlalchemy import Index

# Create indexes for performance
__table_args__ = (
    Index('idx_transaction_account_date', 'account_id', 'created_at'),
    Index('idx_transaction_user_date', 'user_id', 'created_at'),
)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AccountType',
    'TransactionType',
    'TransactionStatus',
    'LoanStatus',
    'CreditCardType',
    'CreditCardStatus',
    'PayrollStatus',
    'BankAccountModel',
    'TransactionModel',
    'LoanModel',
    'LoanPaymentModel',
    'CreditCardModel',
    'CreditCardTransactionModel',
    'TransferModel',
    'StatementModel',
    'PayrollEmployeeModel',
    'PayrollPaymentModel',
    'PayrollRunModel'
]
