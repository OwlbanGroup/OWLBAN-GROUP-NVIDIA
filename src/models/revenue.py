Integrations

On
ACHQ
Onboard ACH customers and get paid faster with ACHQ's API plus Plaid

View documentation


On
Roll by ADP
Authenticate your customers’ bank accounts for secure payroll direct deposit support.

View documentation


On
Adyen
Validate bank accounts and reduce returns through an end-to-end pay-with-your-bank solution with Adyen

View documentation


On
Alloy
Instantly authenticate your customers’ bank accounts with Alloy’s API and third-party data sources

View documentation


On
Alpaca
Use Alpaca with Plaid Auth to send and receive payments

View documentation


On
Ansa
Instantly authenticate your customers' bank accounts for use with the Ansa API to enable wallet funding over ACH

View documentation


On
Apex Fintech Solutions
Instantly authenticate your investors’ bank accounts for use with Apex Fintech Solutions Cash API

View documentation


On
Astra
Instantly authenticate your customer’s bank accounts for automated ACH transfers through the Astra platform

View documentation


On
Atomic
Instantly authenticate your customers' bank accounts to seamlessly fund investment accounts

View documentation


On
Bakkt
Instantly authenticate your customer’s accounts to use with Bakkt Fiat Services API for ACH based money movement

View documentation


On
Bond
Instantly authenticate your customers’ bank accounts for use with Bond’s ACH Transfers API

View documentation


On
Boom
Report your customers’ rent payments to Experian, Equifax, and TransUnion using the bank account they've linked via Plaid in your platform

View documentation


On
Cardlytics
Turn your customers' everyday purchases into personalized cashback rewards with Cardlytics Rewards Platform.

View documentation


On
Check
Check lets you embed payroll in your product and easily configure and authenticate direct deposit payments

View documentation


On
Checkbook
Instantly authenticate your customers’ bank accounts for use with Checkbook’s payment solution -- including ACH, real-time payments, push to card, virtual cards, and checks

View documentation


On
Checkout.com
Instantly authenticate bank account details for use with Checkout.com’s Unified Payments API, and unlock unrivaled payment performance

View documentation


On
DriveWealth
Enable your customers to instantly and securely fund DriveWealth supported investment accounts by linking their bank account with Plaid

View documentation


On
Dwolla
Instantly authenticate your customers' bank accounts for use with Dwolla's ACH API

View documentation


On
Esusu
Report positive-only rent payments to Experian, Equifax, and TransUnion with Esusu and Plaid

View documentation


On
Finix
Instantly add customer bank accounts to Finix to accept and send payments with tokenized payment information. It’s fast, frictionless, and secure.

View documentation


On
Fortress Trust
Instantly allow end customers to connect, verify and authorize funding to their Fortress Trust account from their external bank accounts

View documentation


On
Gainbridge
Instantly authenticate your customers' bank accounts to fund policies through Gainbridge's Bank Accounts API

View documentation


On
Galileo
Instantly authenticate your customers' bank accounts for account opening and funding

View documentation


On
Gusto
Build payroll with Gusto, then use Plaid Auth to instantly connect your customers’ bank accounts and run payroll faster

View documentation


On
Highnote
Instantly authenticate your customer’s bank accounts for use with the Highnote Platform to store account details, transfer funds, and make payments

View documentation


On
Knot
Instantly make your card top-of-wallet at merchants with Knot’s API

View documentation


On
Layer
Offer SMB accounting embedded directly within your platform.

View documentation


On
Marqeta
Integrate Plaid and Marqeta’s APIs to seamlessly authenticate your customer’s bank account prior to an ACH transfer

View documentation


On
Modern Treasury
Instantly authenticate your customers' bank accounts for use with Modern Treasury's ACH API

View documentation


On
Moov
Instantly authenticate your customers’ bank accounts to enable them to accept, store, and disburse funds with Moov

View documentation


On
Ocrolus
Receive digitized bank data through your Ocrolus API integration

View documentation


On
Open Ledger
Connect your customer's bankaccount to Open Ledger's embedded accounting API

View documentation


On
Paynote
Enhance your checkout and reduce returns with Paynote ACH. Utilize Plaid’s verification and real-time balance checks to instantly debit and credit bank accounts.

View documentation


On
Pinwheel
Activate new accounts with the industry’s top performing direct deposit and bill switching solution, The Pinwheel Switch Kit.

View documentation


On
Lithic
Instant bank authentication for ACH payments and card loads

View documentation


On
Riskified
Provide fraud screening and ACH payment guarantee

View documentation


On
Rize
Use Plaid to instantly verify and connect your customers’ external bank accounts for use within the Rize Platform

View documentation


On
Sardine
Protect your users by using Sardine’s all-in-one fraud and compliance API

View documentation


On
ScribeUp
Embed subscription management into your digital banking experience

View documentation


On
sFox
Use sFOX Connect with Plaid to link and verify bank accounts for crypto payments and trading.

View documentation


On
Sila Money
Banking, Digital Wallets, and ACH Payments API for software teams

View documentation


On
Solid
Instantly authenticate external bank accounts for use with the Solid Platform

View documentation


On
Silicon Valley Bank

"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
import enum

from .base import Base

class RevenueType(enum.Enum):
    """Types of revenue transactions"""
    PURCHASE = "purchase"
    BILL_PAY = "bill_pay"
    SUBSCRIPTION = "subscription"
    INVESTMENT = "investment"
    LOAN = "loan"
    INSURANCE = "insurance"
    PAYROLL = "payroll"
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
    __table_args__ = {'extend_existing': True}

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
    __table_args__ = {'extend_existing': True}

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
