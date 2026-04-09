"""Banking Service for JPMorgan Financial APIs
CRUD operations for bank accounts and transactions with validation and ACID compliance."""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import uuid
from src.database_fixed import db_manager
from src.banking_data_models import BankAccountModel, TransactionModel, AccountType, TransactionType
from src.transaction_manager import transaction_manager
from src.logger import telemetry_logger

class BankingService:
    def __init__(self):
        self.logger = telemetry_logger.get_logger()

    def create_account(self, account_data: Dict[str, Any], session: Session = None) -> BankAccountModel:
        """Create new bank account
        account_data: dict with user_id, account_type ('checking'|'savings'), initial_balance=0.0
        """
        def operation(session: Session):
            # Generate unique account number
            account_number = f"ACC{uuid.uuid4().hex[:12].upper()}"
            
            account = BankAccountModel(
                account_number=account_number,
                **account_data,
                status='active',
                balance=account_data.get('initial_balance', 0.0),
                available_balance=account_data.get('initial_balance', 0.0),
                opened_at=datetime.now(timezone.utc)
            )
            session.add(account)
            session.flush()  # Flush to get ID
            session.commit()
            self.logger.info(f"Created account {account.id} for user {account.user_id}")
            return account

        return transaction_manager.execute_with_retry(operation)

    def get_accounts(self, user_id: str, session: Optional[Session] = None) -> List[BankAccountModel]:
        """Get all accounts for user"""
        def operation(session: Session):
            accounts = session.query(BankAccountModel).filter(
                BankAccountModel.user_id == user_id,
                BankAccountModel.status == 'active'
            ).order_by(BankAccountModel.created_at.desc()).all()
            return accounts

        return transaction_manager.execute_with_retry(operation)

    def get_account(self, account_id: int, user_id: str, session: Optional[Session] = None) -> Optional[BankAccountModel]:
        """Get specific account by ID, verify ownership"""
        def operation(session: Session):
            account = session.query(BankAccountModel).filter(
                BankAccountModel.id == account_id,
                BankAccountModel.user_id == user_id
            ).first()
            return account

        return transaction_manager.execute_with_retry(operation)

    def update_account(self, account_id: int, user_id: str, updates: Dict[str, Any], session: Optional[Session] = None) -> Optional[BankAccountModel]:
        """Update account details"""
        allowed_updates = ['interest_rate', 'overdraft_limit', 'monthly_fee']
        def operation(session: Session):
            account = session.query(BankAccountModel).filter(
                BankAccountModel.id == account_id,
                BankAccountModel.user_id == user_id
            ).first()
            if not account:
                raise ValueError(f"Account {account_id} not found for user {user_id}")
            for key, value in updates.items():
                if key in allowed_updates:
                    setattr(account, key, value)
            session.commit()
            session.refresh(account)
            return account

        return transaction_manager.execute_with_retry(operation)

    def validate_account(self, account_id: int, user_id: str, min_balance: float = 0.0, session: Optional[Session] = None) -> Dict[str, Any]:
        """Validate account status and balance"""
        def operation(session: Session):
            account = session.query(BankAccountModel).filter(
                BankAccountModel.id == account_id,
                BankAccountModel.user_id == user_id
            ).first()
            if not account:
                raise ValueError(f"Account {account_id} not found")
            if account.status != 'active':
                raise ValueError(f"Account inactive: {account.status}")
            if account.available_balance < min_balance:
                raise ValueError(f"Insufficient balance: {account.available_balance} < {min_balance}")
            return {
                'valid': True,
                'account_id': account.id,
                'balance': account.balance,
                'available_balance': account.available_balance,
                'status': account.status
            }

        return transaction_manager.execute_with_retry(operation)

    def create_transaction(self, account_id: int, user_id: str, tx_type: str, amount: float, description: str = "", session: Optional[Session] = None) -> TransactionModel:
        """Create transaction (deposit/withdrawal/transfer), update balance"""
        if tx_type not in ['deposit', 'withdrawal', 'transfer']:
            raise ValueError(f"Invalid tx_type: {tx_type}")
        
        def operation(session: Session):
            account = session.query(BankAccountModel).filter(
                BankAccountModel.id == account_id,
                BankAccountModel.user_id == user_id
            ).first()
            if not account:
                raise ValueError(f"Account not found")
            
            # Validate for withdrawal
            if tx_type == 'withdrawal' and account.available_balance < amount:
                raise ValueError(f"Insufficient funds: {account.available_balance} < {amount}")
            
            # Create transaction
            transaction_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
            is_credit = tx_type == 'deposit'
            new_balance = account.balance + (amount if is_credit else -amount)
            new_available = account.available_balance + (amount if is_credit else -amount)
            
            tx = TransactionModel(
                transaction_id=transaction_id,
                account_id=account_id,
                user_id=user_id,
                transaction_type=tx_type,
                amount=abs(amount),
                balance_after=new_balance,
                description=description,
                status='completed'
            )
            session.add(tx)
            
            # Update account balance
            account.balance = new_balance
            account.available_balance = new_available
            account.last_transaction_at = datetime.now(timezone.utc)
            
            session.commit()
            self.logger.info(f"Created {tx_type} TXN{tx.id} for account {account_id}: ${amount}")
            return tx

        return transaction_manager.execute_with_retry(operation)

    def get_account_transactions(self, account_id: int, user_id: str, limit: int = 50, session: Optional[Session] = None) -> List[TransactionModel]:
        """Get recent transactions for account"""
        def operation(session: Session):
            txs = session.query(TransactionModel).filter(
                TransactionModel.account_id == account_id,
                TransactionModel.user_id == user_id
            ).order_by(TransactionModel.created_at.desc()).limit(limit).all()
            return txs

        return transaction_manager.execute_with_retry(operation)

# Global service instance
banking_service = BankingService()

