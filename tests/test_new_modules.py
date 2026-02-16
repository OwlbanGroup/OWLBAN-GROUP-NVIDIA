"""
Unit tests for Phase 3-7 modules
"""

import os
import sys
import json
import pytest
from datetime import datetime, timezone, timedelta

# Set testing environment
os.environ['TESTING'] = '1'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataImporter:
    """Tests for data_importer.py"""
    
    def test_import_json_single_record(self):
        """Test importing a single JSON record"""
        from src.data_importer import DataImporter
        
        importer = DataImporter()
        
        json_data = json.dumps({
            'username': 'testuser',
            'email': 'test@example.com',
            'phone': '555-1234'
        })
        
        result = importer.import_from_json(json_data, 'user')
        
        assert result['status'] == 'success'
        assert result['imported_count'] == 1
    
    def test_import_json_multiple_records(self):
        """Test importing multiple JSON records"""
        from src.data_importer import DataImporter
        
        importer = DataImporter()
        
        json_data = json.dumps([
            {'username': 'user1', 'email': 'user1@example.com'},
            {'username': 'user2', 'email': 'user2@example.com'}
        ])
        
        result = importer.import_from_json(json_data, 'user')
        
        assert result['status'] == 'success'
        assert result['imported_count'] == 2
    
    def test_import_csv(self):
        """Test importing CSV data"""
        from src.data_importer import DataImporter
        
        importer = DataImporter()
        
        csv_data = "username,email\nuser1,user1@example.com\nuser2,user2@example.com"
        
        result = importer.import_from_csv(csv_data, 'user')
        
        assert result['status'] == 'success'
        assert result['imported_count'] == 2
    
    def test_validation_error(self):
        """Test validation error handling"""
        from src.data_importer import DataImporter
        
        importer = DataImporter()
        
        # Missing required email field
        json_data = json.dumps({'username': 'testuser'})
        
        result = importer.import_from_json(json_data, 'user')
        
        assert result['failed_count'] == 1
    
    def test_invalid_json(self):
        """Test invalid JSON handling"""
        from src.data_importer import DataImporter
        
        importer = DataImporter()
        
        result = importer.import_from_json('invalid json', 'user')
        
        assert result['status'] == 'error'
        assert 'Invalid JSON' in result['message']


class TestBankingModels:
    """Tests for banking_data_models.py"""
    
    def test_account_type_enum(self):
        """Test AccountType enum"""
        from src.banking_data_models import AccountType
        
        assert AccountType.CHECKING.value == 'checking'
        assert AccountType.SAVINGS.value == 'savings'
        assert AccountType.CREDIT.value == 'credit'
    
    def test_transaction_status_enum(self):
        """Test TransactionStatus enum"""
        from src.banking_data_models import TransactionStatus
        
        assert TransactionStatus.PENDING.value == 'pending'
        assert TransactionStatus.COMPLETED.value == 'completed'
        assert TransactionStatus.FAILED.value == 'failed'
    
    def test_loan_status_enum(self):
        """Test LoanStatus enum"""
        from src.banking_data_models import LoanStatus
        
        assert LoanStatus.APPLICATION.value == 'application'
        assert LoanStatus.APPROVED.value == 'approved'
        assert LoanStatus.ACTIVE.value == 'active'


class TestPayrollService:
    """Tests for payroll_service.py"""
    
    def test_create_employee(self):
        """Test creating an employee"""
        from src.payroll_service import payroll_service
        
        employee_data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john.doe@company.com',
            'salary': 75000,
            'pay_frequency': 'biweekly'
        }
        
        result = payroll_service.create_employee('test_user', employee_data)
        
        assert result['status'] == 'success'
        assert result['employee']['first_name'] == 'John'
        assert 'employee_id' in result['employee']
    
    def test_list_employees(self):
        """Test listing employees"""
        from src.payroll_service import payroll_service
        
        # Create an employee first
        employee_data = {
            'first_name': 'Jane',
            'last_name': 'Smith',
            'email': 'jane@company.com',
            'salary': 65000
        }
        payroll_service.create_employee('test_user', employee_data)
        
        result = payroll_service.list_employees('test_user')
        
        assert result['status'] == 'success'
        assert result['count'] >= 1
    
    def test_create_payroll_run(self):
        """Test creating a payroll run"""
        from src.payroll_service import payroll_service
        
        run_data = {
            'pay_period_start': '2024-01-01',
            'pay_period_end': '2024-01-15',
            'payment_date': '2024-01-15'
        }
        
        result = payroll_service.create_payroll_run('test_user', run_data)
        
        assert result['status'] == 'success'
        assert 'run_id' in result['run']
    
    def test_calculate_employee_taxes(self):
        """Test tax calculation"""
        from src.payroll_service import TaxCalculator
        
        taxes = TaxCalculator.calculate_all_taxes(75000, 0.05)
        
        assert 'federal_tax' in taxes
        assert 'state_tax' in taxes
        assert 'social_security' in taxes
        assert 'medicare' in taxes
        assert taxes['federal_tax'] > 0


class TestLoansBlueprint:
    """Tests for loans blueprint"""

    @pytest.mark.skip(reason="Loans blueprint not implemented yet")
    def test_create_loan(self):
        """Test creating a loan"""
        pass

    @pytest.mark.skip(reason="Loans blueprint not implemented yet")
    def test_list_loans(self):
        """Test listing loans"""
        pass


class TestCreditBlueprint:
    """Tests for credit blueprint"""

    @pytest.mark.skip(reason="Credit blueprint not implemented yet")
    def test_create_card(self):
        """Test creating a credit card"""
        pass

    @pytest.mark.skip(reason="Credit blueprint not implemented yet")
    def test_get_cards_by_user(self):
        """Test getting cards by user"""
        pass


class TestTransfersBlueprint:
    """Tests for transfers blueprint"""

    @pytest.mark.skip(reason="Transfers blueprint not implemented yet")
    def test_create_transfer(self):
        """Test creating a transfer"""
        pass

    @pytest.mark.skip(reason="Transfers blueprint not implemented yet")
    def test_get_fees(self):
        """Test getting transfer fees"""
        pass


class TestStatementsBlueprint:
    """Tests for statements blueprint"""

    @pytest.mark.skip(reason="Statements blueprint not implemented yet")
    def test_create_statement(self):
        """Test creating a statement"""
        pass


class TestMFAService:
    """Tests for mfa_service.py"""

    @pytest.mark.skip(reason="pyotp module not installed")
    def test_setup_totp(self):
        """Test setting up TOTP MFA"""
        pass

    @pytest.mark.skip(reason="pyotp module not installed")
    def test_setup_sms(self):
        """Test setting up SMS MFA"""
        pass

    @pytest.mark.skip(reason="pyotp module not installed")
    def test_invalid_mfa_method(self):
        """Test invalid MFA method"""
        pass

    @pytest.mark.skip(reason="pyotp module not installed")
    def test_get_mfa_status_not_setup(self):
        """Test getting MFA status when not setup"""
        pass


class TestDelegationService:
    """Tests for account_delegation.py"""
    
    def test_request_delegation(self):
        """Test requesting account delegation"""
        from src.account_delegation import delegation_service
        
        result = delegation_service.request_delegation(
            grantor_id='owner123',
            grantee_id='user456',
            account_id='ACC001',
            permissions='view'
        )
        
        assert result['status'] == 'success'
        assert 'request' in result
    
    def test_invalid_permissions(self):
        """Test invalid permissions"""
        from src.account_delegation import delegation_service
        
        result = delegation_service.request_delegation(
            grantor_id='owner123',
            grantee_id='user456',
            account_id='ACC001',
            permissions='invalid'
        )
        
        assert result['status'] == 'error'
    
    def test_check_access_no_delegation(self):
        """Test checking access with no delegation"""
        from src.account_delegation import delegation_service
        
        result = delegation_service.check_access('user123', 'ACC001', 'read_account')
        
        assert result['status'] == 'error'
        assert result['has_access'] is False


class TestBackupService:
    """Tests for backup_recovery.py"""
    
    def test_create_backup(self):
        """Test creating a backup"""
        from src.backup_recovery import backup_service
        
        test_data = {
            'users': [{'id': 1, 'name': 'Test User'}],
            'accounts': [{'id': 1, 'balance': 1000}]
        }
        
        result = backup_service.create_backup('test_user', test_data, 'full')
        
        assert result['status'] == 'success'
        assert 'backup_id' in result['backup']
        assert result['backup']['file_size'] > 0
    
    def test_verify_backup(self):
        """Test verifying a backup"""
        from src.backup_recovery import backup_service
        
        test_data = {'test': 'data'}
        create_result = backup_service.create_backup('test_user', test_data)
        backup_id = create_result['backup']['backup_id']
        
        verify_result = backup_service.verify_backup(backup_id)
        
        assert verify_result['status'] == 'success'
    
    def test_list_backups(self):
        """Test listing backups"""
        from src.backup_recovery import backup_service
        
        # Create a backup first
        backup_service.create_backup('test_user', {'data': 'test'})
        
        result = backup_service.list_backups('test_user')
        
        assert result['status'] == 'success'
        assert result['count'] >= 1
    
    def test_delete_backup(self):
        """Test deleting a backup"""
        from src.backup_recovery import backup_service
        
        # Create a backup
        create_result = backup_service.create_backup('test_user', {'data': 'test'})
        backup_id = create_result['backup']['backup_id']
        
        # Delete it
        delete_result = backup_service.delete_backup(backup_id)
        
        assert delete_result['status'] == 'success'
        
        # Verify it's gone
        get_result = backup_service.store.get_backup(backup_id)
        assert get_result is None


class TestTaxCalculator:
    """Tests for tax calculations"""
    
    def test_federal_tax_bracket(self):
        """Test federal tax bracket calculation"""
        from src.payroll_service import TaxCalculator
        
        tax = TaxCalculator.calculate_federal_tax(50000)
        
        assert tax > 0
        assert tax < 50000  # Should be less than income
    
    def test_state_tax(self):
        """Test state tax calculation"""
        from src.payroll_service import TaxCalculator
        
        tax = TaxCalculator.calculate_state_tax(50000, 0.05)
        
        assert tax == 2500  # 5% of 50000
    
    def test_social_security(self):
        """Test Social Security calculation"""
        from src.payroll_service import TaxCalculator
        
        ss_tax = TaxCalculator.calculate_social_security(100000)
        
        assert ss_tax > 0
        assert ss_tax <= 100000 * 0.062  # Should not exceed max rate
    
    def test_medicare(self):
        """Test Medicare calculation"""
        from src.payroll_service import TaxCalculator
        
        medicare = TaxCalculator.calculate_medicare(100000)
        
        assert medicare > 0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
