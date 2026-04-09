import pytest
from unittest.mock import patch, MagicMock
from jpmorgan_financial_apis.blueprints.pfm import pfm_bp, categorize_transaction, calculate_budget_spent

class TestPFMBlueprints:
    @patch('jpmorgan_financial_apis.blueprints.pfm._mock_accounts')
    def test_link_financial_account(self, mock_accounts, pfm_client):
        mock_accounts.__setitem__.return_value = None
        rv = pfm_client.post('/pfm/accounts/link', json={
            'user_id': 'test',
            'institution_id': 'chase',
            'account_type': 'checking',
            'account_name': 'Chase Checking'
        })
        assert rv.status_code == 200
        assert rv.json['status'] == 'success'

    @patch('jpmorgan_financial_apis.blueprints.pfm._mock_accounts')
    @patch('jpmorgan_financial_apis.blueprints.pfm._mock_budgets')
    def test_create_budget(self, mock_budgets, mock_accounts, pfm_client):
        rv = pfm_client.post('/pfm/budgets', json={
            'user_id': 'test',
            'name': 'Groceries',
            'category': 'groceries',
            'amount': 400.0,
            'period': 'monthly'
        })
        assert rv.status_code == 201
        assert rv.json['status'] == 'success'

    @patch('jpmorgan_financial_apis.blueprints.pfm._mock_accounts')
    def test_get_linked_accounts(self, mock_accounts, pfm_client):
        mock_accounts.get.return_value = [{
            'account_id': 'acc1',
            'account_type': 'checking',
            'account_name': 'Primary Checking',
            'institution_id': 'chase',
            'balance': 1000
        }]
        rv = pfm_client.get('/pfm/accounts?user_id=test')
        assert rv.status_code == 200
        assert rv.json['status'] == 'success'
        assert len(rv.json['accounts']) == 1

    def test_categorize_transaction_groceries(self):
        assert categorize_transaction('Grocery Store') == 'groceries'

    def test_categorize_transaction_dining(self):
        assert categorize_transaction('Starbucks') == 'dining'

    def test_calculate_budget_spent(self):
        assert calculate_budget_spent('test', 'groceries') >= 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
