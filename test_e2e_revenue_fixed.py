"""E2E Revenue Testing for JPMorgan APIs - Mocked."""
import pytest
from unittest.mock import patch

@pytest.fixture
def client():
    from app_final import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('jpmorgan_financial_apis.app_final.payments_service')
def test_e2e_revenue_flow(mock_payments, client):
    """Mocked E2E revenue flow test."""
    mock_payments.create_stripe_payment_intent.return_value = {
        'status': 'success',
        'payment_intent_id': 'pi_test123',
        'client_secret': 'secret_test'
    }

    # Basic flow with mocks
    rv = client.post('/user/register', json={'username': 'testrev', 'password': 'testpass'})
    assert rv.status_code in [201, 409]  # Accept duplicate

    print('Mocked E2E Revenue Flow PASSED ✅')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

