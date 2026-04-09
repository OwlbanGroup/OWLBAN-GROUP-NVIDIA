import pytest
from unittest.mock import Mock, patch
from flask import Flask
from jpmorgan_financial_apis.blueprints.payments import payments_bp

@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(payments_bp)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

class TestPaymentsBlueprint:
    @patch('jpmorgan_financial_apis.blueprints.payments.payments_service')
    def test_add_payment_method(self, mock_service, client):
        mock_service.create_payment.return_value = Mock()
        rv = client.post('/payments/methods', json={
            'type': 'card',
            'provider': 'visa',
            'last_four': '4242'
        })
        assert rv.status_code == 201
        assert rv.json['status'] == 'success'

    @patch('jpmorgan_financial_apis.blueprints.payments.payments_service')
    def test_load_card(self, mock_service, client):
        payment_obj = Mock()
        payment_obj.id = "pay_123"
        mock_service.create_payment.return_value = payment_obj
        mock_service.process_payment.return_value = True
        rv = client.post('/payments/load', json={
            'method_id': 'pm_123',
            'amount': 100.0
        })
        assert rv.status_code == 200
        assert rv.json['status'] == 'success'

    # Add tests for all endpoints to cover missing lines
    def test_process_payment(self):
        # Test logic
        pass

    # Cover all 299 lines by testing each route/method/error case

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
