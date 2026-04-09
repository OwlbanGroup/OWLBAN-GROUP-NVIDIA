import pytest
from unittest.mock import Mock, patch
from jpmorgan_financial_apis.src.payments_service import payments_service

class TestPaymentsService:
    @patch('stripe.PaymentIntent.create')
    def test_create_stripe_payment_intent(self, mock_stripe):
        mock_pi = Mock()
        mock_pi.id = 'pi_test'
        mock_pi.client_secret = 'secret'
        mock_stripe.return_value = mock_pi

        result = payments_service.create_stripe_payment_intent(amount=1000, currency='usd')
        assert result['status'] == 'success'
        assert result['payment_intent_id'] == 'pi_test'

    @patch('stripe.PaymentIntent.retrieve')
    def test_confirm_stripe_payment(self, mock_retrieve):
        mock_pi = Mock()
        mock_pi.status = 'succeeded'
        mock_retrieve.return_value = mock_pi

        result = payments_service.confirm_stripe_payment('pi_test')
        assert result['status'] == 'success'
        assert mock_pi.status == 'succeeded'

    # Add tests for all methods to cover payments_service.py lines

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src/payments_service'])

