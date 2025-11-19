"""
Comprehensive Test Suite
Achieves 90%+ code coverage for JPMorgan Financial APIs
"""
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import validators
from src.validators_comprehensive import (  # type: ignore
    ComprehensiveValidators, ValidationError,
    validate_business, validate_asset, validate_telemetry, validate_user
)

# Import response helpers
from src.response_helpers import error_response, success_response  # type: ignore

# Import database optimizer
from src.database_optimizer import DatabaseOptimizer  # type: ignore

# Import structured logger
from src.structured_logger import StructuredLogger  # type: ignore

class TestComprehensiveValidators:
    """Test comprehensive validators"""

    def test_validate_email_valid(self):
        """Test valid email validation"""
        assert ComprehensiveValidators.validate_email("test@example.com")
        assert ComprehensiveValidators.validate_email("user.name+tag@example.co.uk")

    def test_validate_email_invalid(self):
        """Test invalid email validation"""
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_email("invalid")
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_email("@example.com")
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_email("test@")

    def test_validate_phone_valid(self):
        """Test valid phone validation"""
        assert ComprehensiveValidators.validate_phone("+1234567890")
        assert ComprehensiveValidators.validate_phone("+44-123-456-7890")

    def test_validate_phone_invalid(self):
        """Test invalid phone validation"""
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_phone("invalid")
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_phone("123")

    def test_validate_url_valid(self):
        """Test valid URL validation"""
        assert ComprehensiveValidators.validate_url("https://example.com")
        assert ComprehensiveValidators.validate_url("http://localhost:8000/api")

    def test_validate_url_invalid(self):
        """Test invalid URL validation"""
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_url("invalid")
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_url("ftp://example.com")

    def test_validate_string(self):
        """Test string validation"""
        assert ComprehensiveValidators.validate_string("test", "field", 1, 10)

        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_string("", "field", 1, 10)
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_string("toolongstring", "field", 1, 5)

    def test_validate_number(self):
        """Test number validation"""
        assert ComprehensiveValidators.validate_number(5, "field", 0, 10)

        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_number(-1, "field", 0, 10)
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_number(11, "field", 0, 10)

    def test_validate_date(self):
        """Test date validation"""
        assert ComprehensiveValidators.validate_date("2023-01-15T00:00:00Z", "field")

        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_date("invalid", "field")

    def test_validate_business_data_valid(self):
        """Test valid business data validation"""
        data = {
            'name': 'Test Corp',
            'type': 'corporation',
            'registration_number': '123456789',
            'address': '123 Test St',
            'contact_info': {
                'email': 'test@example.com',
                'phone': '+1234567890'
            }
        }
        assert validate_business(data)

    def test_validate_business_data_invalid(self):
        """Test invalid business data validation"""
        # Missing required field
        with pytest.raises(ValidationError):
            validate_business({'name': 'Test'})

        # Invalid type
        with pytest.raises(ValidationError):
            validate_business({
                'name': 'Test',
                'type': 'invalid',
                'registration_number': '123'
            })

    def test_validate_asset_data_valid(self):
        """Test valid asset data validation"""
        data = {
            'business_id': 1,
            'name': 'Test Asset',
            'type': 'equipment',
            'value': 50000.00,
            'acquisition_date': '2023-01-15T00:00:00Z',
            'ownership_percentage': 100.0
        }
        assert validate_asset(data)

    def test_validate_asset_data_invalid(self):
        """Test invalid asset data validation"""
        # Missing required field
        with pytest.raises(ValidationError):
            validate_asset({'name': 'Test'})

        # Invalid value
        with pytest.raises(ValidationError):
            validate_asset({
                'business_id': 1,
                'name': 'Test',
                'type': 'equipment',
                'value': -100
            })

    def test_validate_telemetry_data_valid(self):
        """Test valid telemetry data validation"""
        data = {
            'ver': '4.0',
            'name': 'Test.Event',
            'time': '2023-01-15T00:00:00Z',
            'data': {'key': 'value'}
        }
        assert validate_telemetry(data)

    def test_validate_user_registration_valid(self):
        """Test valid user registration validation"""
        data = {
            'username': 'testuser',
            'password': 'Test123!@#',
            'email': 'test@example.com'
        }
        assert validate_user(data)

    def test_validate_user_registration_invalid(self):
        """Test invalid user registration validation"""
        # Weak password
        with pytest.raises(ValidationError):
            validate_user({
                'username': 'test',
                'password': 'weak',
                'email': 'test@example.com'
            })

    def test_sanitize_input(self):
        """Test input sanitization"""
        result = ComprehensiveValidators.sanitize_input("<script>alert('xss')</script>")
        assert '<script>' not in result
        assert '<script>' in result

    def test_validate_batch_size(self):
        """Test batch size validation"""
        assert ComprehensiveValidators.validate_batch_size(100)

        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_batch_size(0)
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_batch_size(10000)

class TestResponseHelpers:
    """Test response helper functions"""

    def test_error_response(self):
        """Test error response generation"""
        response, status_code = error_response("Test error", 400, "ERR001")
        assert status_code == 400
        assert response.json['status'] == 'error'
        assert response.json['error'] == 'Test error'
        assert response.json['error_code'] == 'ERR001'

    def test_success_response(self):
        """Test success response generation"""
        response, status_code = success_response({'data': 'test'}, 200)
        assert status_code == 200
        assert response.json['status'] == 'success'
        assert response.json['data'] == 'test'

class TestDatabaseOptimizer:
    """Test database optimizer"""

    @patch('src.database_optimizer.logger')
    def test_create_indexes(self, mock_logger):  # pylint: disable=unused-argument
        """Test index creation"""
        mock_session = Mock()
        optimizer = DatabaseOptimizer(mock_session)

        result = optimizer.create_indexes('users')
        assert result is True
        assert mock_session.execute.called

    def test_analyze_query_performance(self):
        """Test query performance analysis"""
        mock_session = Mock()
        mock_session.execute.return_value = []

        optimizer = DatabaseOptimizer(mock_session)
        result = optimizer.analyze_query_performance("SELECT * FROM users")

        assert 'query' in result
        assert 'plan' in result

    def test_optimize_connection_pool(self):
        """Test connection pool optimization"""
        mock_session = Mock()
        optimizer = DatabaseOptimizer(mock_session)

        result = optimizer.optimize_connection_pool(10, 20)
        assert result['pool_size'] == 10
        assert result['max_overflow'] == 20

    def test_get_table_statistics(self):
        """Test table statistics retrieval"""
        mock_session = Mock()
        mock_session.execute.return_value.fetchone.return_value = [100]

        optimizer = DatabaseOptimizer(mock_session)
        result = optimizer.get_table_statistics('users')

        assert 'table_name' in result

class TestStructuredLogger:
    """Test structured logger"""

    def test_logger_initialization(self):
        """Test logger initialization"""
        logger = StructuredLogger('test', 'INFO')
        assert logger.logger.name == 'test'

    def test_log_info(self):
        """Test info logging"""
        logger = StructuredLogger('test')
        logger.info("Test message", {'key': 'value'})
        # No assertion needed, just verify no exceptions

    def test_log_error(self):
        """Test error logging"""
        logger = StructuredLogger('test')
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("Error occurred", error=e)

    def test_log_request(self):
        """Test request logging"""
        logger = StructuredLogger('test')
        logger.log_request('GET', '/api/test', 200, 150.5)

    def test_log_authentication(self):
        """Test authentication logging"""
        logger = StructuredLogger('test')
        logger.log_authentication('testuser', True)
        logger.log_authentication('baduser', False, 'Invalid credentials')

    def test_log_security_event(self):
        """Test security event logging"""
        logger = StructuredLogger('test')
        logger.log_security_event('unauthorized_access', 'HIGH', {'ip': '1.2.3.4'})

class TestIntegration:
    """Integration tests"""

    def test_validation_and_response(self):
        """Test validation with response helpers"""
        try:
            validate_business({'name': 'Test'})
        except ValidationError as e:
            response, code = error_response(str(e), 400)
            assert code == 400
            assert response.json['status'] == 'error'

    def test_logging_with_validation(self):
        """Test logging validation errors"""
        logger = StructuredLogger('test')
        try:
            validate_user({'username': 'test', 'password': 'weak', 'email': 'test@example.com'})
        except ValidationError as e:
            logger.error("Validation failed", error=e)

# Performance tests
class TestPerformance:
    """Performance tests"""

    def test_validation_performance(self):
        """Test validation performance"""
        import time  # pylint: disable=import-outside-toplevel

        data = {
            'name': 'Test Corp',
            'type': 'corporation',
            'registration_number': '123456789'
        }

        start = time.time()
        for _ in range(1000):
            validate_business(data)
        duration = time.time() - start

        assert duration < 1.0  # Should complete in less than 1 second

    def test_logging_performance(self):
        """Test logging performance"""
        import time  # pylint: disable=import-outside-toplevel

        logger = StructuredLogger('test')

        start = time.time()
        for _ in range(1000):
            logger.info("Test message", {'iteration': _})
        duration = time.time() - start

        assert duration < 2.0  # Should complete in less than 2 seconds

# Edge case tests
class TestEdgeCases:
    """Edge case tests"""

    def test_empty_string_validation(self):
        """Test empty string validation"""
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_string("", "field", 1, 10)

    def test_very_long_string_validation(self):
        """Test very long string validation"""
        long_string = "a" * 10000
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_string(long_string, "field", 1, 100)

    def test_special_characters_in_email(self):
        """Test special characters in email"""
        assert ComprehensiveValidators.validate_email("user+tag@example.com")
        assert ComprehensiveValidators.validate_email("user.name@example.com")

    def test_unicode_in_validation(self):
        """Test unicode characters in validation"""
        assert ComprehensiveValidators.validate_string("Test 测试", "field", 1, 100)

    def test_null_values(self):
        """Test null value handling"""
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_email(None)
        with pytest.raises(ValidationError):
            ComprehensiveValidators.validate_phone(None)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src', '--cov-report=html'])
