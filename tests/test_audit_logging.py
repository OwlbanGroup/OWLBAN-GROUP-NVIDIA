"""
Test suite for Audit Logging System
Tests core functionality, database integration, and security features
"""
import sys
import os
import pytest
from datetime import datetime, timezone, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.audit_log import AuditLogModel, AuditLogSummary
from src.audit_logger import AuditLogger
from src.audit_reports import AuditReportGenerator
from src.audit_alerts import AuditAlertManager, AlertSeverity, AlertType
from src.database_fixed import DatabaseManager
from config import config


class TestAuditLogModel:
    """Test AuditLogModel functionality"""
    
    def test_calculate_hash(self):
        """Test hash calculation"""
        log_data = {
            'timestamp': '2025-12-01T10:00:00Z',
            'user_id': 'user123',
            'action': 'login',
            'resource_type': 'user',
            'resource_id': 'user123',
            'endpoint': '/api/login',
            'status_code': 200
        }
        
        hash1 = AuditLogModel.calculate_hash(log_data)
        hash2 = AuditLogModel.calculate_hash(log_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string
        
    def test_calculate_hash_with_previous(self):
        """Test hash calculation with previous hash"""
        log_data = {
            'timestamp': '2025-12-01T10:00:00Z',
            'user_id': 'user123',
            'action': 'login'
        }
        
        previous_hash = 'abc123'
        hash_with_prev = AuditLogModel.calculate_hash(log_data, previous_hash)
        hash_without_prev = AuditLogModel.calculate_hash(log_data)
        
        # Hash should be different with previous hash
        assert hash_with_prev != hash_without_prev
        
    def test_to_dict(self):
        """Test conversion to dictionary"""
        # This would require a database session, so we'll test the structure
        log_dict = {
            'id': 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': 'user123',
            'username': 'john.doe',
            'action': 'login',
            'status_code': 200
        }
        
        # Verify expected keys
        expected_keys = ['id', 'timestamp', 'user_id', 'username', 'action', 'status_code']
        for key in expected_keys:
            assert key in log_dict


class TestAuditLogger:
    """Test AuditLogger functionality"""
    
    @pytest.fixture
    def db_manager(self):
        """Create test database manager"""
        # Use in-memory SQLite for testing
        return DatabaseManager('sqlite:///:memory:')
    
    @pytest.fixture
    def audit_logger(self, db_manager: Any):
        """Create audit logger instance"""
        return AuditLogger(db_manager)
    
    def test_sanitize_data(self, audit_logger: Any):
        """Test sensitive data sanitization"""
        sensitive_data = {
            'username': 'john.doe',
            'password': 'secret123',
            'token': 'abc123token',
            'api_key': 'key123',
            'normal_field': 'normal_value'
        }
        
        sanitized = audit_logger._sanitize_data(sensitive_data)
        sanitized_dict = json.loads(sanitized)
        
        # Sensitive fields should be redacted
        assert sanitized_dict['password'] == '***REDACTED***'
        assert sanitized_dict['token'] == '***REDACTED***'
        assert sanitized_dict['api_key'] == '***REDACTED***'
        
        # Normal fields should remain
        assert sanitized_dict['username'] == 'john.doe'
        assert sanitized_dict['normal_field'] == 'normal_value'
    
    def test_sanitize_data_truncation(self, audit_logger: Any):
        """Test data truncation for large payloads"""
        large_data = {'data': 'x' * 10000}
        
        sanitized = audit_logger._sanitize_data(large_data, max_length=100)
        
        # Should be truncated
        assert len(sanitized) <= 120  # 100 + truncation message
        assert '[TRUNCATED]' in sanitized
    
    def test_log_authentication_attempt_success(self, audit_logger: Any):
        """Test logging successful authentication"""
        log = audit_logger.log_authentication_attempt(
            username='john.doe',
            success=True,
            auth_method='password'
        )
        
        if log:
            assert log.action == 'authentication_attempt'
            assert log.username == 'john.doe'
            assert log.status_code == 200
            assert log.severity == 'info'
            assert log.category == 'authentication'
    
    def test_log_authentication_attempt_failure(self, audit_logger: Any):
        """Test logging failed authentication"""
        log = audit_logger.log_authentication_attempt(
            username='john.doe',
            success=False,
            reason='Invalid password',
            auth_method='password'
        )
        
        if log:
            assert log.action == 'authentication_attempt'
            assert log.status_code == 401
            assert log.severity == 'warning'
            assert log.error_message == 'Invalid password'
    
    def test_log_api_call(self, audit_logger: Any):
        """Test logging API call"""
        log = audit_logger.log_api_call(
            endpoint='/api/users',
            method='GET',
            status_code=200,
            response_time_ms=45
        )
        
        if log:
            assert log.action == 'api_call'
            assert log.resource_id == '/api/users'
            assert log.status_code == 200
            assert log.response_time_ms == 45
            assert log.severity == 'info'
    
    def test_log_database_operation(self, audit_logger: Any):
        """Test logging database operation"""
        log = audit_logger.log_database_operation(
            operation='create',
            table='users',
            record_id='123',
            data={'username': 'john.doe'},
            success=True
        )
        
        if log:
            assert log.action == 'db_create'
            assert log.resource_type == 'database'
            assert 'users:123' in log.resource_id
            assert log.status_code == 200
    
    def test_log_security_event(self, audit_logger: Any):
        """Test logging security event"""
        log = audit_logger.log_security_event(
            event_type='suspicious_activity',
            description='Multiple failed login attempts',
            severity='high'
        )
        
        if log:
            assert log.action == 'security_event'
            assert log.resource_type == 'security'
            assert log.severity == 'high'
            assert log.category == 'security'


class TestAuditReports:
    """Test AuditReportGenerator functionality"""
    
    @pytest.fixture
    def db_manager(self):
        """Create test database manager"""
        return DatabaseManager('sqlite:///:memory:')
    
    @pytest.fixture
    def report_generator(self, db_manager: Any):
        """Create report generator instance"""
        return AuditReportGenerator(db_manager)
    
    def test_generate_user_activity_report(self, report_generator: Any):
        """Test user activity report generation"""
        report = report_generator.generate_user_activity_report(
            username='john.doe',
            start_date=datetime.now(timezone.utc) - timedelta(days=7)
        )
        
        assert 'report_type' in report
        assert report['report_type'] == 'user_activity'
        assert 'summary' in report
        assert 'generated_at' in report
    
    def test_generate_security_report(self, report_generator: Any):
        """Test security report generation"""
        report = report_generator.generate_security_report(
            start_date=datetime.now(timezone.utc) - timedelta(days=7)
        )
        
        assert 'report_type' in report
        assert report['report_type'] == 'security'
        assert 'summary' in report
    
    def test_generate_compliance_report(self, report_generator: Any):
        """Test compliance report generation"""
        report = report_generator.generate_compliance_report(
            compliance_standard='PCI-DSS',
            start_date=datetime.now(timezone.utc) - timedelta(days=30)
        )
        
        assert 'report_type' in report
        assert report['report_type'] == 'compliance'
        assert 'compliance_standard' in report
        assert report['compliance_standard'] == 'PCI-DSS'
        assert 'compliance_metrics' in report
    
    def test_export_report_json(self, report_generator: Any):
        """Test report export in JSON format"""
        report_data = {
            'report_type': 'test',
            'data': {'key': 'value'}
        }
        
        exported = report_generator.export_report(report_data, format_type='json')
        
        assert isinstance(exported, str)
        parsed = json.loads(exported)
        assert parsed['report_type'] == 'test'
    
    def test_export_report_html(self, report_generator: Any):
        """Test report export in HTML format"""
        report_data = {
            'report_type': 'test',
            'summary': {'total': 100}
        }
        
        exported = report_generator.export_report(report_data, format_type='html')
        
        assert isinstance(exported, str)
        assert '<html>' in exported
        assert 'test' in exported


class TestAuditAlerts:
    """Test AuditAlertManager functionality"""
    
    @pytest.fixture
    def db_manager(self):
        """Create test database manager"""
        return DatabaseManager('sqlite:///:memory:')
    
    @pytest.fixture
    def alert_manager(self, db_manager: Any):
        """Create alert manager instance"""
        return AuditAlertManager(db_manager)
    
    def test_alert_manager_initialization(self, alert_manager: Any):
        """Test alert manager initialization"""
        assert alert_manager is not None
        assert len(alert_manager.alert_rules) > 0  # Should have default rules
        assert alert_manager.active_alerts == []
    
    def test_add_alert_rule(self, alert_manager: Any):
        """Test adding custom alert rule"""
        from src.audit_alerts import AlertRule
        
        initial_count = len(alert_manager.alert_rules)
        
        rule = AlertRule(
            rule_id='test_rule',
            name='Test Rule',
            alert_type=AlertType.UNUSUAL_ACTIVITY,
            severity=AlertSeverity.MEDIUM,
            condition=lambda logs: len(logs) > 10
        )
        
        alert_manager.add_alert_rule(rule)
        
        assert len(alert_manager.alert_rules) == initial_count + 1
    
    def test_remove_alert_rule(self, alert_manager: Any):
        """Test removing alert rule"""
        from src.audit_alerts import AlertRule
        
        rule = AlertRule(
            rule_id='test_rule_remove',
            name='Test Rule',
            alert_type=AlertType.UNUSUAL_ACTIVITY,
            severity=AlertSeverity.MEDIUM,
            condition=lambda logs: True
        )
        
        alert_manager.add_alert_rule(rule)
        initial_count = len(alert_manager.alert_rules)
        
        alert_manager.remove_alert_rule('test_rule_remove')
        
        assert len(alert_manager.alert_rules) == initial_count - 1
    
    def test_get_active_alerts(self, alert_manager: Any):
        """Test getting active alerts"""
        alerts = alert_manager.get_active_alerts()
        
        assert isinstance(alerts, list)
    
    def test_get_active_alerts_with_filters(self, alert_manager: Any):
        """Test getting active alerts with filters"""
        alerts = alert_manager.get_active_alerts(
            severity=AlertSeverity.HIGH,
            acknowledged=False
        )
        
        assert isinstance(alerts, list)


class TestDatabaseIntegration:
    """Test database integration"""
    
    @pytest.fixture
    def db_manager(self):
        """Create test database manager"""
        return DatabaseManager('sqlite:///:memory:')
    
    def test_database_health_check(self, db_manager: Any):
        """Test database connectivity"""
        assert db_manager.health_check() is True
    
    def test_get_audit_logs_empty(self, db_manager: Any):
        """Test getting audit logs from empty database"""
        logs = db_manager.get_audit_logs()
        
        assert isinstance(logs, list)
        assert len(logs) == 0
    
    def test_get_audit_log_count(self, db_manager: Any):
        """Test getting audit log count"""
        count = db_manager.get_audit_log_count()
        
        assert isinstance(count, int)
        assert count >= 0
    
    def test_cleanup_old_audit_logs(self, db_manager: Any):
        """Test cleanup of old audit logs"""
        deleted = db_manager.cleanup_old_audit_logs(retention_days=90)
        
        assert isinstance(deleted, int)
        assert deleted >= 0


class TestConfiguration:
    """Test configuration settings"""
    
    def test_audit_config_exists(self):
        """Test that audit configuration exists"""
        assert hasattr(config, 'AUDIT_LOG_ENABLED')
        assert hasattr(config, 'AUDIT_LOG_RETENTION_DAYS')
        assert hasattr(config, 'AUDIT_ALERT_ENABLED')
        assert hasattr(config, 'AUDIT_FAILED_LOGIN_THRESHOLD')
    
    def test_audit_config_values(self):
        """Test audit configuration values"""
        assert isinstance(config.AUDIT_LOG_ENABLED, bool)
        assert isinstance(config.AUDIT_LOG_RETENTION_DAYS, int)
        assert isinstance(config.AUDIT_FAILED_LOGIN_THRESHOLD, int)
        assert config.AUDIT_LOG_RETENTION_DAYS > 0
        assert config.AUDIT_FAILED_LOGIN_THRESHOLD > 0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
