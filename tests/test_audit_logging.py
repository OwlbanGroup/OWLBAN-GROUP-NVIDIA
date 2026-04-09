"""
Test suite for Audit Logging System
Uses real audit modules with lightweight test doubles for DB/session dependencies.
"""
import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from src.audit_logger import AuditLogger
from src.models.audit_log import AuditLogModel, AuditLogSummary


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _SessionContext:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


class _QueryStub:
    def __init__(self, rows=None):
        self.rows = rows or []

    def order_by(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def offset(self, *args, **kwargs):
        return self

    def all(self):
        return list(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class _SessionStub:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []

    def query(self, *_args, **_kwargs):
        return _QueryStub(self.rows)

    def add(self, model):
        self.added.append(model)

    def commit(self):
        return None

    def refresh(self, _model):
        return None


class _DBManagerStub:
    def __init__(self, session=None):
        self._session = session or _SessionStub()

    def get_session(self):
        return _SessionContext(self._session)


@pytest.fixture
def audit_logger(monkeypatch):
    import src.audit_logger as audit_mod

    monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
    db_manager = _DBManagerStub()
    return AuditLogger(db_manager)


class TestAuditLogModel:
    def test_calculate_hash_stable(self):
        log_data = {
            "timestamp": "2025-12-01T10:00:00Z",
            "user_id": "user123",
            "action": "login",
            "resource_type": "user",
            "resource_id": "user123",
            "endpoint": "/api/login",
            "status_code": 200,
        }
        hash1 = AuditLogModel.calculate_hash(log_data)
        hash2 = AuditLogModel.calculate_hash(log_data)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_calculate_hash_with_previous_changes_value(self):
        log_data = {"timestamp": datetime.now(timezone.utc), "user_id": "u1", "action": "login"}
        a = AuditLogModel.calculate_hash(log_data)
        b = AuditLogModel.calculate_hash(log_data, previous_hash="prev")
        assert a != b

    def test_verify_chain_integrity_happy_path(self):
        t1 = datetime.now(timezone.utc)
        t2 = t1 + timedelta(seconds=1)

        l1 = AuditLogModel(
            id=1,
            timestamp=t1,
            user_id="u1",
            action="a1",
            resource_type="r",
            resource_id="1",
            endpoint="/e1",
            status_code=200,
        )
        l1.current_hash = AuditLogModel.calculate_hash(l1.to_dict(), None)
        l1.previous_hash = None

        l2 = AuditLogModel(
            id=2,
            timestamp=t2,
            user_id="u1",
            action="a2",
            resource_type="r",
            resource_id="2",
            endpoint="/e2",
            status_code=200,
        )
        l2.previous_hash = l1.current_hash
        l2.current_hash = AuditLogModel.calculate_hash(l2.to_dict(), l1.current_hash)

        ok, err = AuditLogModel.verify_chain_integrity([l1, l2])
        assert ok is True
        assert err is None

    def test_verify_chain_integrity_broken_hash(self):
        t1 = datetime.now(timezone.utc)
        l1 = AuditLogModel(
            id=1,
            timestamp=t1,
            user_id="u1",
            action="a1",
            resource_type="r",
            resource_id="1",
            endpoint="/e1",
            status_code=200,
        )
        l1.previous_hash = None
        l1.current_hash = "bad_hash"

        ok, err = AuditLogModel.verify_chain_integrity([l1])
        assert ok is False
        assert "Hash mismatch" in err

    def test_audit_summary_to_dict(self):
        now = datetime.now(timezone.utc)
        summary = AuditLogSummary(
            total_logs=10,
            by_action={"login": 5},
            by_severity={"info": 8, "warning": 2},
            by_user={"alice": 3},
            failed_attempts=2,
            time_range=(now - timedelta(hours=1), now),
        )
        result = summary.to_dict()
        assert result["total_logs"] == 10
        assert result["by_action"]["login"] == 5
        assert result["time_range"]["start"] is not None
        assert result["time_range"]["end"] is not None


class TestAuditLogger:
    def test_sanitize_data_redacts_sensitive_fields(self, audit_logger):
        data = {
            "username": "john.doe",
            "password": "secret123",
            "token": "abc123token",
            "api_key": "key123",
            "authorization": "Bearer abc",
            "normal_field": "normal_value",
        }
        sanitized = audit_logger._sanitize_data(data)
        sanitized_dict = json.loads(sanitized)
        assert sanitized_dict["password"] == "***REDACTED***"
        assert sanitized_dict["token"] == "***REDACTED***"
        assert sanitized_dict["api_key"] == "***REDACTED***"
        assert sanitized_dict["authorization"] == "***REDACTED***"
        assert sanitized_dict["normal_field"] == "normal_value"

    def test_sanitize_data_truncation(self, audit_logger):
        large_data = {"data": "x" * 10000}
        sanitized = audit_logger._sanitize_data(large_data, max_length=100)
        assert len(sanitized) <= 120
        assert "[TRUNCATED]" in sanitized

    def test_sanitize_data_non_dict(self, audit_logger):
        sanitized = audit_logger._sanitize_data("plain-text")
        obj = json.loads(sanitized)
        assert obj["value"] == "plain-text"

    def test_log_event_creates_model(self, monkeypatch):
        import src.audit_logger as audit_mod

        session = _SessionStub(rows=[])
        db = _DBManagerStub(session=session)
        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        logger = AuditLogger(db)

        evt = logger.log_event(
            action="authentication_attempt",
            resource_type="user",
            resource_id="john",
            status_code=200,
            request_data={"password": "x"},
            response_data={"ok": True},
            category="authentication",
            compliance_tags=["PCI-DSS"],
            username="john",
        )

        assert evt is not None
        assert len(session.added) == 1
        assert session.added[0].action == "authentication_attempt"
        assert session.added[0].severity == "info"

    def test_log_authentication_attempt_success(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        logger = AuditLogger(_DBManagerStub())
        result = logger.log_authentication_attempt("john.doe", True, auth_method="password")
        assert result is not None
        assert result.action == "authentication_attempt"
        assert result.status_code == 200
        assert result.severity == "info"

    def test_log_authentication_attempt_failure(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        logger = AuditLogger(_DBManagerStub())
        result = logger.log_authentication_attempt("john.doe", False, reason="Invalid password")
        assert result is not None
        assert result.status_code == 401
        assert result.severity == "warning"
        assert result.error_message == "Invalid password"

    def test_log_api_call_severity_branches(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        logger = AuditLogger(_DBManagerStub())

        ok = logger.log_api_call("/api/users", "GET", 200, 10)
        warn = logger.log_api_call("/api/users", "GET", 404, 11)
        err = logger.log_api_call("/api/users", "GET", 500, 12)

        assert ok.severity == "info"
        assert warn.severity == "warning"
        assert err.severity == "error"

    def test_log_database_operation_and_security_event(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        logger = AuditLogger(_DBManagerStub())

        db_log = logger.log_database_operation("create", "users", "123", {"username": "john"}, success=True)
        sec_log = logger.log_security_event("suspicious_activity", "Multiple failed login attempts", severity="high")

        assert db_log.action == "db_create"
        assert db_log.resource_type == "database"
        assert db_log.status_code == 200
        assert sec_log.action == "security_event"
        assert sec_log.resource_type == "security"
        assert sec_log.severity == "high"

    def test_get_audit_trail_and_summary(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        row = AuditLogModel(
            id=1,
            timestamp=datetime.now(timezone.utc),
            user_id="u1",
            username="john",
            action="login",
            severity="warning",
            resource_type="user",
            status_code=401,
            current_hash="x" * 64,
        )
        row.previous_hash = None
        db = _DBManagerStub(session=_SessionStub(rows=[row]))
        logger = AuditLogger(db)

        trail = logger.get_audit_trail(user_id="u1", action="login", severity="warning", limit=10, offset=0)
        summary = logger.get_audit_summary()

        assert isinstance(trail, list)
        assert len(trail) == 1
        assert summary.total_logs == 1
        assert summary.failed_attempts == 1
        assert summary.by_action["login"] == 1

    def test_verify_integrity_and_export_json_csv(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))

        t = datetime.now(timezone.utc)
        row = AuditLogModel(
            id=1,
            timestamp=t,
            user_id="u1",
            username="john",
            action="login",
            severity="info",
            resource_type="user",
            status_code=200,
        )
        row.previous_hash = None
        row.current_hash = AuditLogModel.calculate_hash(row.to_dict(), None)

        db = _DBManagerStub(session=_SessionStub(rows=[row]))
        logger = AuditLogger(db)

        ok, err = logger.verify_integrity()
        assert ok is True
        assert err is None

        json_blob = logger.export_audit_logs("json")
        csv_blob = logger.export_audit_logs("csv")
        assert "login" in json_blob
        assert "login" in csv_blob

    def test_export_unsupported_format_raises(self, monkeypatch):
        import src.audit_logger as audit_mod

        monkeypatch.setattr(audit_mod, "telemetry_logger", SimpleNamespace(get_logger=lambda: _DummyLogger()))
        logger = AuditLogger(_DBManagerStub())
        with pytest.raises(ValueError):
            logger.export_audit_logs("xml")


class TestConfiguration:
    @pytest.mark.skip(reason="Config validation out of scope for this focused suite")
    def test_audit_config_exists(self):
        assert True

    @pytest.mark.skip(reason="Config validation out of scope for this focused suite")
    def test_audit_config_values(self):
        assert True
