import json
import types
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

import src.audit_logger as al


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))

    def debug(self, *args, **kwargs):
        self.messages.append(("debug", args, kwargs))


class DummyAuditLogModel:
    id = 1
    timestamp = datetime.now(timezone.utc)
    user_id = "u1"
    username = "name"
    action = "a"
    severity = "info"
    status_code = 200
    resource_type = "x"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if "timestamp" not in self.__dict__:
            self.timestamp = datetime.now(timezone.utc)
        if "action" not in self.__dict__:
            self.action = "a"
        if "severity" not in self.__dict__:
            self.severity = "info"

    @staticmethod
    def calculate_hash(log_data, previous_hash):
        return "hash_" + (previous_hash or "root")

    @staticmethod
    def verify_chain_integrity(logs):
        if logs and getattr(logs[0], "current_hash", None) == "bad":
            return False, "bad chain"
        return True, "ok"

    def to_dict(self):
        return {
            "id": getattr(self, "id", 1),
            "action": getattr(self, "action", "a"),
            "severity": getattr(self, "severity", "info"),
            "timestamp": getattr(self, "timestamp", datetime.now(timezone.utc)).isoformat(),
        }


class DummyAuditLogSummary:
    def __init__(self, total_logs, by_action, by_severity, by_user, failed_attempts, time_range):
        self.total_logs = total_logs
        self.by_action = by_action
        self.by_severity = by_severity
        self.by_user = by_user
        self.failed_attempts = failed_attempts
        self.time_range = time_range


class FakeQuery:
    def __init__(self, rows=None, first_row=None, raise_on_all=False):
        self.rows = rows if rows is not None else []
        self.first_row = first_row
        self.raise_on_all = raise_on_all

    def order_by(self, *_a, **_k):
        return self

    def filter(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def offset(self, *_a, **_k):
        return self

    def first(self):
        return self.first_row

    def all(self):
        if self.raise_on_all:
            raise RuntimeError("query failed")
        return self.rows


class FakeSession:
    def __init__(self, query_obj):
        self.query_obj = query_obj
        self.added = []

    def query(self, *_a, **_k):
        return self.query_obj

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        return None

    def refresh(self, _obj):
        return None


class FakeDBManager:
    def __init__(self, session):
        self.session = session

    @contextmanager
    def get_session(self):
        yield self.session


@pytest.fixture
def patched_module(monkeypatch):
    dlog = DummyLogger()
    monkeypatch.setattr(al, "telemetry_logger", types.SimpleNamespace(get_logger=lambda: dlog), raising=False)
    monkeypatch.setattr(al, "AuditLogModel", DummyAuditLogModel, raising=False)
    monkeypatch.setattr(al, "AuditLogSummary", DummyAuditLogSummary, raising=False)
    return dlog


def test_sanitize_data_error_path(patched_module, monkeypatch):
    logger = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery())))
    monkeypatch.setattr(al.json, "dumps", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = logger._sanitize_data({"x": 1})
    assert "Failed to sanitize data" in out


def test_sanitize_data_none_returns_none(patched_module):
    logger = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery())))
    assert logger._sanitize_data(None) is None


def test_extract_user_context_outside_request_context(patched_module):
    logger = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery())))
    ctx = logger._extract_user_context()
    assert isinstance(ctx, dict)
    assert "ip_address" in ctx


def test_get_audit_trail_exception_returns_empty(patched_module):
    q = FakeQuery(raise_on_all=True)
    logger = al.AuditLogger(FakeDBManager(FakeSession(q)))
    rows = logger.get_audit_trail(user_id="u", action="a", resource_type="r", severity="warning")
    assert rows == []


def test_get_audit_summary_success_and_exception(patched_module):
    rows = [
        DummyAuditLogModel(action="login", severity="info", username="a", status_code=200, timestamp=datetime.now(timezone.utc)),
        DummyAuditLogModel(action="login", severity="warning", username="a", status_code=401, timestamp=datetime.now(timezone.utc)),
        DummyAuditLogModel(action="api", severity="error", username="b", status_code=500, timestamp=datetime.now(timezone.utc)),
    ]
    logger = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery(rows=rows))))
    summary = logger.get_audit_summary()
    assert summary.total_logs == 3
    assert summary.failed_attempts == 2
    assert summary.by_action["login"] == 2

    bad = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery(raise_on_all=True))))
    summary2 = bad.get_audit_summary()
    assert summary2.total_logs == 0


def test_verify_integrity_paths(patched_module):
    good_rows = [DummyAuditLogModel(current_hash="ok")]
    bad_rows = [DummyAuditLogModel(current_hash="bad")]
    logger_good = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery(rows=good_rows))))
    assert logger_good.verify_integrity() == (True, "ok")

    logger_bad = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery(rows=bad_rows))))
    is_valid, _msg = logger_bad.verify_integrity()
    assert is_valid is False


def test_export_audit_logs_json_csv_and_invalid(patched_module):
    rows = [DummyAuditLogModel(action="x")]
    logger = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery(rows=rows))))

    j = logger.export_audit_logs("json")
    assert j.startswith("[")

    c = logger.export_audit_logs("csv")
    assert "action" in c

    with pytest.raises(ValueError):
        logger.export_audit_logs("xml")


def test_export_audit_logs_csv_fallback_object_and_to_dict_failure(monkeypatch, patched_module):
    class ObjOnly:
        def __init__(self):
            self.action = "obj_action"
            self.status_code = 204
            self.username = "u"

    class BadToDict:
        def __init__(self):
            self.action = "bad_to_dict_action"
            self.severity = "warning"

        def to_dict(self):
            raise RuntimeError("nope")

    logger = al.AuditLogger(FakeDBManager(FakeSession(FakeQuery(rows=[]))))
    monkeypatch.setattr(logger, "get_audit_trail", lambda **_k: [ObjOnly(), BadToDict()])

    c = logger.export_audit_logs("csv")

    assert "action" in c
    assert "obj_action" in c
    assert "bad_to_dict_action" in c


def test_log_event_handles_db_exception(patched_module):
    class BadSession(FakeSession):
        def commit(self):
            raise RuntimeError("db fail")

    logger = al.AuditLogger(FakeDBManager(BadSession(FakeQuery())))
    out = logger.log_event(action="a")
    assert out is None


def test_decorator_wrapper_logging_failure_path(monkeypatch, patched_module):
    class CurrentApp:
        audit_logger = types.SimpleNamespace(
            log_event=lambda **_k: (_ for _ in ()).throw(RuntimeError("log fail"))
        )

    monkeypatch.setattr("flask.current_app", CurrentApp(), raising=False)

    @al.audit_log(action="x", resource_type="endpoint")
    def endpoint_ok():
        return {"ok": True}, 200

    # Should not raise due to logging failure swallowed in decorator
    resp = endpoint_ok()
    assert isinstance(resp, tuple)
