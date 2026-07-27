import types
from datetime import datetime, timezone

import pytest

import app_final as app_mod


@pytest.fixture
def client(monkeypatch):
    app_mod.app.config["TESTING"] = True
    app_mod.app.config["RATELIMIT_ENABLED"] = False

    class DummyLogger:
        def info(self, *args, **kwargs): return None
        def warning(self, *args, **kwargs): return None
        def error(self, *args, **kwargs): return None
        def debug(self, *args, **kwargs): return None

    monkeypatch.setattr(app_mod, "telemetry_logger", types.SimpleNamespace(
        get_logger=lambda: DummyLogger(),
        log_error=lambda *a, **k: None
    ))

    class TelemetryStub:
        def process_single_event(self, data): return data.get("ok", True)
        def process_batch(self, events): return {"successful": len(events), "total": len(events)}
        def get_metrics(self, _hours): return {"events_processed": 10, "anomalies_detected": 1}
        def export_events(self, operation=None, limit=1000):
            return [{"id": 1, "operation": operation or "x"}][:limit]
        def detect_anomalies_in_batch(self, events): return [{"idx": i, "anomaly": False} for i, _ in enumerate(events)]

    class AIServiceStub:
        def analyze_financial_data(self, *args, **kwargs): return {"status": "success", "result": "analysis"}
        def assess_transaction_risk(self, *args, **kwargs): return {"status": "success", "risk": "low"}
        def process_natural_language_query(self, *args, **kwargs): return {"status": "success", "answer": "ok"}
        def get_service_status(self): return {"provider": "stub", "healthy": True}

    class SyncServiceStub:
        def get_business_intelligence(self, *_a, **_k): return {"status": "success", "data": {}}
        def forecast_revenue(self, *_a, **_k): return {"status": "success", "forecast": []}
        def sync_payment_to_revenue(self, *_a, **_k): return {"status": "success", "id": "r1"}

    class SyncSchedulerStub:
        def __init__(self):
            self.running = False
            self.db_manager = types.SimpleNamespace(connection_string="postgresql://invalid")
        def start_scheduler(self): self.running = True
        def stop_scheduler(self): self.running = False
        def get_job_status(self): return {"job": "ok"}
        def run_job_now(self, job_id):
            if job_id == "bad":
                raise ValueError("invalid job")
            return {"job_id": job_id, "status": "done"}

    class PaymentsServiceStub:
        def create_stripe_payment_intent(self, amount, currency, description, metadata):
            if amount <= 0:
                return {"status": "error", "error": "invalid amount"}
            return {"status": "success", "payment_intent_id": "pi_1", "client_secret": "sec", "amount": amount, "currency": currency}
        def confirm_stripe_payment(self, pid):
            if pid == "pending":
                return {"status": "pending", "payment_intent_id": pid}
            if pid == "bad":
                return {"status": "error", "error": "bad"}
            return {"status": "success", "payment_intent_id": pid, "amount": 100, "currency": "USD"}
        def create_stripe_refund(self, payment_intent_id, amount=None, reason="requested_by_customer"):
            if payment_intent_id == "bad":
                return {"status": "error", "error": "bad refund"}
            return {"status": "success", "refund_id": "re_1", "amount": amount or 100, "currency": "USD"}
        def process_stripe_webhook(self, payload, sig):
            if "bad" in payload:
                return {"status": "error", "error": "bad payload"}
            return {"status": "success", "event_type": "payment_intent.succeeded", "event_id": "evt_1"}

    class _ORM:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DBStub:
        def get_all_businesses(self): return [_ORM(id=1, name="B1")]
        def get_all_assets(self): return [_ORM(id=1, name="A1", business_id=1)]
        def get_business_by_id(self, bid): return _ORM(id=bid, name="B1") if bid == 1 else None
        def get_assets_by_business_id(self, bid): return [_ORM(id=1, name="A1", business_id=bid)]

    monkeypatch.setattr(app_mod, "telemetry_handler", TelemetryStub())
    monkeypatch.setattr(app_mod, "ai_service", AIServiceStub())
    monkeypatch.setattr(app_mod, "sync_service", SyncServiceStub())
    monkeypatch.setattr(app_mod, "sync_scheduler", SyncSchedulerStub())
    monkeypatch.setattr(app_mod, "payments_service", PaymentsServiceStub())
    monkeypatch.setattr(app_mod, "db_manager", DBStub())

    app_mod.users["u_cov"] = {
        "password": app_mod.generate_password_hash("p"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "token": "tok_cov",
    }

    with app_mod.app.test_client() as c:
        yield c


def _auth():
    return {"Authorization": "Bearer tok_cov"}


def test_data_convert_and_sync_status_not_initialized(client, monkeypatch):
    monkeypatch.setattr(app_mod, "convert_data_format_logic", lambda req: (app_mod.jsonify({"status": "success"}), 200))
    assert client.post("/data/convert", json={"from_format": "json", "to_format": "csv", "data": [{"a": 1}]}).status_code == 200

    monkeypatch.setattr(app_mod, "sync_scheduler", None)
    r = client.get("/sync/status", headers=_auth())
    assert r.status_code == 200
    assert r.get_json().get("scheduler_status") == "not_initialized"


def test_sync_logs_db_failure_and_limits(client, monkeypatch):
    assert client.get("/sync/logs?limit=501", headers=_auth()).status_code == 400

    def _raise_connect(*_a, **_k):
        raise RuntimeError("db down")
    monkeypatch.setattr(app_mod.psycopg2, "connect", _raise_connect)
    assert client.get("/sync/logs?limit=5", headers=_auth()).status_code == 500


def test_stripe_error_paths(client):
    assert client.post("/stripe/payment-intent", headers=_auth(), json={"amount": 0, "currency": "USD"}).status_code == 400
    assert client.post("/stripe/payment-intent/bad/confirm", headers=_auth()).status_code == 400
    assert client.post("/stripe/refund", headers=_auth(), json={"payment_intent_id": "bad"}).status_code == 400
    assert client.post("/stripe/webhook", data="bad payload", headers={"stripe-signature": "sig"}).status_code == 400


def test_api_alias_error_paths(client, monkeypatch):
    monkeypatch.setattr(app_mod, "sync_service", types.SimpleNamespace(
        get_business_intelligence=lambda *_a, **_k: {"status": "error", "message": "bad"},
        forecast_revenue=lambda *_a, **_k: {"status": "error", "message": "bad"},
        sync_payment_to_revenue=lambda *_a, **_k: {"status": "error", "message": "bad"},
    ))
    assert client.get("/api/business/intelligence/u1", headers=_auth()).status_code == 400
    assert client.get("/api/business/forecast/u1", headers=_auth()).status_code == 400

    monkeypatch.setattr(app_mod, "ai_service", types.SimpleNamespace(
        analyze_financial_data=lambda *_a, **_k: {"status": "error"},
        assess_transaction_risk=lambda *_a, **_k: {"status": "error"},
        process_natural_language_query=lambda *_a, **_k: {"status": "error"},
        get_service_status=lambda: {"provider": "stub", "healthy": True},
    ))
    assert client.post("/api/ai/analyze", headers=_auth(), json={"data": {"a": 1}, "question": "q"}).status_code == 400
    assert client.post("/api/ai/risk-assess", headers=_auth(), json={"transaction_data": {"a": 1}}).status_code == 400
    assert client.post("/api/ai/query", headers=_auth(), json={"query": "q"}).status_code == 400


def test_lightweight_aliases_and_batch_controls(client, monkeypatch):
    assert client.get("/storage/files").status_code == 200
    assert client.get("/benefits").status_code == 200
    assert client.get("/payroll").status_code == 200
    assert client.get("/patterns").status_code == 200
    assert client.get("/traction").status_code == 200
    assert client.get("/purchasing").status_code == 200
    assert client.get("/bill-pay").status_code == 200
    assert client.get("/ws/status").status_code == 200
    assert client.get("/batch/status").status_code == 200
    assert client.post("/batch/start").status_code == 200
    assert client.post("/batch/stop").status_code == 200
    assert client.get("/data/formats").status_code == 200

    monkeypatch.setattr(
        app_mod,
        "get_storage_files_logic",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    assert client.get("/storage/files").status_code == 500


def test_dashboard_and_sync_run_paths(client, monkeypatch):
    app_mod.REQUEST_HISTORY.clear()
    assert client.get("/api/dashboard/trends?points=999").status_code == 200
    assert client.get("/api/dashboard/summary").status_code == 200

    app_mod.sync_scheduler = None
    assert client.post("/sync/run/job1", headers=_auth()).status_code == 400

    class RunStub:
        running = False
        db_manager = types.SimpleNamespace(connection_string="postgresql://invalid")

        def run_job_now(self, job_id):
            if job_id == "bad":
                raise ValueError("invalid job")
            return {"ok": True, "job_id": job_id}

    app_mod.sync_scheduler = RunStub()
    assert client.post("/sync/run/good", headers=_auth()).status_code == 200
    assert client.post("/sync/run/bad", headers=_auth()).status_code == 400


def test_sync_start_stop_and_github_paths(client, monkeypatch):
    app_mod.sync_scheduler = None
    assert client.post("/sync/start", headers=_auth()).status_code in (200, 500)
    assert client.post("/sync/stop", headers=_auth()).status_code in (200, 400, 500)

    class Resp:
        def __init__(self, code, payload):
            self.status_code = code
            self._payload = payload

        def json(self):
            return self._payload

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(200, [{"id": 1}]))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer t"}).status_code == 200
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 200
    assert client.get("/api/github/repos", headers={"Authorization": "Bearer t"}).status_code == 400
    assert client.get("/api/github/orgs").status_code == 401


def test_github_status_mappings_and_request_exception(client, monkeypatch):
    class Resp:
        def __init__(self, code, payload):
            self.status_code = code
            self._payload = payload

        def json(self):
            return self._payload

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(401, {"x": 1}))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer t"}).status_code == 401

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(404, {"x": 1}))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer t"}).status_code == 404

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(502, {"x": 1}))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer t"}).status_code == 502

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(418, {"x": 1}))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer t"}).status_code == 500

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(400, {"x": 1}))
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 400

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(401, {"x": 1}))
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 401

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(404, {"x": 1}))
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 404

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(502, {"x": 1}))
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 502

    monkeypatch.setattr(app_mod.requests, "get", lambda *_a, **_k: Resp(418, {"x": 1}))
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 500

    def _raise_req_exc(*_a, **_k):
        raise app_mod.requests.RequestException("network down")

    monkeypatch.setattr(app_mod.requests, "get", _raise_req_exc)
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer t"}).status_code == 500
    assert client.get("/api/github/repos?owner=o", headers={"Authorization": "Bearer t"}).status_code == 500


def test_ml_train_and_webhook_validation_paths(client, monkeypatch):
    assert client.post("/ml/train", headers=_auth(), json={}).status_code == 400
    assert client.post("/ml/train", headers=_auth(), json={"training_data": [[1], [2]]}).status_code == 400
    assert client.post("/ml/train", headers=_auth(), json={"training_data": [[1]] * 10, "contamination": 0.9}).status_code == 400
    assert client.post("/ml/train", headers=_auth(), json={"telemetry_data": [{"data": {"Op": "a"}}]}).status_code == 400

    monkeypatch.setattr(app_mod.psycopg2, "connect", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert client.post("/webhooks/jpmorgan/transactions", headers=_auth(), json={"transaction": {"id": 1}}).status_code == 500
    assert client.post("/webhooks/jpmorgan/accounts", headers=_auth(), json={"account": {"id": 1}}).status_code == 500


def test_additional_error_handlers_and_auth_paths(client, monkeypatch):
    # trigger internal 500 handler route path without late route registration
    monkeypatch.setattr(
        app_mod,
        "convert_data_format_logic",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    r500 = client.post(
        "/data/convert",
        json={"from_format": "json", "to_format": "csv", "data": [{"a": 1}]},
    )
    assert r500.status_code == 500

    # require_auth non-testing branch
    app_mod.app.config["TESTING"] = False
    assert client.post("/telemetry", json={"ok": True}).status_code == 401
    app_mod.app.config["TESTING"] = True

    # token_auth_required non-testing branch
    app_mod.app.config["TESTING"] = False
    assert client.get("/user/profile").status_code == 401
    app_mod.app.config["TESTING"] = True

    # login/register exception handlers
    monkeypatch.setattr(app_mod, "create_user", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert client.post("/user/register", json={"username": "u", "password": "p"}).status_code == 500

    monkeypatch.setattr(app_mod, "verify_user", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert client.post("/user/login", json={"username": "u", "password": "p"}).status_code == 500


def test_to_int_and_dashboard_trends_points_sanitization(client):
    assert app_mod._to_int(None, 7) == 7
    assert app_mod._to_int(True, 0) == 1
    assert app_mod._to_int("12.0", 0) == 12
    assert app_mod._to_int("", 9) == 9
    assert app_mod._to_int("bad", 5) == 5

    assert client.get("/api/dashboard/trends?points=0").status_code == 200
    assert client.get("/api/dashboard/trends?points=999").status_code == 200
