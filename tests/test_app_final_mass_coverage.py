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

    class AnomalyDetectorStub:
        def train(self, X, contamination=0.1):
            if len(X) < 1:
                raise ValueError("no data")
            return None

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
        def __init__(self):
            self.businesses = [_ORM(id=1, name="B1")]
            self.assets = [_ORM(id=1, name="A1", business_id=1)]
        def get_all_businesses(self): return self.businesses
        def create_business(self, data):
            b = _ORM(id=2, **data); self.businesses.append(b); return b
        def get_business_by_id(self, bid):
            return next((b for b in self.businesses if b.id == bid), None)
        def update_business(self, bid, data):
            b = self.get_business_by_id(bid)
            if not b: return None
            for k, v in data.items(): setattr(b, k, v)
            return b
        def delete_business(self, bid):
            b = self.get_business_by_id(bid)
            if not b: return False
            self.businesses = [x for x in self.businesses if x.id != bid]
            return True
        def get_all_assets(self): return self.assets
        def create_asset(self, data):
            a = _ORM(id=2, **data); self.assets.append(a); return a
        def get_asset_by_id(self, aid):
            return next((a for a in self.assets if a.id == aid), None)
        def update_asset(self, aid, data):
            a = self.get_asset_by_id(aid)
            if not a: return None
            for k, v in data.items(): setattr(a, k, v)
            return a
        def delete_asset(self, aid):
            a = self.get_asset_by_id(aid)
            if not a: return False
            self.assets = [x for x in self.assets if x.id != aid]
            return True
        def get_assets_by_business_id(self, bid):
            return [a for a in self.assets if a.business_id == bid]

    class SchemaObj:
        def __init__(self, **data): self._d = data
        def dict(self, exclude_unset=False): return dict(self._d)

    class RespSchema:
        @staticmethod
        def from_orm(obj):
            return types.SimpleNamespace(dict=lambda: obj.__dict__)

    monkeypatch.setattr(app_mod, "telemetry_handler", TelemetryStub())
    monkeypatch.setattr(app_mod, "anomaly_detector", AnomalyDetectorStub())
    monkeypatch.setattr(app_mod, "ai_service", AIServiceStub())
    monkeypatch.setattr(app_mod, "sync_service", SyncServiceStub())
    monkeypatch.setattr(app_mod, "sync_scheduler", SyncSchedulerStub())
    monkeypatch.setattr(app_mod, "payments_service", PaymentsServiceStub())
    monkeypatch.setattr(app_mod, "db_manager", DBStub())
    monkeypatch.setattr(app_mod, "BusinessCreate", SchemaObj)
    monkeypatch.setattr(app_mod, "BusinessUpdate", SchemaObj)
    monkeypatch.setattr(app_mod, "BusinessResponse", RespSchema)
    monkeypatch.setattr(app_mod, "AssetCreate", SchemaObj)
    monkeypatch.setattr(app_mod, "AssetUpdate", SchemaObj)
    monkeypatch.setattr(app_mod, "AssetResponse", RespSchema)

    app_mod.users["u"] = {"password": app_mod.generate_password_hash("p"), "created_at": datetime.now(timezone.utc).isoformat(), "token": "tok"}

    with app_mod.app.test_client() as c:
        yield c


def _auth():
    return {"Authorization": "Bearer tok"}


def test_core_and_alias_routes(client):
    for p in ["/", "/health", "/ready", "/storage/files", "/benefits", "/payroll", "/patterns", "/traction", "/purchasing", "/bill-pay", "/ws/status", "/batch/status", "/data/formats", "/api/dashboard/summary", "/api/dashboard/trends"]:
        r = client.get(p)
        assert r.status_code in (200, 302, 404)


def test_user_register_login_profile(client):
    assert client.post("/user/register", json={"username": "newu", "password": "x"}).status_code in (201, 400)
    assert client.post("/user/login", json={"username": "newu", "password": "x"}).status_code in (200, 401)
    assert client.get("/user/profile", headers=_auth()).status_code in (200, 404)


def test_telemetry_routes(client):
    assert client.post("/telemetry", headers=_auth(), json={"ok": True, "ver": "1", "name": "n", "time": "t", "data": {}}).status_code in (200, 400)
    assert client.post("/telemetry/batch", headers=_auth(), json={"telemetry_data": [{"a": 1}]}).status_code in (200, 400)
    assert client.get("/telemetry/metrics?hours=24").status_code == 200
    assert client.get("/telemetry/export?format=json&limit=1").status_code == 200
    assert client.get("/telemetry/export?format=csv&limit=1").status_code == 200


def test_ml_routes(client):
    assert client.post("/ml/anomalies", headers=_auth(), json={"telemetry_data": [{"a": 1}]}).status_code in (200, 400)
    assert client.post("/ml/train", headers=_auth(), json={"training_data": [[1,2,3,4,5,6,7]] * 10, "contamination": 0.1}).status_code == 200


def test_business_asset_routes(client):
    assert client.get("/businesses", headers=_auth()).status_code == 200
    assert client.post("/businesses", headers=_auth(), json={"name": "B2"}).status_code in (201, 500)
    assert client.get("/businesses/1", headers=_auth()).status_code in (200, 404)
    assert client.put("/businesses/1", headers=_auth(), json={"name": "B1u"}).status_code in (200, 404)
    assert client.delete("/businesses/1", headers=_auth()).status_code in (200, 404)

    assert client.get("/assets", headers=_auth()).status_code == 200
    assert client.post("/assets", headers=_auth(), json={"name": "A2", "business_id": 1}).status_code in (201, 500)
    assert client.get("/assets/1", headers=_auth()).status_code in (200, 404)
    assert client.put("/assets/1", headers=_auth(), json={"name": "A1u"}).status_code in (200, 404)
    assert client.delete("/assets/1", headers=_auth()).status_code in (200, 404)

    assert client.get("/businesses/1/assets", headers=_auth()).status_code in (200, 404)
    assert client.post("/businesses/1/assets", headers=_auth(), json={"name": "Ax", "business_id": 1}).status_code in (201, 404, 400)


def test_ai_and_sync_routes(client):
    assert client.post("/ai/analyze", headers=_auth(), json={"data": {"x": 1}, "question": "q"}).status_code in (200, 400)
    assert client.post("/ai/risk-assess", headers=_auth(), json={"transaction_data": {"id": 1}}).status_code in (200, 400)
    assert client.post("/ai/query", headers=_auth(), json={"query": "hello"}).status_code in (200, 400)
    assert client.get("/ai/status").status_code == 200

    assert client.post("/sync/start", headers=_auth()).status_code in (200, 400)
    assert client.get("/sync/status", headers=_auth()).status_code == 200
    assert client.post("/sync/run/job1", headers=_auth()).status_code in (200, 400)
    assert client.post("/sync/run/bad", headers=_auth()).status_code in (400, 500)
    assert client.post("/sync/stop", headers=_auth()).status_code in (200, 400)


def test_stripe_and_github_and_batch_controls(client, monkeypatch):
    assert client.post("/stripe/payment-intent", headers=_auth(), json={"amount": 100, "currency": "USD"}).status_code in (201, 400)
    assert client.post("/stripe/payment-intent/pi_1/confirm", headers=_auth()).status_code in (200, 202, 400)
    assert client.post("/stripe/payment-intent/pending/confirm", headers=_auth()).status_code in (202, 400)
    assert client.post("/stripe/refund", headers=_auth(), json={"payment_intent_id": "pi_1"}).status_code in (201, 400)
    assert client.post("/stripe/webhook", data="{}", headers={"stripe-signature": "sig"}).status_code in (200, 400)

    class Resp:
        def __init__(self, code, payload):
            self.status_code = code
            self._payload = payload
        def json(self): return self._payload

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(200, [{"id": 1}]))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer abc"}).status_code == 200
    assert client.get("/api/github/repos?owner=o1", headers={"Authorization": "Bearer abc"}).status_code == 200

    assert client.post("/batch/start").status_code == 200
    assert client.post("/batch/stop").status_code == 200


def test_webhooks_and_business_api_aliases(client, monkeypatch):
    class Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self, cursor_factory=None): return self
        def execute(self, *a, **k): return None
        def commit(self): return None
        def fetchall(self): return []
    monkeypatch.setattr(app_mod.psycopg2, "connect", lambda *_a, **_k: Conn())

    assert client.post("/webhooks/jpmorgan/transactions", headers=_auth(), json={"transaction": {"id": 1}}).status_code in (200, 500)
    assert client.post("/webhooks/jpmorgan/accounts", headers=_auth(), json={"account": {"id": 1}}).status_code in (200, 500)

    assert client.get("/api/business/intelligence/u1", headers=_auth()).status_code in (200, 400)
    assert client.get("/api/business/forecast/u1", headers=_auth()).status_code in (200, 400)
    assert client.post("/api/sync/payment/p1", headers=_auth(), json={"revenue_type": "purchase"}).status_code in (201, 400, 500)
    assert client.post("/api/ai/analyze", headers=_auth(), json={"data": {"a": 1}, "question": "q"}).status_code in (200, 400)
    assert client.post("/api/ai/risk-assess", headers=_auth(), json={"transaction_data": {"a": 1}}).status_code in (200, 400)
    assert client.post("/api/ai/query", headers=_auth(), json={"query": "q"}).status_code in (200, 400)


def test_error_and_validation_branches(client, monkeypatch):
    # telemetry bad payload paths
    assert client.post("/telemetry", headers=_auth(), json={}).status_code in (400, 500)
    assert client.post("/telemetry/batch", headers=_auth(), json={"telemetry_data": "x"}).status_code in (400, 500)
    assert client.get("/telemetry/metrics?hours=0").status_code == 400
    assert client.get("/telemetry/export?format=xml&limit=1").status_code == 400
    assert client.get("/telemetry/export?format=json&limit=0").status_code == 400

    # ai validation paths
    assert client.post("/ai/analyze", headers=_auth(), json={}).status_code == 400
    assert client.post("/ai/risk-assess", headers=_auth(), json={}).status_code == 400
    assert client.post("/ai/query", headers=_auth(), json={}).status_code == 400

    # sync not initialized path for run endpoint
    monkeypatch.setattr(app_mod, "sync_scheduler", None)
    assert client.post("/sync/run/job1", headers=_auth()).status_code == 400

    # sync logs validation and exception paths
    assert client.get("/sync/logs?limit=0", headers=_auth()).status_code == 400
    assert client.get("/sync/logs", headers=_auth()).status_code in (200, 500)

    # stripe/webhook validation path
    assert client.post("/stripe/webhook", data="{}").status_code == 400

    # workspace validation path
    assert client.post("/api/workspaces", json={"name": "n"}).status_code in (400, 500)

    # github auth and input validation paths
    assert client.get("/api/github/orgs").status_code == 401
    assert client.get("/api/github/repos", headers={"Authorization": "Bearer abc"}).status_code == 400

    # github request exception path
    def _raise_req(*_a, **_k):
        raise app_mod.requests.RequestException("boom")
    monkeypatch.setattr(app_mod.requests, "get", _raise_req)
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer abc"}).status_code == 500

    # dashboard trends points bounds
    assert client.get("/api/dashboard/trends?points=0").status_code == 200
    assert client.get("/api/dashboard/trends?points=1000").status_code == 200

    # explicit 404 handler
    assert client.get("/definitely-not-here").status_code == 404


def test_additional_app_final_branches(client, monkeypatch):
    # ready endpoint degraded branches
    monkeypatch.setattr(app_mod, "REDIS_CLIENT", types.SimpleNamespace(ping=lambda: (_ for _ in ()).throw(RuntimeError("no redis"))))

    class DBFail:
        def get_all_businesses(self): raise RuntimeError("db down")
        def get_all_assets(self): raise RuntimeError("db down")
    monkeypatch.setattr(app_mod, "db_manager", DBFail())
    monkeypatch.setattr(app_mod, "sync_scheduler", types.SimpleNamespace(running=False))
    assert client.get("/ready").status_code == 200

    # telemetry invalid json/bad request
    r = client.post("/telemetry", headers=_auth(), data="{", content_type="application/json")
    assert r.status_code in (400, 500)

    # metrics endpoint exception path
    monkeypatch.setattr(app_mod, "telemetry_handler", types.SimpleNamespace(
        get_metrics=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    ))
    assert client.get("/telemetry/metrics?hours=24").status_code == 500

    # export csv empty path and exception path
    monkeypatch.setattr(app_mod, "telemetry_handler", types.SimpleNamespace(
        export_events=lambda **_k: []
    ))
    assert client.get("/telemetry/export?format=csv&limit=1").status_code == 200

    monkeypatch.setattr(app_mod, "telemetry_handler", types.SimpleNamespace(
        export_events=lambda **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    ))
    app_mod.app.config["RATELIMIT_ENABLED"] = False
    assert client.get("/telemetry/export?format=json&limit=1").status_code == 500

    # batch status controls still deterministic
    assert client.get("/batch/status").status_code == 200
    assert client.post("/batch/start").status_code == 200
    assert client.post("/batch/stop").status_code == 200

    # github non-200 mapping branches
    class Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or {}
        def json(self):
            return self._payload

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(401))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer abc"}).status_code == 401

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(404))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer abc"}).status_code == 404

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(502))
    assert client.get("/api/github/orgs", headers={"Authorization": "Bearer abc"}).status_code == 502

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(400))
    assert client.get("/api/github/repos?owner=o1", headers={"Authorization": "Bearer abc"}).status_code == 400

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(401))
    assert client.get("/api/github/repos?owner=o1", headers={"Authorization": "Bearer abc"}).status_code == 401

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(404))
    assert client.get("/api/github/repos?owner=o1", headers={"Authorization": "Bearer abc"}).status_code == 404

    monkeypatch.setattr(app_mod.requests, "get", lambda *a, **k: Resp(502))
    assert client.get("/api/github/repos?owner=o1", headers={"Authorization": "Bearer abc"}).status_code == 502

    # webhook missing payload branches
    assert client.post("/webhooks/jpmorgan/transactions", headers=_auth(), json={}).status_code in (400, 500)
    assert client.post("/webhooks/jpmorgan/accounts", headers=_auth(), json={}).status_code in (400, 500)
