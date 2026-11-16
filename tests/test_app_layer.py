import json
import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.app_factory import create_app
from app.config import get_settings
from app.logging_config import JsonFormatter
from app.security import _generate_bearer_token
from app.storage import AppStore
from auth_lib import generate_api_key


def test_health_and_optimization_endpoints():
    client = TestClient(create_app())
    settings = get_settings()
    auth = (settings.api_username, settings.api_password)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["x-request-id"]
    payload = health.json()
    assert payload["status"] == "ok"

    optimize = client.post(
        "/ai/optimize",
        json={
            "assets": [
                {"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92},
                {"symbol": "MSFT", "value": 130000.0, "volatility": 0.14, "score": 0.88},
            ],
            "iterations": 2,
            "target_return": 0.1,
        },
        auth=auth,
    )
    assert optimize.status_code == 200
    body = optimize.json()
    assert body["best_asset"] in {"NVDA", "MSFT"}
    assert body["metrics"]["asset_count"] == 2

    risk = client.post(
        "/ai/risk",
        json={
            "portfolio": [
                {"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92},
            ],
            "stress_scenario": "stress",
        },
        auth=auth,
    )
    assert risk.status_code == 200
    assert risk.json()["status"] == "ok"

    quantum = client.post("/quantum/demo", auth=auth)
    assert quantum.status_code == 200
    assert quantum.json()["status"] == "ok"

    bearer_token = _generate_bearer_token(settings.api_username)
    bearer = client.post(
        "/ai/optimize",
        json={
            "assets": [{"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92}],
            "iterations": 1,
            "target_return": 0.1,
        },
        headers={"Authorization": f"Bearer {bearer_token}"},
    )
    assert bearer.status_code == 200

    api_key = generate_api_key("admin@owlban.com", name="test_key")
    key_auth = client.post(
        "/ai/risk",
        json={
            "portfolio": [{"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92}],
            "stress_scenario": "stress",
        },
        headers={"X-API-Key": api_key},
    )
    assert key_auth.status_code == 200


def test_protected_endpoint_rejects_missing_credentials():
    response = TestClient(create_app()).post(
        "/ai/optimize",
        json={
            "assets": [{"symbol": "NVDA", "value": 150000.0}],
            "iterations": 1,
        },
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic")


def test_optimizer_audit_does_not_store_portfolio_payload(tmp_path):
    client = TestClient(create_app())
    settings = get_settings()
    db_path = tmp_path / "events.sqlite"
    client.app.state.store = AppStore(db_path)

    response = client.post(
        "/ai/optimize",
        json={
            "assets": [{"symbol": "PRIVATE-ASSET", "value": 150000.0, "score": 0.9}],
            "iterations": 1,
        },
        auth=(settings.api_username, settings.api_password),
    )
    assert response.status_code == 200

    with sqlite3.connect(db_path) as connection:
        stored_payload = connection.execute(
            "SELECT payload FROM events WHERE event_name = 'ai_optimize'"
        ).fetchone()[0]
    assert "PRIVATE-ASSET" not in stored_payload
    assert json.loads(stored_payload)["asset_count"] == 1


def test_production_settings_require_secrets(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_PASSWORD", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="API_PASSWORD"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_wildcard_cors_origin_is_rejected(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="must not contain"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_json_logging_includes_request_context():
    record = logging.LogRecord(
        name="app.request",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"
    record.method = "GET"
    record.path = "/health"
    record.status_code = 200
    record.duration_ms = 1.25
    formatted = json.loads(JsonFormatter().format(record))
    assert formatted["request_id"] == "request-123"
    assert formatted["status_code"] == 200


def test_rate_limit_returns_429_and_keeps_request_id(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        client = TestClient(create_app())
        first = client.post(
            "/quantum/demo",
            auth=(settings.api_username, settings.api_password),
        )
        second = client.post(
            "/quantum/demo",
            auth=(settings.api_username, settings.api_password),
        )
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.headers["x-request-id"]
    finally:
        get_settings.cache_clear()
