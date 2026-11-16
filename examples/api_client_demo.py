"""Example client usage for the production OWLBAN FastAPI layer."""

from __future__ import annotations

import json

import requests

from app.config import get_settings


def call_api(base_url: str = "http://127.0.0.1:8000") -> dict:
    settings = get_settings()
    auth = (settings.api_username, settings.api_password)

    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()

    payload = {
        "assets": [
            {"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92},
            {"symbol": "MSFT", "value": 130000.0, "volatility": 0.14, "score": 0.88},
            {"symbol": "AAPL", "value": 100000.0, "volatility": 0.16, "score": 0.81},
        ],
        "iterations": 3,
        "target_return": 0.1,
    }
    optimize = requests.post(f"{base_url}/ai/optimize", json=payload, auth=auth, timeout=10)
    optimize.raise_for_status()

    risk = requests.post(
        f"{base_url}/ai/risk",
        json={
            "portfolio": payload["assets"],
            "stress_scenario": "stress",
        },
        auth=auth,
        timeout=10,
    )
    risk.raise_for_status()

    return {
        "health": health.json(),
        "optimization": optimize.json(),
        "risk": risk.json(),
    }


if __name__ == "__main__":
    try:
        result = call_api()
        print(json.dumps(result, indent=2))
    except Exception as exc:  # pragma: no cover
        print(f"Unable to reach API: {exc}")
