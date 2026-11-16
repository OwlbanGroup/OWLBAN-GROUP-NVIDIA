"""Example end-to-end workflow using the production app services."""

from __future__ import annotations

from app.services.ai_service import AIService


def run_example() -> dict:
    service = AIService()
    health = service.health()
    portfolio = [
        {"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92},
        {"symbol": "MSFT", "value": 130000.0, "volatility": 0.14, "score": 0.88},
        {"symbol": "AAPL", "value": 100000.0, "volatility": 0.16, "score": 0.81},
    ]
    optimize = service.optimize_assets(portfolio, iterations=3)
    risk = service.evaluate_risk(portfolio, stress_scenario="stress")
    quantum = service.quantum_demo()

    return {
        "health": health,
        "optimization": optimize,
        "risk": risk,
        "quantum": quantum,
    }


if __name__ == "__main__":
    import json

    result = run_example()
    print(json.dumps(result, indent=2, default=str))
