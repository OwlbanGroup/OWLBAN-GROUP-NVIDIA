"""Application service business logic for AI and quantum features."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from combined_nim_owlban_ai import CombinedSystem
except Exception:  # pragma: no cover - optional runtime dependency
    CombinedSystem = None

try:
    from quantum_software import QuantumSoftware
except Exception:  # pragma: no cover - optional runtime dependency
    QuantumSoftware = None


class AIService:
    """Thin, resilient wrapper over the repository's AI and quantum capabilities."""

    def __init__(self) -> None:
        self.combined_system = CombinedSystem() if CombinedSystem is not None else None
        self.quantum_software = QuantumSoftware(qubit_count=2) if QuantumSoftware is not None else None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "ai_available": self.combined_system is not None,
            "quantum_available": self.quantum_software is not None,
            "environment": "production" if self.combined_system is not None else "safe-mode",
        }

    def optimize_assets(self, assets: List[Dict[str, Any]], iterations: int = 3, target_return: float = 0.0) -> Dict[str, Any]:
        scores = [float(item.get("score", 0.5)) for item in assets]
        best_index = max(range(len(scores)), key=lambda idx: scores[idx]) if scores else 0
        best_score = scores[best_index] if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0

        result = {
            "status": "ok",
            "best_asset": assets[best_index].get("symbol", "N/A") if assets else "N/A",
            "best_score": best_score,
            "mean_score": mean_score,
            "winner_index": best_index,
            "metrics": {
                "iterations": iterations,
                "target_return": target_return,
                "asset_count": len(assets),
                "volatility_sum": sum(float(item.get("volatility", 0.0)) for item in assets),
                "weights": [float(item.get("value", 0.0)) for item in assets],
            },
        }

        if self.quantum_software is not None:
            try:
                quantum_result = self.quantum_software.grover_search(scores or [0.5], iterations=max(1, iterations))
                result["metrics"]["quantum_winner"] = int(quantum_result.get("winner", best_index))
                result["metrics"]["quantum_best_value"] = float(quantum_result.get("winner_value", best_score))
            except Exception:  # pragma: no cover - best effort
                pass

        return result

    def evaluate_risk(self, portfolio: List[Dict[str, Any]], stress_scenario: str = "baseline") -> Dict[str, Any]:
        if not portfolio:
            return {"status": "ok", "risk_score": 0.0, "scenario": stress_scenario, "recommendation": "No portfolio data provided."}

        total_value = sum(float(item.get("value", 0.0)) for item in portfolio)
        total_volatility = sum(float(item.get("volatility", 0.0)) * float(item.get("value", 0.0)) for item in portfolio)
        risk_score = (total_volatility / total_value) if total_value else 0.0
        factor = {"baseline": 1.0, "stress": 1.25, "shock": 1.5}.get(stress_scenario.lower(), 1.0)
        adjusted_risk = min(1.0, risk_score * factor)
        recommendation = "Diversify into lower-volatility assets" if adjusted_risk > 0.55 else "Current portfolio is within risk tolerance"
        return {
            "status": "ok",
            "risk_score": round(adjusted_risk, 4),
            "scenario": stress_scenario,
            "total_value": total_value,
            "recommendation": recommendation,
        }

    def run_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.combined_system is not None:
            try:
                candidate = getattr(self.combined_system, "run_inference", None)
                if callable(candidate):
                    return candidate(payload)
            except Exception:  # pragma: no cover - best effort
                pass

        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        if isinstance(metrics, dict):
            items = list(metrics.values())
            if items:
                score = float(sum(float(item) for item in items if isinstance(item, (int, float))) / len(items))
            else:
                score = 0.5
        else:
            score = 0.5

        return {"status": "ok", "summary": "CPU-safe AI fallback result", "score": round(score, 4), "source": "fallback"}

    def quantum_demo(self) -> Dict[str, Any]:
        if self.quantum_software is None:
            return {"status": "ok", "message": "Quantum module not available in this environment", "source": "fallback"}

        bell_pair = self.quantum_software.bell_pair()
        probabilities = bell_pair.probabilities()
        return {
            "status": "ok",
            "message": "Bell state prepared successfully",
            "probabilities": [float(value) for value in probabilities],
            "source": "quantum_software",
        }
