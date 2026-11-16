"""Predictive Medicine Oracle.

This module provides a lightweight, deterministic clinical-risk assessment engine
that remains usable in CPU-only environments.
"""

import logging
from typing import Any, Dict, Iterable, Mapping, Optional


class PredictiveMedicineOracle:
    """Generates a preventive health-risk assessment from patient vitals."""

    def __init__(self, triton_server=None, rapids_processor=None, dcgm_monitor=None):
        self.logger = logging.getLogger("PredictiveMedicineOracle")
        self.triton = triton_server
        self.rapids_processor = rapids_processor
        self.dcgm_monitor = dcgm_monitor

    def _normalize_vitals(self, vitals: Mapping[str, Any]) -> Dict[str, float]:
        normalized = {}
        for key, value in (vitals or {}).items():
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                normalized[key] = 0.0
        return normalized

    def predict_risk(self, vitals: Mapping[str, Any]) -> Dict[str, Any]:
        """Forecast a risk score and recommended intervention."""
        values = self._normalize_vitals(vitals)
        if not values:
            return {"risk_score": 0.0, "risk_level": "low", "recommendation": "No patient data provided."}

        risk_score = 0.0
        if "heart_rate" in values:
            risk_score += max(0.0, abs(values["heart_rate"] - 72.0)) / 30.0
        if "blood_pressure" in values:
            risk_score += max(0.0, abs(values["blood_pressure"] - 120.0)) / 40.0
        if "temperature" in values:
            risk_score += max(0.0, abs(values["temperature"] - 98.6)) / 5.0
        if "oxygen_level" in values:
            risk_score += max(0.0, (100.0 - values["oxygen_level"]) / 12.0)

        risk_score = min(risk_score, 1.0)
        if risk_score >= 0.75:
            risk_level = "critical"
            recommendation = "Immediate clinical evaluation and intervention recommended."
        elif risk_score >= 0.45:
            risk_level = "moderate"
            recommendation = "Scheduled follow-up and additional monitoring are recommended."
        else:
            risk_level = "low"
            recommendation = "Continue routine monitoring and preventative care."

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "vitals": values,
        }

    def monitor_vitals(self, vitals: Mapping[str, Any]) -> Dict[str, Any]:
        return self.predict_risk(vitals)

    def generate_health_report(self, vitals: Mapping[str, Any]) -> Dict[str, Any]:
        result = self.predict_risk(vitals)
        result["model"] = "PredictiveMedicineOracle"
        return result
