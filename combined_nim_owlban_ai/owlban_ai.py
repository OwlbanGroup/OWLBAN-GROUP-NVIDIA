"""Minimal OWLBAN inference engine used by the public package API."""

import logging
from typing import Any, Dict

import numpy as np


class OwlbanAI:
    """Small CPU-safe inference wrapper compatible with the project tests."""

    def __init__(self):
        self.logger = logging.getLogger("OwlbanAI")
        self._loaded = False

    def load_models(self):
        self._loaded = True
        return True

    def run_inference(self, data: Any) -> Dict[str, Any]:
        values = self._flatten_numeric(data)
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            prediction = 0.0
            confidence = 0.5
        elif arr.ndim == 0:
            prediction = float(arr)
            confidence = 0.9
        else:
            prediction = float(np.mean(arr))
            confidence = 0.85 if np.isfinite(prediction) else 0.0

        return {"prediction": prediction, "confidence": confidence}

    def _flatten_numeric(self, data: Any):
        if isinstance(data, dict):
            values = []
            for value in data.values():
                values.extend(self._flatten_numeric(value))
            return values
        if isinstance(data, (list, tuple)):
            values = []
            for value in data:
                values.extend(self._flatten_numeric(value))
            return values
        if isinstance(data, np.ndarray):
            return [float(item) for item in data.reshape(-1).tolist() if np.isfinite(item)]
        if isinstance(data, (int, float, np.number)):
            return [float(data)]
        return []

    def predict(self, data: Any) -> Dict[str, Any]:
        return self.run_inference(data)
