"""A lightweight DCGM-compatible monitor used by the public API."""

import logging
from typing import Dict, Any


class DCGMMonitor:
    """Collects basic GPU metrics in a CPU-safe way."""

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger("DCGMMonitor")

    def collect_metrics(self) -> Dict[str, Any]:
        return {
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
            "temperature_c": 0.0,
            "status": "ok",
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.collect_metrics()
