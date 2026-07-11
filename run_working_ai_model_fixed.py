<![CDATA[#!/usr/bin/env python3
"""
Working AI model runner (fixed script).

Runs: performance_optimization/advanced_anomaly_detection.py
Class: AdvancedAnomalyDetection
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_working_ai_model_fixed")


def main() -> int:
    try:
        from performance_optimization.advanced_anomaly_detection import AdvancedAnomalyDetection
    except Exception as e:
        logger.error("Failed to import AdvancedAnomalyDetection: %s", e)
        return 2

    detector = AdvancedAnomalyDetection(use_gpu=True)
    logger.info("Detector GPU status: %s", detector.get_gpu_status())

    sample_vector: List[float] = [
        0.10, 0.25, 0.40, 0.55, 0.60, 0.70, 0.80, 0.65, 0.50, 0.30
    ]
    is_anomaly_vec, reconstruction_error_vec = detector.detect(sample_vector)
    print("\n=== Anomaly Detection (vector) ===")
    print("is_anomaly:", is_anomaly_vec)
    print("reconstruction_error:", reconstruction_error_vec)

    sample_dict: Dict[str, Any] = {
        "GPU_Usage%": "45%",
        "GPU_MemoryGB": "12GB",
        "CPU_Usage%": "30%",
        "CPU_MemoryGB": "8GB",
        "Other_Feature": "some_string_value",
    }
    is_anomaly_dict, reconstruction_error_dict = detector.detect(sample_dict)
    print("\n=== Anomaly Detection (dict) ===")
    print("is_anomaly:", is_anomaly_dict)
    print("reconstruction_error:", reconstruction_error_dict)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
]]>
