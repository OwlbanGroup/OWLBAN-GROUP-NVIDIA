"""NVIDIA NGC Catalog integration for OWLBAN GROUP AI systems.

This module provides a lightweight, deterministic catalog model derived from the
NVIDIA NGC Catalog content so it can be used inside the automation stack without
requiring live network access at import time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class NGCatalogManager:
    """Manage NVIDIA NGC catalog content and expose it through the app stack."""

    def __init__(self, items: Optional[List[Dict[str, Any]]] = None):
        self.initialized = False
        self.items = items or self._default_items()
        self.logger_name = "NGCatalogManager"

    def initialize(self) -> None:
        """Initialize the catalog manager and mark it ready for use."""
        self.initialized = True

    def get_catalog_items(self) -> List[Dict[str, Any]]:
        """Return the full catalog item list."""
        return [dict(item) for item in self.items]

    def get_featured_items(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Return the most relevant featured items."""
        return [dict(item) for item in self.items[:limit]]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search catalog items by name, category, or tag."""
        if not query:
            return self.get_catalog_items()

        term = query.strip().lower()
        matches: List[Dict[str, Any]] = []
        for item in self.items:
            haystack = " ".join(
                [
                    item.get("name", ""),
                    item.get("category", ""),
                    item.get("type", ""),
                    " ".join(item.get("tags", [])),
                    item.get("summary", ""),
                ]
            ).lower()
            if term in haystack:
                matches.append(dict(item))
        return matches

    def get_catalog_summary(self) -> Dict[str, Any]:
        """Return a summary object that can be consumed by APIs and dashboards."""
        featured = self.get_featured_items(limit=6)
        return {
            "initialized": self.initialized,
            "featured_count": len(featured),
            "featured_items": featured,
            "top_technology": [
                item for item in featured if item.get("category") == "Technology"
            ],
            "use_cases": [
                "Natural Language Processing",
                "Video Analytics",
                "Speech Recognition",
                "Simulation and Modeling",
            ],
            "source": "NVIDIA NGC Catalog",
        }

    @staticmethod
    def _default_items() -> List[Dict[str, Any]]:
        return [
            {
                "name": "NVIDIA PyTorch",
                "type": "Container",
                "category": "Technology",
                "summary": "GPU-accelerated tensor framework for deep learning and scientific computing.",
                "tags": ["pytorch", "deep-learning", "gpu"],
            },
            {
                "name": "NVIDIA Triton Inference Server",
                "type": "Container",
                "category": "Technology",
                "summary": "Deploy trained AI models from any framework on GPU or CPU infrastructure.",
                "tags": ["inference", "deployment", "gpu"],
            },
            {
                "name": "NVIDIA CUDA",
                "type": "Container",
                "category": "Technology",
                "summary": "Official CUDA container registry for accelerated computing workloads.",
                "tags": ["cuda", "containers", "compute"],
            },
            {
                "name": "NVIDIA DeepSeek-R1 NIM",
                "type": "Container",
                "category": "Model",
                "summary": "NVIDIA NIM for GPU-accelerated DeepSeek-R1 inference through OpenAI-compatible APIs.",
                "tags": ["nim", "llm", "inference"],
            },
            {
                "name": "NVIDIA Cosmos World Foundation Models",
                "type": "Collection",
                "category": "Model",
                "summary": "Physics-aware world foundation models for physical AI and simulation workflows.",
                "tags": ["simulation", "world-models", "physical-ai"],
            },
            {
                "name": "NVIDIA DeepStream SDK",
                "type": "Container",
                "category": "Technology",
                "summary": "Accelerated video analytics and intelligent video applications.",
                "tags": ["video-analytics", "computer-vision", "ivs"],
            },
        ]
