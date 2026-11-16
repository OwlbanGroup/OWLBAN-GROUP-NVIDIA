"""Lightweight NGC catalog manager for the public API surface."""

from typing import Any, Dict, List


class NGCatalogManager:
    def __init__(self):
        self.catalog = [
            {"name": "owlbannim-base", "type": "model", "status": "available"},
            {"name": "owlbannim-vision", "type": "vision", "status": "available"},
        ]

    def initialize(self):
        return True

    def get_catalog_summary(self) -> Dict[str, Any]:
        return {"count": len(self.catalog), "models": self.catalog}

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = (query or "").lower()
        return [item for item in self.catalog if q in item.get("name", "").lower() or q in item.get("type", "").lower()]
