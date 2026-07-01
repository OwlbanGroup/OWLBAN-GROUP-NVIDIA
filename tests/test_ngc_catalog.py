import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combined_nim_owlban_ai.ngc_catalog import NGCatalogManager


def test_catalog_manager_returns_featured_items():
    manager = NGCatalogManager()
    summary = manager.get_catalog_summary()

    assert summary["featured_count"] > 0
    assert len(summary["featured_items"]) == summary["featured_count"]
    assert len(summary["top_technology"]) > 0
    assert len(summary["use_cases"]) > 0
    assert any(item["name"] == "NVIDIA PyTorch" for item in summary["featured_items"])
