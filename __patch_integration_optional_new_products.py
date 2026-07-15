from pathlib import Path

p = Path("combined_nim_owlban_ai/integration.py")
s = p.read_text(encoding="utf-8")

old = "\nfrom new_products.infrastructure_optimizer import InfrastructureOptimizer\nfrom new_products.telehealth_analytics import NVIDIATelehealthAnalytics\nfrom new_products.model_deployment_manager import ModelDeploymentManager\nfrom new_products.anomaly_detection import AnomalyDetection\nfrom new_products.revenue_optimizer import RevenueOptimizer\nfrom new_products.stripe_integration import StripeIntegration\n"

if old not in s:
    raise SystemExit("Target new_products import block not found")

new = """
try:
    from new_products.infrastructure_optimizer import InfrastructureOptimizer
    from new_products.telehealth_analytics import NVIDIATelehealthAnalytics
    from new_products.model_deployment_manager import ModelDeploymentManager
    from new_products.anomaly_detection import AnomalyDetection
    from new_products.revenue_optimizer import RevenueOptimizer
    from new_products.stripe_integration import StripeIntegration
except ModuleNotFoundError:
    # Optional dependencies: allow the integration module to be imported in minimal test environments.
    InfrastructureOptimizer = None  # type: ignore
    NVIDIATelehealthAnalytics = None  # type: ignore
    ModelDeploymentManager = None  # type: ignore
    AnomalyDetection = None  # type: ignore
    RevenueOptimizer = None  # type: ignore
    StripeIntegration = None  # type: ignore
""".lstrip("\n")

p.write_text(s.replace(old, "\n" + new, 1), encoding="utf-8")
print("Patched new_products imports to be optional")
