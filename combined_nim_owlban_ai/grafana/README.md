Quantum Financial Monitoring — Grafana Dashboard

Files:
- `quantum_financial_dashboard.json` — Grafana dashboard export configured to read Prometheus metrics produced by the monitoring system.

How to import

1. Start Grafana and ensure Prometheus is added as a data source named "Prometheus".
2. In Grafana UI, go to "Create" -> "Import".
3. Upload `quantum_financial_dashboard.json` or paste its contents.
4. Select the Prometheus data source when prompted and import.

Notes
- The dashboard uses the Prometheus metric names defined in `combined_nim_owlban_ai/quantum_monitor.py` (e.g., `quantum_circuit_latency_seconds_bucket`, `quantum_error_rate`, `gpu_utilization_percent`, `portfolio_performance`, `trading_volume_total`).
- If your Prometheus instance uses a different data source name, choose it during import or edit the dashboard JSON.
- Refresh interval is set to 5s; adjust as needed for your environment.

Quick run (example Docker Prometheus + Grafana)

# Start Prometheus and Grafana (example using Docker Compose)
# See your infra docs for a production-ready setup.

# After Grafana is up, import the dashboard via the UI as described above.
