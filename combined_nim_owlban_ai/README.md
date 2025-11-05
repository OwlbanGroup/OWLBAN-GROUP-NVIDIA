# OWLBAN-GROUP-NVIDIA Integration

This package provides integration between NVIDIA NIM and OWLBAN Group AI systems, with quantum acceleration and financial monitoring.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure credentials:

- Copy `config/financial_providers.example.json` to `config/financial_providers.json`
- Add your API keys for Bloomberg, Refinitiv, or other providers
- Optional: Configure Azure Quantum workspace details in environment variables:

```bash
AZURE_QUANTUM_WORKSPACE_ID=your-workspace-id
AZURE_QUANTUM_LOCATION=your-location
AZURE_QUANTUM_RESOURCE_ID=your-resource-id
```

1. Start the quantum monitor:

```bash
python -m combined_nim_owlban_ai.quantum_monitor
```

The monitor will automatically:

- Detect available GPU resources using NVML
- Connect to configured financial data providers
- Start collecting quantum metrics
- Export Prometheus metrics on port 8000 by default

1. View metrics:

- Prometheus metrics available at `http://localhost:8000/metrics`
- Import the Grafana dashboard from `grafana/quantum_financial_dashboard.json`
- Configure Grafana data source named "Prometheus" pointing to the metrics endpoint

## Configuration

See `financial_data_config.py` for data provider configuration options.

Metrics will fall back to simulation if providers are not configured:

- GPU metrics require NVIDIA drivers and NVML
- Financial data requires at least one configured provider
- Quantum metrics will be simulated if no quantum backend is available

## Monitoring Setup

1. Start Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'quantum_monitor'
    static_configs:
      - targets: ['localhost:8000']
```

1. Configure Grafana:

- Add Prometheus data source
- Import dashboard from `grafana/quantum_financial_dashboard.json`
- Optional: Configure alerting rules in Grafana UI

## Architecture

The monitoring system consists of:

- Quantum metrics collection (circuit execution times, error rates)
- GPU resource monitoring via NVML
- Financial data ingestion with provider fallbacks
- Prometheus metric export
- Grafana visualization

Real-time metrics are collected for:

- Quantum circuit latency and error rates
- GPU utilization and memory
- Financial portfolio metrics (VaR, Sharpe ratio)
- Trading volume and system health
