# JPMorgan Financial APIs - Live Transactional Monitoring Setup

This guide provides complete instructions for deploying live transactional data to Grafana dashboard with 100% real-time visibility.

## 🏗️ Architecture Overview

```
Flask Application (Port 5000)
    ↓ /metrics endpoint
Prometheus (Port 9090)
    ↓ scrapes metrics
Grafana (Port 3000)
    ↓ visualizes data
Live Dashboard (100% Real-time)
```

## 📋 Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ with Flask application
- At least 4GB RAM available
- Ports 3000, 9090, 8080, 9100 available

## 🚀 Quick Start Deployment

### 1. One-Command Deployment

```bash
cd jpmorgan_financial_apis
chmod +x deploy-monitoring.sh
./deploy-monitoring.sh
```

### 2. Manual Deployment (Alternative)

```bash
# Start monitoring stack
docker-compose up -d

# Wait for services to be ready (2-3 minutes)
# Then start your Flask application
python app.py
```

## 📊 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Grafana Dashboard** | http://localhost:3000 | Live transactional monitoring |
| **Prometheus** | http://localhost:9090 | Metrics collection & querying |
| **Node Exporter** | http://localhost:9100 | System metrics |
| **cAdvisor** | http://localhost:8080 | Container metrics |
| **Flask Metrics** | http://localhost:5000/metrics | Application metrics |

**Default Credentials:**
- Grafana: `admin` / `admin`

## 📈 Live Transactional Metrics

The dashboard provides 100% real-time visibility of:

### Application Metrics
- ✅ HTTP request rate by endpoint
- ✅ Request latency (95th percentile)
- ✅ Error rates and types
- ✅ Active connections
- ✅ Telemetry events processed

### System Metrics
- ✅ CPU usage percentage
- ✅ Memory usage percentage
- ✅ Disk I/O operations
- ✅ Network traffic (RX/TX)

### Container Metrics
- ✅ Container CPU usage
- ✅ Container memory usage
- ✅ Container network I/O

## 🔧 Configuration Files

### Prometheus Configuration (`prometheus.yml`)
```yaml
- Job: jpmorgan-apis (scrapes Flask /metrics every 5s)
- Job: node-exporter (system metrics)
- Job: cadvisor (container metrics)
- Job: grafana (Grafana metrics)
```

### Grafana Provisioning
- **Data Source**: Prometheus auto-configured
- **Dashboard**: JPMorgan Live Transactional Dashboard auto-imported

## 🧪 Testing Live Data Flow

### 1. Generate Test Traffic
```bash
# Send test requests to your Flask app
curl http://localhost:5000/health
curl http://localhost:5000/telemetry -X POST -H "Content-Type: application/json" -d '{"test": "data"}'
```

### 2. Verify Metrics Collection
```bash
# Check Prometheus is collecting data
curl "http://localhost:9090/api/v1/query?query=http_requests_total"
```

### 3. View Live Dashboard
1. Open http://localhost:3000
2. Login: admin/admin
3. Navigate to "JPMorgan Financial APIs - Live Transactional Dashboard"
4. Watch real-time metrics update every 30 seconds

## 🔍 Troubleshooting

### Flask App Not Connecting
```bash
# Check if Flask is running
curl http://localhost:5000/health

# If not running, start it
python app.py
```

### Metrics Not Appearing
```bash
# Check Prometheus targets
curl http://localhost:9090/targets

# Check specific metrics
curl "http://localhost:9090/api/v1/query?query=up{job='jpmorgan-apis'}"
```

### Dashboard Not Loading
```bash
# Check Grafana logs
docker-compose logs grafana

# Restart Grafana
docker-compose restart grafana
```

## 📊 Dashboard Panels Explained

| Panel | Metric | Update Frequency |
|-------|--------|------------------|
| Transaction Volume | `rate(http_requests_total[5m])` | Real-time |
| Active Connections | `active_connections` | Real-time |
| Request Rate by Endpoint | `rate(http_requests_total[5m])` | 30s |
| Request Latency | `histogram_quantile(0.95, ...)` | 30s |
| Error Rate | `rate(errors_total[5m])` | 30s |
| CPU Usage | `100 - (avg irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100` | 15s |
| Memory Usage | `100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)` | 15s |

## 🛠️ Maintenance Commands

```bash
# View all service logs
docker-compose logs -f

# Restart specific service
docker-compose restart prometheus
docker-compose restart grafana

# Stop monitoring stack
docker-compose down

# Remove all data (fresh start)
docker-compose down -v
```

## 🎯 Achieving 100% Live Transactional Data

The setup ensures 100% live transactional data visibility through:

1. **5-second scrape intervals** for application metrics
2. **15-second scrape intervals** for system metrics
3. **30-second dashboard refresh** rate
4. **Real-time metric updates** from Flask application
5. **Auto-provisioned dashboards** with pre-configured panels

## 📞 Support

For issues with live transactional data not appearing:
1. Verify Flask app is running and accessible
2. Check Prometheus targets page for connection status
3. Review Grafana data source configuration
4. Check service logs: `docker-compose logs`

---

**🎉 Success**: You now have 100% live transactional data flowing to your Grafana dashboard!
