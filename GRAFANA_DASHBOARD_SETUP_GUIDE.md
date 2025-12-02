# 📊 Grafana Dashboard Setup Guide

**Complete guide to setting up and using Grafana dashboards for JPMorgan Financial APIs**

---

## 🚀 Quick Start

### Access Grafana
```
URL: http://localhost:3000
Username: admin
Password: SecureGrafanaP@ss2024
```

---

## 📋 TABLE OF CONTENTS

1. [First Time Setup](#first-time-setup)
2. [Add Prometheus Data Source](#add-prometheus-data-source)
3. [Create Your First Dashboard](#create-your-first-dashboard)
4. [Pre-Built Dashboard Panels](#pre-built-dashboard-panels)
5. [Import Existing Dashboard](#import-existing-dashboard)
6. [Dashboard Best Practices](#dashboard-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 FIRST TIME SETUP

### Step 1: Login to Grafana

1. Open your browser and navigate to: http://localhost:3000
2. Login with credentials:
   - Username: `admin`
   - Password: `SecureGrafanaP@ss2024`
3. (Optional) Change your password on first login

### Step 2: Verify Grafana is Running

```powershell
# Check Grafana health
Invoke-WebRequest http://localhost:3000/api/health

# Expected response: {"database":"ok","version":"..."}
```

---

## 🔌 ADD PROMETHEUS DATA SOURCE

### Method 1: Via Web UI (Recommended)

1. **Navigate to Data Sources**
   - Click the gear icon (⚙️) in the left sidebar
   - Click "Data sources"
   - Click "Add data source"

2. **Select Prometheus**
   - Find and click "Prometheus" from the list

3. **Configure Prometheus**
   ```
   Name: Prometheus
   URL: http://prometheus:9090
   Access: Server (default)
   ```

4. **Save & Test**
   - Scroll down and click "Save & test"
   - You should see: "Data source is working"

### Method 2: Via Configuration File

Create or edit `grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

---

## 📊 CREATE YOUR FIRST DASHBOARD

### Step 1: Create New Dashboard

1. Click the "+" icon in the left sidebar
2. Select "Dashboard"
3. Click "Add new panel"

### Step 2: Add a Panel

**Example: API Request Rate**

1. **Query Configuration:**
   ```promql
   rate(http_requests_total_final[5m])
   ```

2. **Panel Settings:**
   - Title: "API Request Rate"
   - Description: "Requests per second over last 5 minutes"
   - Unit: "requests/sec"

3. **Visualization:**
   - Type: Time series
   - Legend: Show
   - Tooltip: All series

4. **Click "Apply"**

### Step 3: Save Dashboard

1. Click the save icon (💾) at the top
2. Give it a name: "JPMorgan API Monitoring"
3. Click "Save"

---

## 🎨 PRE-BUILT DASHBOARD PANELS

### Panel 1: Service Health Status

**Query:**
```promql
up
```

**Settings:**
- Visualization: Stat
- Title: "Service Status"
- Value mappings:
  - 1 = "UP" (Green)
  - 0 = "DOWN" (Red)

---

### Panel 2: API Response Time

**Query:**
```promql
rate(http_request_duration_seconds_final_sum[5m]) / rate(http_request_duration_seconds_final_count[5m]) * 1000
```

**Settings:**
- Visualization: Time series
- Title: "API Response Time (ms)"
- Unit: milliseconds (ms)
- Thresholds:
  - Green: < 100ms
  - Yellow: 100-500ms
  - Red: > 500ms

---

### Panel 3: Request Rate

**Query:**
```promql
sum(rate(http_requests_total_final[5m]))
```

**Settings:**
- Visualization: Graph
- Title: "Total Request Rate"
- Unit: requests/sec
- Fill: 1
- Line width: 2

---

### Panel 4: Error Rate

**Query:**
```promql
sum(rate(errors_total_final[5m]))
```

**Settings:**
- Visualization: Time series
- Title: "Error Rate"
- Unit: errors/sec
- Color: Red
- Alert threshold: > 0.01

---

### Panel 5: Active Connections

**Query:**
```promql
active_connections_final
```

**Settings:**
- Visualization: Gauge
- Title: "Active Connections"
- Min: 0
- Max: 1000
- Thresholds:
  - Green: 0-500
  - Yellow: 500-800
  - Red: 800-1000

---

### Panel 6: Database Status

**Query:**
```promql
up{job="postgres"}
```

**Settings:**
- Visualization: Stat
- Title: "PostgreSQL Status"
- Value mappings:
  - 1 = "Connected" (Green)
  - 0 = "Disconnected" (Red)

---

### Panel 7: Redis Status

**Query:**
```promql
up{job="redis"}
```

**Settings:**
- Visualization: Stat
- Title: "Redis Status"
- Value mappings:
  - 1 = "Connected" (Green)
  - 0 = "Disconnected" (Red)

---

### Panel 8: CPU Usage

**Query:**
```promql
rate(process_cpu_seconds_total[5m]) * 100
```

**Settings:**
- Visualization: Gauge
- Title: "CPU Usage %"
- Unit: percent (0-100)
- Max: 100

---

### Panel 9: Memory Usage

**Query:**
```promql
process_resident_memory_bytes / 1024 / 1024
```

**Settings:**
- Visualization: Time series
- Title: "Memory Usage (MB)"
- Unit: megabytes (MB)

---

### Panel 10: Telemetry Events

**Query:**
```promql
rate(telemetry_events_processed_total_final[5m])
```

**Settings:**
- Visualization: Stat
- Title: "Telemetry Events/sec"
- Unit: events/sec

---

## 📥 IMPORT EXISTING DASHBOARD

### Method 1: Import from File

1. Click "+" → "Import"
2. Click "Upload JSON file"
3. Select `grafana_dashboard.json` from your project
4. Click "Load"
5. Select "Prometheus" as data source
6. Click "Import"

### Method 2: Import from Grafana.com

1. Click "+" → "Import"
2. Enter dashboard ID (e.g., 1860 for Node Exporter)
3. Click "Load"
4. Select "Prometheus" as data source
5. Click "Import"

### Popular Dashboard IDs:
- **1860** - Node Exporter Full
- **3662** - Prometheus 2.0 Overview
- **7362** - PostgreSQL Database
- **11835** - Redis Dashboard

---

## 🎯 COMPLETE DASHBOARD LAYOUT

### Recommended Dashboard Structure:

```
┌─────────────────────────────────────────────────────┐
│  JPMorgan Financial APIs - Production Monitoring    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Service Status] [API Health] [DB Status] [Redis]  │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [API Response Time Graph - Full Width]             │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Request Rate]          [Error Rate]               │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Active Connections]    [CPU Usage]                │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Memory Usage Graph - Full Width]                  │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Telemetry Events]      [Anomaly Detections]       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ DASHBOARD BEST PRACTICES

### 1. Organization

- **Group related metrics** together
- **Use rows** to organize panels by category
- **Add descriptions** to panels for clarity
- **Use consistent colors** across panels

### 2. Performance

- **Limit time range** for better performance
- **Use appropriate intervals** (5m, 15m, 1h)
- **Avoid too many panels** on one dashboard
- **Use variables** for dynamic filtering

### 3. Visualization

- **Choose appropriate chart types:**
  - Time series: Trends over time
  - Gauge: Current values with thresholds
  - Stat: Single values
  - Table: Multiple metrics
  - Heatmap: Distribution patterns

### 4. Alerts

- **Set meaningful thresholds**
- **Configure notifications** (email, Slack)
- **Test alerts** before production
- **Document alert responses**

---

## 🔧 ADVANCED FEATURES

### Variables

Create dashboard variables for dynamic filtering:

1. **Dashboard Settings** → **Variables** → **Add variable**

2. **Example: Service Variable**
   ```
   Name: service
   Type: Query
   Query: label_values(up, job)
   ```

3. **Use in queries:**
   ```promql
   up{job="$service"}
   ```

### Templating

Use variables in panel titles:
```
Title: $service Status
```

### Annotations

Add event markers to graphs:

1. **Dashboard Settings** → **Annotations**
2. **Add annotation query**
3. **Example: Deployment events**

---

## 📱 MOBILE & SHARING

### Mobile Access

Grafana is mobile-responsive. Access from any device:
```
http://your-server-ip:3000
```

### Share Dashboard

1. Click share icon at top of dashboard
2. Options:
   - **Link:** Share URL
   - **Snapshot:** Create static snapshot
   - **Export:** Download JSON
   - **Embed:** Get iframe code

### Public Dashboards

1. **Dashboard Settings** → **General**
2. Enable "Public dashboard"
3. Copy public URL

---

## 🚨 ALERTS CONFIGURATION

### Create Alert Rule

1. **Edit panel** → **Alert** tab
2. **Create alert rule**

3. **Example: High Response Time Alert**
   ```
   Condition: WHEN avg() OF query(A, 5m, now) IS ABOVE 500
   
   Evaluate every: 1m
   For: 5m
   
   Notification:
   - Send to: Email/Slack
   - Message: "API response time is above 500ms"
   ```

### Alert Channels

1. **Alerting** → **Notification channels**
2. **Add channel**
3. **Types:**
   - Email
   - Slack
   - PagerDuty
   - Webhook
   - Microsoft Teams

---

## 🎨 CUSTOM DASHBOARD JSON

### Complete Dashboard Template

Save this as `custom_dashboard.json`:

```json
{
  "dashboard": {
    "title": "JPMorgan API Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_request_duration_seconds_final_sum[5m]) / rate(http_request_duration_seconds_final_count[5m]) * 1000"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total_final[5m]))"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      }
    ]
  }
}
```

---

## 🔍 TROUBLESHOOTING

### Issue: "Data source is not working"

**Solution:**
```powershell
# Check Prometheus is running
docker ps | Select-String "prometheus"

# Check Prometheus health
Invoke-WebRequest http://localhost:9090/-/healthy

# Verify Grafana can reach Prometheus
docker exec jpmorgan-grafana-prod curl http://prometheus:9090/-/healthy
```

### Issue: "No data points"

**Solutions:**
1. Check time range (top right)
2. Verify query syntax
3. Check if metrics exist:
   ```promql
   {__name__=~".+"}
   ```
4. Verify Prometheus is scraping targets

### Issue: "Panel shows error"

**Solutions:**
1. Check query syntax in Prometheus first
2. Verify data source is selected
3. Check panel time range
4. Review Grafana logs:
   ```powershell
   docker logs jpmorgan-grafana-prod
   ```

### Issue: "Dashboard not saving"

**Solutions:**
1. Check browser console for errors
2. Verify Grafana database is healthy
3. Check disk space
4. Restart Grafana:
   ```powershell
   docker-compose -f docker-compose.production.yml restart grafana
   ```

---

## 📚 LEARNING RESOURCES

### Official Documentation
- **Grafana Docs:** https://grafana.com/docs/
- **Grafana Tutorials:** https://grafana.com/tutorials/
- **Prometheus Queries:** https://prometheus.io/docs/prometheus/latest/querying/basics/

### Video Tutorials
- Grafana Fundamentals: https://grafana.com/tutorials/grafana-fundamentals/
- Creating Dashboards: https://www.youtube.com/watch?v=sKNZMtoSHN4

### Community
- Grafana Community: https://community.grafana.com/
- Dashboard Library: https://grafana.com/grafana/dashboards/

---

## 🚀 QUICK SETUP SCRIPT

### PowerShell Script to Open Grafana

```powershell
# Open Grafana and navigate to dashboard creation
Start-Process "http://localhost:3000"

WriteThe user is asking me to set up the Grafana dashboard. They provided a link to Grafana fundamentals tutorial. I need to create a proper Grafana dashboard configuration for the JPMorgan Financial APIs. Let me check what we have and create a comprehensive dashboard.
<read_file>
<path>c:/Users/bizle/Desktop/jpmorgan_financial_apis/grafana_dashboard.json</path>
</read_file>
