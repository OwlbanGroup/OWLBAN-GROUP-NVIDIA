# 📊 Prometheus Setup & Usage Guide

## 🎯 Overview

Prometheus is already running in your production environment! This guide will show you how to use it effectively.

---

## ✅ Current Status

Your Prometheus instance is:
- **Running:** ✅ Yes
- **URL:** http://localhost:9090
- **Status:** Healthy
- **Collecting Metrics:** Yes

---

## 🚀 Quick Start - Using Prometheus

### 1. Access Prometheus Web UI

Open your browser and navigate to:
```
http://localhost:9090
```

### 2. Basic Queries

#### Check All Available Metrics
In the Prometheus UI, go to the "Graph" tab and try these queries:

**See all metrics:**
```promql
{__name__=~".+"}
```

**Check which services are up:**
```promql
up
```

**API Request Count:**
```promql
http_requests_total_final
```

**API Response Time:**
```promql
http_request_duration_seconds_final
```

**Active Connections:**
```promql
active_connections_final
```

**Error Count:**
```promql
errors_total_final
```

### 3. Using the Query API

You can also query Prometheus via HTTP:

```powershell
# Check if services are up
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/query?query=up"

# Get API request count
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/query?query=http_requests_total_final"

# Get active connections
Invoke-RestMethod -Uri "http://localhost:9090/api/v1/query?query=active_connections_final"
```

---

## 📊 Common Prometheus Queries (PromQL)

### Service Health Monitoring

```promql
# Check if all services are up
up == 1

# Count how many services are down
count(up == 0)

# Show services that are down
up{job!=""} == 0
```

### API Performance Metrics

```promql
# Total HTTP requests
sum(http_requests_total_final)

# Requests per second (rate over 5 minutes)
rate(http_requests_total_final[5m])

# Average response time
rate(http_request_duration_seconds_final_sum[5m]) / rate(http_request_duration_seconds_final_count[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_final_bucket[5m]))
```

### Error Monitoring

```promql
# Total errors
sum(errors_total_final)

# Error rate (errors per second)
rate(errors_total_final[5m])

# Error rate by type
sum by (type) (rate(errors_total_final[5m]))
```

### Resource Monitoring

```promql
# Active connections
active_connections_final

# Telemetry events processed
telemetry_events_processed_total_final

# Anomaly detections
anomaly_detections_total_final
```

---

## 🎨 Prometheus Web UI Features

### 1. Graph Tab
- **Purpose:** Visualize metrics over time
- **Usage:** Enter a PromQL query and click "Execute"
- **Features:** 
  - Time range selector
  - Graph/Table view toggle
  - Export to CSV

### 2. Alerts Tab
- **Purpose:** View active alerts
- **URL:** http://localhost:9090/alerts
- **Shows:** Current alert status and rules

### 3. Status Tab
- **Purpose:** Check Prometheus configuration
- **Sections:**
  - Runtime & Build Information
  - Command-Line Flags
  - Configuration
  - Rules
  - Targets
  - Service Discovery

### 4. Targets Tab
- **Purpose:** See all monitored endpoints
- **URL:** http://localhost:9090/targets
- **Shows:** Health status of each target

---

## 🔧 Advanced Usage

### Time Series Queries

```promql
# Data from last 5 minutes
http_requests_total_final[5m]

# Data from last hour
http_requests_total_final[1h]

# Data from last day
http_requests_total_final[1d]
```

### Aggregation Functions

```promql
# Sum all requests
sum(http_requests_total_final)

# Average response time
avg(http_request_duration_seconds_final)

# Maximum value
max(active_connections_final)

# Minimum value
min(active_connections_final)

# Count of metrics
count(up)
```

### Filtering and Grouping

```promql
# Filter by label
http_requests_total_final{method="GET"}

# Group by endpoint
sum by (endpoint) (http_requests_total_final)

# Group by status code
sum by (status_code) (http_requests_total_final)
```

---

## 📈 Creating Dashboards in Grafana

Your Grafana instance is connected to Prometheus. Access it at:
```
http://localhost:3000
Username: admin
Password: SecureGrafanaP@ss2024
```

### Steps to Create a Dashboard:

1. **Login to Grafana**
   - Navigate to http://localhost:3000
   - Login with credentials above

2. **Create New Dashboard**
   - Click "+" icon → "Dashboard"
   - Click "Add new panel"

3. **Add Prometheus Query**
   - In the query editor, select "Prometheus" as data source
   - Enter your PromQL query
   - Example: `rate(http_requests_total_final[5m])`

4. **Customize Visualization**
   - Choose graph type (Time series, Gauge, Stat, etc.)
   - Set title and description
   - Configure axes and legends

5. **Save Dashboard**
   - Click "Save" icon
   - Give it a name
   - Click "Save"

---

## 🎯 Useful Prometheus Endpoints

### Health Check
```
http://localhost:9090/-/healthy
```

### Ready Check
```
http://localhost:9090/-/ready
```

### Configuration
```
http://localhost:9090/api/v1/status/config
```

### All Targets
```
http://localhost:9090/api/v1/targets
```

### All Metrics
```
http://localhost:9090/api/v1/label/__name__/values
```

### Query API
```
http://localhost:9090/api/v1/query?query=up
```

### Query Range API
```
http://localhost:9090/api/v1/query_range?query=up&start=2024-01-01T00:00:00Z&end=2024-01-01T01:00:00Z&step=15s
```

---

## 🔍 Troubleshooting

### Prometheus Not Accessible

```powershell
# Check if Prometheus container is running
docker ps | Select-String "prometheus"

# Check Prometheus logs
docker logs jpmorgan-prometheus-prod

# Restart Prometheus
docker-compose -f docker-compose.production.yml restart prometheus
```

### No Metrics Showing

```powershell
# Check if API is exposing metrics
Invoke-WebRequest http://localhost:8000/metrics

# Verify Prometheus targets
Invoke-RestMethod http://localhost:9090/api/v1/targets
```

### Slow Queries

- Reduce time range
- Use rate() instead of raw counters
- Add more specific label filters
- Increase Prometheus resources if needed

---

## 📚 Example Queries for Your API

### Monitor API Health

```promql
# API is responding
up{job="jpmorgan-api"}

# Request rate (requests per second)
rate(http_requests_total_final[1m])

# Average response time in milliseconds
rate(http_request_duration_seconds_final_sum[5m]) / rate(http_request_duration_seconds_final_count[5m]) * 1000

# Error rate percentage
(sum(rate(errors_total_final[5m])) / sum(rate(http_requests_total_final[5m]))) * 100
```

### Monitor Database

```promql
# Database connections (if exposed)
database_connections_active

# Query duration
database_query_duration_seconds
```

### Monitor Business Metrics

```promql
# Telemetry events processed
rate(telemetry_events_processed_total_final[5m])

# Anomalies detected
rate(anomaly_detections_total_final[5m])

# Batch processing size
telemetry_batch_size_final
```

---

## 🎨 PowerShell Script to Query Prometheus

```powershell
# Function to query Prometheus
function Get-PrometheusMetric {
    param(
        [string]$Query,
        [string]$PrometheusUrl = "http://localhost:9090"
    )
    
    $encodedQuery = [System.Web.HttpUtility]::UrlEncode($Query)
    $url = "$PrometheusUrl/api/v1/query?query=$encodedQuery"
    
    try {
        $response = Invoke-RestMethod -Uri $url -Method Get
        if ($response.status -eq "success") {
            return $response.data.result
        } else {
            Write-Error "Query failed: $($response.error)"
        }
    } catch {
        Write-Error "Failed to query Prometheus: $_"
    }
}

# Example usage
Get-PrometheusMetric -Query "up"
Get-PrometheusMetric -Query "http_requests_total_final"
Get-PrometheusMetric -Query "rate(http_requests_total_final[5m])"
```

---

## 📊 Monitoring Best Practices

### 1. Use Rate for Counters
```promql
# Good - shows rate of change
rate(http_requests_total_final[5m])

# Bad - raw counter value (always increasing)
http_requests_total_final
```

### 2. Set Appropriate Time Windows
- **Real-time monitoring:** [1m] to [5m]
- **Recent trends:** [15m] to [1h]
- **Historical analysis:** [1h] to [1d]

### 3. Use Labels for Filtering
```promql
# Filter by specific endpoint
http_requests_total_final{endpoint="/health"}

# Filter by status code
http_requests_total_final{status_code="200"}
```

### 4. Create Alerts
Define alerts in `alerts.yml` for:
- High error rates
- Slow response times
- Service downtime
- Resource exhaustion

---

## 🚀 Quick Commands

```powershell
# Open Prometheus in browser
Start-Process "http://localhost:9090"

# Open Grafana in browser
Start-Process "http://localhost:3000"

# Check Prometheus health
Invoke-WebRequest http://localhost:9090/-/healthy

# Get all metrics
Invoke-RestMethod "http://localhost:9090/api/v1/label/__name__/values"

# Query specific metric
Invoke-RestMethod "http://localhost:9090/api/v1/query?query=up"

# Check targets
Invoke-RestMethod "http://localhost:9090/api/v1/targets"
```

---

## 📖 Additional Resources

### Official Documentation
- **Prometheus:** https://prometheus.io/docs/
- **PromQL:** https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Grafana:** https://grafana.com/docs/

### Your Configuration Files
- **Prometheus Config:** `prometheus.yml`
- **Alert Rules:** `alerts.yml`
- **Grafana Dashboard:** `grafana_dashboard.json`

---

## ✅ Summary

Your Prometheus setup is:
- ✅ **Running** at http://localhost:9090
- ✅ **Collecting metrics** from your API
- ✅ **Connected to Grafana** for visualization
- ✅ **Monitoring** all services
- ✅ **Ready to use** for queries and alerts

**Start exploring:** http://localhost:9090

---

**Need Help?** Check the Prometheus logs:
```powershell
docker logs jpmorgan-prometheus-prod
