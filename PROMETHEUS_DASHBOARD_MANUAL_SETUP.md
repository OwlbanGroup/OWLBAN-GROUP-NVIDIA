# 🎯 Prometheus Dashboard Manual Setup Guide

**Complete step-by-step guide to create a Prometheus monitoring dashboard in Grafana**

---

## 📋 QUICK SETUP (5 Minutes)

### Step 1: Access Grafana (30 seconds)
```
URL: http://localhost:3000
Username: admin
Password: SecureGrafanaP@ss2024
```

### Step 2: Add Prometheus Data Source (2 minutes)

1. **Click the gear icon (⚙️)** on the left sidebar
2. **Click "Data sources"**
3. **Click "Add data source"**
4. **Select "Prometheus"**
5. **Configure:**
   ```
   Name: Prometheus
   URL: http://prometheus:9090
   Access: Server (default)
   ```
6. **Click "Save & test"**
7. **Verify:** You should see ✅ "Data source is working"

### Step 3: Import Prometheus Dashboard (2 minutes)

**Option A: Import from JSON File**
1. **Click "+" icon** → "Import"
2. **Click "Upload JSON file"**
3. **Select:** `prometheus_dashboard.json`
4. **Select data source:** Prometheus
5. **Click "Import"**
6. **Done!** ✅

**Option B: Manual Import**
1. **Click "+" icon** → "Import"
2. **Paste the dashboard JSON** (from prometheus_dashboard.json)
3. **Click "Load"**
4. **Select data source:** Prometheus
5. **Click "Import"**

---

## 📊 PROMETHEUS DASHBOARD FEATURES

### 16 Monitoring Panels Included:

#### **Status & Health (4 Panels)**
1. **Prometheus Status** - UP/DOWN indicator
2. **All Services Status** - Status of all monitored services
3. **Prometheus Targets** - Count of up/down targets
4. **Scrape Duration** - Time taken to scrape metrics

#### **Performance Metrics (4 Panels)**
5. **Scrape Samples Rate** - Samples collected per second
6. **Query Rate** - Prometheus query requests per second
7. **Target Scrape Health** - Health timeline of all targets
8. **HTTP Request Duration** - Request latency percentiles (99th, 95th, 50th)

#### **Resource Monitoring (4 Panels)**
9. **Memory Usage** - Resident and virtual memory
10. **CPU Usage** - Prometheus CPU consumption
11. **TSDB Head Series** - Number of time series in memory
12. **TSDB Chunks** - Number of chunks in memory

#### **Detailed Metrics (4 Panels)**
13. **Scrape Duration by Job** - Scrape time per job
14. **Scrape Samples by Job** - Samples collected per job
15. **Rule Evaluation Duration** - Time to evaluate rules
16. **Storage Size** - TSDB storage size

---

## 🎨 DASHBOARD LAYOUT

```
┌─────────────────────────────────────────────────────────────┐
│  Prometheus Monitoring Dashboard - JPMorgan APIs            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Row 1: Status Indicators                                   │
│  ┌──────────┬──────────────────┬──────────┬──────────┐    │
│  │Prometheus│ All Services     │ Targets  │ Scrape   │    │
│  │ Status   │ Status           │ Count    │ Duration │    │
│  └──────────┴──────────────────┴──────────┴──────────┘    │
│                                                              │
│  Row 2: Performance Graphs                                  │
│  ┌──────────────────────┬──────────────────────┐          │
│  │ Scrape Samples       │ Query Rate           │          │
│  │ (Graph)              │ (Graph)              │          │
│  └──────────────────────┴──────────────────────┘          │
│                                                              │
│  Row 3: Target Health                                       │
│  ┌─────────────────────────────────────────────┐          │
│  │ Target Scrape Health (Full Width Graph)     │          │
│  └─────────────────────────────────────────────┘          │
│                                                              │
│  Row 4: Resource Usage                                      │
│  ┌──────────────────────┬──────────────────────┐          │
│  │ Memory Usage         │ CPU Usage            │          │
│  └──────────────────────┴──────────────────────┘          │
│                                                              │
│  Row 5: TSDB Metrics                                        │
│  ┌──────────────────────┬──────────────────────┐          │
│  │ TSDB Head Series     │ TSDB Chunks          │          │
│  └──────────────────────┴──────────────────────┘          │
│                                                              │
│  Row 6: Job Metrics                                         │
│  ┌──────────────────────┬──────────────────────┐          │
│  │ Scrape Duration      │ Scrape Samples       │          │
│  │ by Job               │ by Job               │          │
│  └──────────────────────┴──────────────────────┘          │
│                                                              │
│  Row 7: Advanced Metrics                                    │
│  ┌──────────────────────┬──────────────────────┐          │
│  │ Rule Evaluation      │ Storage Size         │          │
│  └──────────────────────┴──────────────────────┘          │
│                                                              │
│  Row 8: Request Latency                                     │
│  ┌─────────────────────────────────────────────┐          │
│  │ HTTP Request Duration (Full Width)           │          │
│  │ 99th, 95th, 50th Percentiles                │          │
│  └─────────────────────────────────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 KEY PROMETHEUS QUERIES USED

### 1. Service Status
```promql
up
```
Shows if services are up (1) or down (0)

### 2. Scrape Samples Rate
```promql
rate(prometheus_tsdb_head_samples_appended_total[5m])
```
Samples being added to TSDB per second

### 3. Query Rate
```promql
rate(prometheus_http_requests_total[5m])
```
HTTP requests to Prometheus per second

### 4. Memory Usage
```promql
process_resident_memory_bytes{job="prometheus"}
process_virtual_memory_bytes{job="prometheus"}
```
Prometheus memory consumption

### 5. CPU Usage
```promql
rate(process_cpu_seconds_total{job="prometheus"}[5m]) * 100
```
Prometheus CPU usage percentage

### 6. TSDB Series
```promql
prometheus_tsdb_head_series
```
Number of time series in memory

### 7. Scrape Duration
```promql
scrape_duration_seconds
```
Time taken to scrape each target

### 8. HTTP Request Latency
```promql
histogram_quantile(0.99, rate(prometheus_http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(prometheus_http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.50, rate(prometheus_http_request_duration_seconds_bucket[5m]))
```
Request duration percentiles

---

## 🚀 ALTERNATIVE: CREATE DASHBOARD FROM SCRATCH

If you prefer to build the dashboard manually:

### Panel 1: Prometheus Status

1. **Create new dashboard** → **Add panel**
2. **Query:**
   ```promql
   up{job="prometheus"}
   ```
3. **Visualization:** Stat
4. **Title:** Prometheus Status
5. **Value mappings:**
   - 1 = "UP" (Green)
   - 0 = "DOWN" (Red)
6. **Apply**

### Panel 2: All Services Status

1. **Add panel**
2. **Query:**
   ```promql
   up
   ```
3. **Visualization:** Stat
4. **Title:** All Services Status
5. **Legend:** {{job}}
6. **Apply**

### Panel 3: Scrape Samples

1. **Add panel**
2. **Query:**
   ```promql
   rate(prometheus_tsdb_head_samples_appended_total[5m])
   ```
3. **Visualization:** Graph
4. **Title:** Prometheus Scrape Samples
5. **Y-axis:** Samples/sec
6. **Apply**

### Panel 4: Query Rate

1. **Add panel**
2. **Query:**
   ```promql
   rate(prometheus_http_requests_total[5m])
   ```
3. **Visualization:** Graph
4. **Title:** Prometheus Query Rate
5. **Legend:** {{handler}}
6. **Apply**

### Panel 5: Memory Usage

1. **Add panel**
2. **Queries:**
   ```promql
   process_resident_memory_bytes{job="prometheus"}
   process_virtual_memory_bytes{job="prometheus"}
   ```
3. **Visualization:** Graph
4. **Title:** Prometheus Memory Usage
5. **Y-axis:** bytes
6. **Apply**

### Panel 6: CPU Usage

1. **Add panel**
2. **Query:**
   ```promql
   rate(process_cpu_seconds_total{job="prometheus"}[5m]) * 100
   ```
3. **Visualization:** Graph
4. **Title:** Prometheus CPU Usage
5. **Y-axis:** percent (0-100)
6. **Apply**

**Continue adding panels following the same pattern...**

---

## 🎯 DASHBOARD SETTINGS

### Time Range
- **Default:** Last 6 hours
- **Refresh:** 30 seconds
- **Timezone:** Browser

### Variables (Optional)
Create a variable for job selection:
1. **Dashboard settings** → **Variables**
2. **Add variable:**
   ```
   Name: job
   Type: Query
   Query: label_values(up, job)
   ```
3. **Use in queries:** `up{job="$job"}`

### Annotations (Optional)
Add deployment markers:
1. **Dashboard settings** → **Annotations**
2. **Add annotation query**
3. **Query:** Custom events or deployment times

---

## 🔧 TROUBLESHOOTING

### Issue: "No data" in panels

**Solutions:**
1. **Check Prometheus is running:**
   ```powershell
   docker ps | Select-String "prometheus"
   ```

2. **Verify Prometheus is scraping:**
   ```
   Open: http://localhost:9090/targets
   Check all targets are "UP"
   ```

3. **Test query in Prometheus:**
   ```
   Open: http://localhost:9090
   Run query: up
   ```

4. **Check time range** in Grafana (top right)

5. **Verify data source connection:**
   ```
   Grafana → Configuration → Data sources → Prometheus → Test
   ```

### Issue: "Data source not found"

**Solution:**
1. Go to **Configuration** → **Data sources**
2. Click **Add data source**
3. Select **Prometheus**
4. URL: `http://prometheus:9090`
5. Click **Save & test**

### Issue: Dashboard not importing

**Solutions:**
1. **Check JSON file exists:**
   ```powershell
   Test-Path prometheus_dashboard.json
   ```

2. **Validate JSON:**
   - Copy content
   - Paste in JSON validator
   - Fix any syntax errors

3. **Manual import:**
   - Copy entire JSON content
   - Grafana → Import → Paste JSON
   - Load → Import

---

## 📱 MOBILE ACCESS

Access dashboard from mobile:
```
http://YOUR-COMPUTER-IP:3000
```

Find your IP:
```powershell
ipconfig | Select-String "IPv4"
```

---

## 🎨 CUSTOMIZATION TIPS

### Change Colors
1. **Edit panel** → **Field** tab
2. **Thresholds:** Set custom values and colors
3. **Color scheme:** Choose from presets

### Add Alerts
1. **Edit panel** → **Alert** tab
2. **Create alert rule**
3. **Set conditions** (e.g., CPU > 80%)
4. **Configure notifications**

### Adjust Refresh Rate
1. **Dashboard settings** (gear icon)
2. **Auto refresh:** Choose interval (10s, 30s, 1m, etc.)

### Export Dashboard
1. **Dashboard settings** → **JSON Model**
2. **Copy JSON** or **Save to file**
3. **Share with team**

---

## ✅ VERIFICATION CHECKLIST

After setup, verify:

- [ ] Grafana accessible at http://localhost:3000
- [ ] Prometheus data source added and tested
- [ ] Dashboard imported successfully
- [ ] All 16 panels showing data
- [ ] No "No data" errors
- [ ] Auto-refresh working (30s)
- [ ] Time range set correctly
- [ ] All services showing as "UP"
- [ ] Metrics updating in real-time

---

## 📚 ADDITIONAL RESOURCES

### Documentation
- **Grafana Docs:** https://grafana.com/docs/
- **Prometheus Docs:** https://prometheus.io/docs/
- **PromQL Guide:** https://prometheus.io/docs/prometheus/latest/querying/basics/

### Related Files
- **prometheus_dashboard.json** - Dashboard definition
- **import_prometheus_dashboard.ps1** - Auto-import script
- **PROMETHEUS_SETUP_GUIDE.md** - Prometheus guide
- **GRAFANA_DASHBOARD_SETUP_GUIDE.md** - Complete Grafana guide

### Quick Commands
```powershell
# Open Grafana
Start-Process "http://localhost:3000"

# Open Prometheus
Start-Process "http://localhost:9090"

# Check services
docker-compose -f docker-compose.production.yml ps

# Restart Grafana
docker-compose -f docker-compose.production.yml restart grafana
```

---

## 🎉 SUCCESS!

Once you see data in all panels, your Prometheus monitoring dashboard is complete!

**You can now monitor:**
- ✅ Prometheus health and status
- ✅ All service availability
- ✅ Scrape performance
- ✅ Query rates
- ✅ Resource usage (CPU, Memory)
- ✅ TSDB metrics
- ✅ Storage size
- ✅ Request latency

**Happy Monitoring!** 🚀
