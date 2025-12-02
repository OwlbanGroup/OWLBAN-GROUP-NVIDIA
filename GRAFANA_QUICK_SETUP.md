# 🚀 Grafana Quick Setup Guide

**5-Minute Setup for JPMorgan Financial APIs Dashboard**

---

## ✅ STEP 1: Access Grafana (1 minute)

### Open Grafana
```
URL: http://localhost:3000
Username: admin
Password: SecureGrafanaP@ss2024
```

**Action:** Open your browser and login to Grafana

---

## ✅ STEP 2: Add Prometheus Data Source (2 minutes)

### Manual Setup:

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
6. **Scroll down and click "Save & test"**
7. **You should see:** ✅ "Data source is working"

### Quick Test:
```powershell
# Verify Prometheus is accessible
Invoke-WebRequest http://localhost:9090/-/healthy
```

---

## ✅ STEP 3: Create Your First Dashboard (2 minutes)

### Option A: Import Existing Dashboard (Recommended)

1. **Click "+" icon** → "Import"
2. **Click "Upload JSON file"**
3. **Select:** `grafana_dashboard.json` from your project folder
4. **Select data source:** Prometheus
5. **Click "Import"**
6. **Done!** ✅

### Option B: Create from Scratch

1. **Click "+" icon** → "Dashboard"
2. **Click "Add new panel"**
3. **Enter query:**
   ```promql
   up
   ```
4. **Set title:** "Service Status"
5. **Click "Apply"**
6. **Click Save icon** (💾) → Give it a name → "Save"

---

## 📊 ESSENTIAL PANELS TO ADD

### Panel 1: API Health
```promql
up{job="jpmorgan-api"}
```
- **Type:** Stat
- **Title:** API Status
- **Mapping:** 1=UP (Green), 0=DOWN (Red)

### Panel 2: Request Rate
```promql
rate(http_requests_total_final[5m])
```
- **Type:** Graph
- **Title:** Request Rate
- **Unit:** requests/sec

### Panel 3: Response Time
```promql
rate(http_request_duration_seconds_final_sum[5m]) / rate(http_request_duration_seconds_final_count[5m]) * 1000
```
- **Type:** Graph
- **Title:** Response Time
- **Unit:** milliseconds (ms)

### Panel 4: Error Rate
```promql
rate(errors_total_final[5m])
```
- **Type:** Graph
- **Title:** Error Rate
- **Unit:** errors/sec

### Panel 5: Active Connections
```promql
active_connections_final
```
- **Type:** Gauge
- **Title:** Active Connections

### Panel 6: Database Status
```promql
up{job="postgres"}
```
- **Type:** Stat
- **Title:** PostgreSQL Status

---

## 🎨 DASHBOARD LAYOUT

### Recommended Structure:

```
Row 1: Status Indicators
┌──────────┬──────────┬──────────┬──────────┐
│ API      │ Database │ Redis    │ Prom     │
│ Status   │ Status   │ Status   │ Status   │
└──────────┴──────────┴──────────┴──────────┘

Row 2: Performance Metrics
┌─────────────────────────────────────────────┐
│ API Response Time (Graph - Full Width)      │
└─────────────────────────────────────────────┘

Row 3: Traffic & Errors
┌──────────────────────┬──────────────────────┐
│ Request Rate         │ Error Rate           │
└──────────────────────┴──────────────────────┘

Row 4: Resources
┌──────────────────────┬──────────────────────┐
│ Active Connections   │ Memory Usage         │
└──────────────────────┴──────────────────────┘
```

---

## 🔧 QUICK COMMANDS

### Open Grafana
```powershell
Start-Process "http://localhost:3000"
```

### Check Grafana Health
```powershell
Invoke-WebRequest http://localhost:3000/api/health
```

### Check Prometheus Connection
```powershell
Invoke-WebRequest http://localhost:9090/-/healthy
```

### Restart Grafana
```powershell
docker-compose -f docker-compose.production.yml restart grafana
```

---

## 🚨 TROUBLESHOOTING

### Issue: Cannot login to Grafana
**Solution:**
```powershell
# Check if Grafana is running
docker ps | Select-String "grafana"

# Check Grafana logs
docker logs jpmorgan-grafana-prod

# Restart Grafana
docker-compose -f docker-compose.production.yml restart grafana
```

### Issue: "Data source is not working"
**Solution:**
```powershell
# Check Prometheus is running
docker ps | Select-String "prometheus"

# Test Prometheus
Invoke-WebRequest http://localhost:9090/-/healthy

# Use correct URL in Grafana: http://prometheus:9090
```

### Issue: No data in panels
**Solutions:**
1. Check time range (top right corner)
2. Verify query in Prometheus first: http://localhost:9090
3. Check if metrics exist:
   ```promql
   {__name__=~".+"}
   ```

---

## 📱 MOBILE ACCESS

Access from any device on your network:
```
http://YOUR-COMPUTER-IP:3000
```

To find your IP:
```powershell
ipconfig | Select-String "IPv4"
```

---

## 🎯 NEXT STEPS

### After Basic Setup:

1. **Customize Panels**
   - Adjust colors and thresholds
   - Add more metrics
   - Organize layout

2. **Set Up Alerts**
   - Configure alert rules
   - Add notification channels
   - Test alerts

3. **Create More Dashboards**
   - Business metrics
   - System resources
   - Custom views

4. **Share Dashboards**
   - Export JSON
   - Create snapshots
   - Set up public access

---

## 📚 HELPFUL RESOURCES

### Quick Links
- **Grafana UI:** http://localhost:3000
- **Prometheus UI:** http://localhost:9090
- **API Docs:** http://localhost:8000/docs

### Documentation
- **Full Guide:** GRAFANA_DASHBOARD_SETUP_GUIDE.md
- **Prometheus Guide:** PROMETHEUS_SETUP_GUIDE.md
- **Deployment Guide:** PRODUCTION_DEPLOYMENT_COMPLETE_SUMMARY.md

### Tutorials
- Grafana Fundamentals: https://grafana.com/tutorials/grafana-fundamentals/
- Creating Dashboards: https://grafana.com/docs/grafana/latest/dashboards/

---

## ✅ VERIFICATION CHECKLIST

After setup, verify:

- [ ] Can login to Grafana
- [ ] Prometheus data source added
- [ ] Data source test passes
- [ ] Dashboard created or imported
- [ ] Panels showing data
- [ ] Time range set correctly
- [ ] Refresh interval configured
- [ ] Dashboard saved

---

## 🎉 SUCCESS!

Once you see data in your panels, you're done!

**Your Grafana dashboard is now monitoring your JPMorgan Financial APIs in real-time.**

### What You Can Monitor:
- ✅ API health and status
- ✅ Request rates and patterns
- ✅ Response times
- ✅ Error rates
- ✅ Database connectivity
- ✅ System resources
- ✅ Business metrics

---

**Need Help?** Check the full guide: GRAFANA_DASHBOARD_SETUP_GUIDE.md

**Questions?** Review Grafana logs:
```powershell
docker logs jpmorgan-grafana-prod
```

🚀 **Happy Monitoring!**
