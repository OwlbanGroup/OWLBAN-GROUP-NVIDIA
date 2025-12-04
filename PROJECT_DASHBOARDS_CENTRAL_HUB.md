# 🎛️ JPMorgan Financial APIs - Central Dashboard Hub

**Last Updated:** December 2025  
**Status:** ✅ All Systems Operational

---

## 🚀 QUICK ACCESS - ALL DASHBOARDS

### Production Monitoring Dashboards

| Dashboard | URL | Credentials | Status |
|-----------|-----|-------------|--------|
| **Grafana Main** | http://localhost:3000 | admin / SecureGrafanaP@ss2024 | ✅ Active |
| **Prometheus** | http://localhost:9090 | No auth required | ✅ Active |
| **AlertManager** | http://localhost:9093 | No auth required | ✅ Active |
| **Node Exporter** | http://localhost:9100/metrics | No auth required | ✅ Active |

### Application Dashboards

| Dashboard | URL | Description | Status |
|-----------|-----|-------------|--------|
| **API Health** | http://localhost:8000/health | System health check | ✅ Active |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation | ✅ Active |
| **Swagger UI** | http://localhost:8000/api/docs | OpenAPI specification | ✅ Active |
| **API Metrics** | http://localhost:8000/metrics | Prometheus metrics endpoint | ✅ Active |

### Database Dashboards

| Service | Connection | Port | Status |
|---------|------------|------|--------|
| **PostgreSQL** | localhost:5432 | 5432 | ✅ Connected |
| **Redis** | localhost:6379 | 6379 | ✅ Connected |

---

## 📊 GRAFANA DASHBOARDS

### Available Dashboards in Grafana

#### 1. **JPMorgan Financial APIs - Main Dashboard**
- **URL:** http://localhost:3000/d/jpmorgan-main
- **Description:** Overview of all services, API performance, and system health
- **Key Metrics:**
  - API response times
  - Request rates
  - Error rates
  - Service uptime
  - Database connections
  - Cache hit rates

#### 2. **System Performance Dashboard**
- **URL:** http://localhost:3000/d/system-performance
- **Description:** CPU, memory, disk, and network metrics
- **Key Metrics:**
  - CPU utilization
  - Memory usage
  - Disk I/O
  - Network traffic
  - Container resource usage

#### 3. **Database Performance Dashboard**
- **URL:** http://localhost:3000/d/database-performance
- **Description:** PostgreSQL and Redis performance metrics
- **Key Metrics:**
  - Query execution times
  - Connection pool status
  - Cache hit/miss ratios
  - Database size and growth
  - Slow queries

#### 4. **API Endpoint Analytics**
- **URL:** http://localhost:3000/d/api-analytics
- **Description:** Detailed analytics for each API endpoint
- **Key Metrics:**
  - Endpoint-specific response times
  - Request volume per endpoint
  - Error rates by endpoint
  - Top slowest endpoints
  - Most used endpoints

#### 5. **Alert Dashboard**
- **URL:** http://localhost:3000/d/alerts
- **Description:** Active alerts and alert history
- **Key Metrics:**
  - Active alerts
  - Alert history
  - Alert trends
  - Alert resolution times

---

## 🔍 PROMETHEUS DASHBOARDS

### Prometheus Query Interface
- **URL:** http://localhost:9090/graph
- **Description:** Custom metric queries and visualization

### Key Prometheus Queries

#### API Performance Queries
```promql
# Average API response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Request rate per second
rate(http_requests_total[1m])

# Error rate percentage
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100

# 95th percentile response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

#### System Resource Queries
```promql
# CPU usage percentage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage percentage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk usage percentage
(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100
```

#### Database Queries
```promql
# PostgreSQL connections
pg_stat_database_numbackends

# Redis memory usage
redis_memory_used_bytes

# Database query rate
rate(pg_stat_database_xact_commit[5m])
```

---

## 🎯 DASHBOARD ACCESS GUIDE

### First-Time Setup

#### 1. Access Grafana
```powershell
# Open Grafana in browser
Start-Process "http://localhost:3000"

# Login credentials
Username: admin
Password: SecureGrafanaP@ss2024
```

#### 2. Import Dashboards
```powershell
# Navigate to project directory
cd c:\Users\bizle\Desktop\jpmorgan_financial_apis

# Import Prometheus dashboard
powershell -ExecutionPolicy Bypass -File import_prometheus_dashboard.ps1

# Or manually import
# 1. Go to Grafana → Dashboards → Import
# 2. Upload prometheus_dashboard.json
# 3. Select Prometheus data source
# 4. Click Import
```

#### 3. Configure Data Sources
```
Grafana → Configuration → Data Sources

Prometheus:
- URL: http://prometheus:9090
- Access: Server (default)
- Scrape interval: 15s

PostgreSQL:
- Host: postgresql:5432
- Database: jpmorgan_financial_apis_prod
- User: jpmorgan_prod
- SSL Mode: disable
```

---

## 📱 MOBILE-FRIENDLY DASHBOARDS

### Grafana Mobile App
- **iOS:** https://apps.apple.com/app/grafana/id1463275047
- **Android:** https://play.google.com/store/apps/details?id=com.grafana.mobile

### Mobile Access Setup
1. Install Grafana mobile app
2. Add server: http://YOUR_IP:3000
3. Login with credentials
4. Access all dashboards on mobile

---

## 🔔 ALERT CONFIGURATION

### AlertManager Dashboard
- **URL:** http://localhost:9093
- **Configuration File:** `alertmanager.yml`

### Active Alert Rules

#### Critical Alerts
- **High Response Time:** API response > 500ms for 5 minutes
- **Service Down:** Any service unavailable for 1 minute
- **High Error Rate:** Error rate > 1% for 5 minutes
- **Database Connection Failed:** Cannot connect to database
- **High Memory Usage:** Memory usage > 90% for 10 minutes
- **Disk Space Low:** Disk usage > 85%

#### Warning Alerts
- **Elevated Response Time:** API response > 200ms for 10 minutes
- **Increased Error Rate:** Error rate > 0.5% for 10 minutes
- **High CPU Usage:** CPU usage > 80% for 15 minutes
- **Cache Miss Rate High:** Redis cache miss rate > 20%

### Alert Notification Channels
```yaml
# Configure in alertmanager.yml
receivers:
  - name: 'email-alerts'
    email_configs:
      - to: 'your-email@example.com'
        
  - name: 'slack-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        
  - name: 'pagerduty-alerts'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

---

## 📈 CUSTOM DASHBOARD CREATION

### Creating Custom Grafana Dashboards

#### Step 1: Create New Dashboard
```
1. Go to Grafana → Dashboards → New Dashboard
2. Click "Add new panel"
3. Select visualization type
4. Configure data source (Prometheus)
5. Write PromQL query
6. Customize appearance
7. Save dashboard
```

#### Step 2: Useful Panel Types
- **Graph:** Time-series data visualization
- **Stat:** Single value display
- **Gauge:** Progress/percentage display
- **Table:** Tabular data display
- **Heatmap:** Density visualization
- **Logs:** Log stream display

#### Step 3: Example Custom Panels

**API Response Time Panel:**
```json
{
  "title": "API Response Time",
  "targets": [
    {
      "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
      "legendFormat": "{{method}} {{endpoint}}"
    }
  ],
  "type": "graph"
}
```

**Request Rate Panel:**
```json
{
  "title": "Requests per Second",
  "targets": [
    {
      "expr": "rate(http_requests_total[1m])",
      "legendFormat": "{{status}}"
    }
  ],
  "type": "graph"
}
```

---

## 🎨 DASHBOARD CUSTOMIZATION

### Theme Options
- **Dark Theme:** Default, better for monitoring
- **Light Theme:** Better for presentations
- **Custom Theme:** Configure in Grafana settings

### Dashboard Variables
Create dynamic dashboards with variables:

```
Variable Name: service
Type: Query
Query: label_values(up, job)
```

Use in queries: `up{job="$service"}`

### Time Range Presets
- Last 5 minutes
- Last 15 minutes
- Last 30 minutes
- Last 1 hour
- Last 3 hours
- Last 6 hours
- Last 12 hours
- Last 24 hours
- Last 7 days
- Last 30 days
- Custom range

---

## 🔧 DASHBOARD MAINTENANCE

### Regular Maintenance Tasks

#### Daily
- [ ] Check for active alerts
- [ ] Review error rates
- [ ] Monitor response times
- [ ] Check service health

#### Weekly
- [ ] Review dashboard performance
- [ ] Update alert thresholds if needed
- [ ] Check disk space trends
- [ ] Review slow queries

#### Monthly
- [ ] Archive old metrics (if needed)
- [ ] Update dashboard layouts
- [ ] Review and optimize queries
- [ ] Update documentation

### Dashboard Backup
```powershell
# Export all dashboards
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/search?type=dash-db | \
  jq -r '.[] | .uid' | \
  xargs -I {} curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/dashboards/uid/{} > dashboard_{}.json

# Or use Grafana CLI
docker exec jpmorgan-grafana-prod grafana-cli admin export-dashboards
```

---

## 📊 DASHBOARD BEST PRACTICES

### Design Principles
1. **Keep it Simple:** Don't overcrowd dashboards
2. **Use Consistent Colors:** Red for errors, green for success
3. **Group Related Metrics:** Organize panels logically
4. **Add Context:** Include descriptions and units
5. **Set Appropriate Time Ranges:** Match to use case

### Performance Tips
1. **Limit Query Complexity:** Use efficient PromQL queries
2. **Set Reasonable Refresh Rates:** 5-15 seconds for most dashboards
3. **Use Dashboard Variables:** Reduce number of dashboards
4. **Archive Old Data:** Keep retention policies reasonable
5. **Optimize Panel Count:** 10-15 panels per dashboard max

### Naming Conventions
- **Dashboards:** `[Service] - [Purpose]` (e.g., "API - Performance")
- **Panels:** Clear, descriptive names (e.g., "API Response Time (p95)")
- **Variables:** Lowercase with underscores (e.g., `service_name`)

---

## 🚨 TROUBLESHOOTING DASHBOARDS

### Common Issues and Solutions

#### Dashboard Not Loading
```powershell
# Check Grafana status
docker logs jpmorgan-grafana-prod

# Restart Grafana
docker-compose -f docker-compose.production.yml restart grafana

# Check data source connection
curl http://localhost:3000/api/datasources
```

#### No Data Showing
```powershell
# Check Prometheus is scraping
curl http://localhost:9090/api/v1/targets

# Verify metrics are being collected
curl http://localhost:8000/metrics

# Check time range settings in dashboard
```

#### Slow Dashboard Performance
```
1. Reduce query complexity
2. Increase refresh interval
3. Limit time range
4. Reduce number of panels
5. Use recording rules for complex queries
```

#### Alerts Not Firing
```powershell
# Check AlertManager status
curl http://localhost:9093/api/v1/status

# View alert rules
curl http://localhost:9090/api/v1/rules

# Check alert configuration
cat alertmanager.yml
```

---

## 📱 DASHBOARD SHORTCUTS

### Grafana Keyboard Shortcuts
- **`d` + `k`:** Open dashboard search
- **`d` + `h`:** Go to home dashboard
- **`d` + `s`:** Save dashboard
- **`d` + `e`:** Expand/collapse row
- **`t` + `z`:** Zoom out time range
- **`t` + `←`:** Move time range back
- **`t` + `→`:** Move time range forward
- **`Ctrl/Cmd` + `S`:** Save dashboard
- **`Esc`:** Exit fullscreen/edit mode

### Quick Access URLs
```powershell
# Open all dashboards at once
Start-Process "http://localhost:3000"      # Grafana
Start-Process "http://localhost:9090"      # Prometheus
Start-Process "http://localhost:8000/docs" # API Docs
Start-Process "http://localhost:9093"      # AlertManager
```

---

## 🎯 DASHBOARD TEMPLATES

### Pre-configured Dashboard Templates

#### 1. Executive Summary Dashboard
**Purpose:** High-level overview for management
**Panels:**
- Total requests today
- Average response time
- Error rate
- Service uptime
- Active users
- Revenue metrics

#### 2. DevOps Dashboard
**Purpose:** Operational monitoring for engineers
**Panels:**
- Service health status
- Container resource usage
- Database performance
- Cache hit rates
- Error logs
- Deployment history

#### 3. Performance Dashboard
**Purpose:** Detailed performance analysis
**Panels:**
- Response time percentiles (p50, p95, p99)
- Request rate by endpoint
- Database query times
- Cache performance
- Network latency
- Resource utilization

#### 4. Business Metrics Dashboard
**Purpose:** Business KPIs and analytics
**Panels:**
- Transaction volume
- Revenue trends
- User activity
- Feature usage
- Conversion rates
- Customer satisfaction

---

## 📚 ADDITIONAL RESOURCES

### Documentation
- **Grafana Docs:** https://grafana.com/docs/
- **Prometheus Docs:** https://prometheus.io/docs/
- **PromQL Guide:** https://prometheus.io/docs/prometheus/latest/querying/basics/

### Local Documentation Files
- `GRAFANA_DASHBOARD_SETUP_GUIDE.md` - Setup instructions
- `GRAFANA_QUICK_SETUP.md` - Quick start guide
- `PROMETHEUS_SETUP_GUIDE.md` - Prometheus configuration
- `PROMETHEUS_DASHBOARD_MANUAL_SETUP.md` - Manual setup steps

### Configuration Files
- `prometheus.yml` - Prometheus configuration
- `alertmanager.yml` - Alert configuration
- `alerts.yml` - Alert rules
- `prometheus_dashboard.json` - Dashboard export
- `grafana_dashboard.json` - Grafana dashboard

### Scripts
- `setup_grafana_dashboard.ps1` - Automated setup
- `import_prometheus_dashboard.ps1` - Import dashboards
- `PROMETHEUS_DEMO.ps1` - Demo script
- `FINAL_PRODUCTION_VERIFICATION.ps1` - Health check

---

## 🎉 QUICK START GUIDE

### For First-Time Users

#### Step 1: Access Grafana (30 seconds)
```powershell
Start-Process "http://localhost:3000"
# Login: admin / SecureGrafanaP@ss2024
```

#### Step 2: View Main Dashboard (1 minute)
```
1. Click "Dashboards" in left menu
2. Select "JPMorgan Financial APIs - Main Dashboard"
3. Explore the metrics and graphs
```

#### Step 3: Customize Time Range (30 seconds)
```
1. Click time picker in top-right
2. Select desired time range
3. Click "Apply"
```

#### Step 4: Create Alert (2 minutes)
```
1. Go to Alerting → Alert rules
2. Click "New alert rule"
3. Configure conditions
4. Set notification channel
5. Save alert
```

#### Step 5: Export Dashboard (1 minute)
```
1. Open dashboard
2. Click share icon
3. Select "Export"
4. Save JSON file
```

---

## 📞 SUPPORT & HELP

### Getting Help
- **Documentation:** Review files in project root
- **Logs:** `docker-compose logs -f grafana`
- **Status Check:** Run `FINAL_PRODUCTION_VERIFICATION.ps1`
- **Community:** Grafana Community Forums

### Common Questions

**Q: How do I reset Grafana password?**
```powershell
docker exec jpmorgan-grafana-prod grafana-cli admin reset-admin-password newpassword
```

**Q: How do I add a new data source?**
```
Grafana → Configuration → Data Sources → Add data source
```

**Q: How do I share a dashboard?**
```
Dashboard → Share → Get link or export JSON
```

**Q: How do I set up email alerts?**
```
Edit alertmanager.yml and add email configuration
```

---

## 🎊 DASHBOARD SUMMARY

### Available Dashboards (8 Total)

| # | Dashboard | Purpose | URL |
|---|-----------|---------|-----|
| 1 | Grafana Main | Overall monitoring | http://localhost:3000 |
| 2 | Prometheus | Metrics & queries | http://localhost:9090 |
| 3 | API Docs | API documentation | http://localhost:8000/docs |
| 4 | AlertManager | Alert management | http://localhost:9093 |
| 5 | Node Exporter | System metrics | http://localhost:9100 |
| 6 | API Health | Health checks | http://localhost:8000/health |
| 7 | Swagger UI | API testing | http://localhost:8000/api/docs |
| 8 | Metrics Endpoint | Raw metrics | http://localhost:8000/metrics |

### Key Features
✅ Real-time monitoring  
✅ Historical data analysis  
✅ Custom alerting  
✅ Multiple visualization types  
✅ Mobile access  
✅ Export/import capabilities  
✅ Role-based access control  
✅ API integration  

---

## 🚀 NEXT STEPS

### Recommended Actions
1. ✅ Access Grafana and explore dashboards
2. ⏳ Customize dashboards for your needs
3. ⏳ Set up alert notifications
4. ⏳ Create custom dashboards
5. ⏳ Configure mobile access
6. ⏳ Set up automated reports
7. ⏳ Train team on dashboard usage

### Advanced Features to Explore
- Dashboard playlists
- Annotations
- Dashboard snapshots
- Public dashboards
- Embedded dashboards
- Dashboard API
- Custom plugins

---

**Last Updated:** December 2025  
**Status:** ✅ All Dashboards Operational  
**Access:** http://localhost:3000  

🎛️ **Your Central Dashboard Hub is Ready!** 🎛️
<read_file>
<path>../../GRAFANA_DASHBOARD_SETUP_GUIDE.md</path>
</read_file>
