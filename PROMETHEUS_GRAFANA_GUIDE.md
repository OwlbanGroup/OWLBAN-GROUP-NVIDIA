# JPMorgan API - Prometheus & Grafana Integration Guide

## 🎯 Overview

This guide covers the complete Prometheus metrics integration for JPMorgan API monitoring with Grafana dashboards.

---

## 📊 Available Metrics

### **Account Metrics**

```promql
# Current balance for each account
jpm_account_balance{accountId="123", accountName="Operating", accountType="CHECKING", currency="USD"}

# Total balance across all accounts
sum(jpm_account_balance)

# Balance by account type
sum(jpm_account_balance) by (accountType)

# Balance by currency
sum(jpm_account_balance) by (currency)
```

### **API Performance Metrics**

```promql
# API call rate (requests per second)
rate(jpm_api_calls_total[5m])

# API call rate by endpoint
rate(jpm_api_calls_total{endpoint="balances"}[5m])

# API success rate
rate(jpm_api_calls_total{status="success"}[5m]) / rate(jpm_api_calls_total[5m])

# API error rate
rate(jpm_api_errors_total[5m])

# API response time (p50, p95, p99)
histogram_quantile(0.50, rate(jpm_api_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(jpm_api_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(jpm_api_duration_seconds_bucket[5m]))
```

### **OAuth2 Token Metrics**

```promql
# Token expiry time (unix timestamp)
jpm_token_expiry_timestamp

# Seconds until token expires
jpm_token_expiry_timestamp - time()

# Token refresh count
sum(jpm_token_refresh_total{status="success"})
sum(jpm_token_refresh_total{status="failure"})

# Token acquisition duration
histogram_quantile(0.95, rate(jpm_token_acquisition_duration_seconds_bucket[5m]))
```

### **Health Metrics**

```promql
# Last successful API call (unix timestamp)
jpm_api_last_success_timestamp{endpoint="balances"}

# Seconds since last successful call
time() - jpm_api_last_success_timestamp{endpoint="balances"}
```

---

## 🚀 Setup Instructions

### **Step 1: Start the NestJS Backend**

```bash
cd nestjs-backend

# Install dependencies (if not already done)
npm install

# Start the server
npm run start:dev
```

The metrics endpoint will be available at:
```
http://localhost:4000/metrics
```

### **Step 2: Configure Prometheus**

Create or update `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'jpmorgan-api'
    static_configs:
      - targets: ['localhost:4000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

Start Prometheus:

```bash
# Using Docker
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Or download and run locally
./prometheus --config.file=prometheus.yml
```

Verify Prometheus is scraping:
1. Open http://localhost:9090
2. Go to Status → Targets
3. Verify `jpmorgan-api` target is UP

### **Step 3: Configure Grafana**

#### **Option A: Using Docker**

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana-oss
```

#### **Option B: Local Installation**

Download from https://grafana.com/grafana/download

#### **Add Prometheus Data Source**

1. Open Grafana: http://localhost:3000 (default: admin/admin)
2. Go to Configuration → Data Sources
3. Click "Add data source"
4. Select "Prometheus"
5. Set URL: `http://localhost:9090`
6. Click "Save & Test"

#### **Import Dashboard**

1. Go to Dashboards → Import
2. Upload `grafana-prometheus-dashboard.json`
3. Select your Prometheus data source
4. Click "Import"

---

## 📈 Dashboard Panels

### **Panel 1: Total Balance**
- **Type:** Stat
- **Query:** `sum(jpm_account_balance)`
- **Description:** Shows total balance across all accounts

### **Panel 2: Account Balances Over Time**
- **Type:** Time Series
- **Query:** `jpm_account_balance`
- **Description:** Line chart showing balance trends for each account

### **Panel 3: Account Details Table**
- **Type:** Table
- **Query:** `jpm_account_balance`
- **Description:** Detailed table with account ID, name, type, currency, and balance

### **Panel 4: Last Successful API Call**
- **Type:** Stat
- **Query:** `time() - jpm_api_last_success_timestamp{endpoint="balances"}`
- **Description:** Shows seconds since last successful API call

### **Panel 5: OAuth2 Token Expiry**
- **Type:** Stat
- **Query:** `jpm_token_expiry_timestamp - time()`
- **Description:** Shows seconds until token expires

### **Panel 6: API Call Rate**
- **Type:** Time Series
- **Query:** `rate(jpm_api_calls_total[5m])`
- **Description:** Shows API calls per second by endpoint and status

### **Panel 7: API Response Time**
- **Type:** Time Series
- **Query:** `histogram_quantile(0.95, rate(jpm_api_duration_seconds_bucket[5m]))`
- **Description:** Shows p50 and p95 response times

### **Panel 8: API Error Rate**
- **Type:** Time Series
- **Query:** `rate(jpm_api_errors_total[5m])`
- **Description:** Shows error rate by endpoint and error type

### **Panel 9: Token Refresh Count**
- **Type:** Stat
- **Query:** `sum(jpm_token_refresh_total{status="success"})`
- **Description:** Shows successful and failed token refreshes

---

## 🔍 Example Queries

### **Find Accounts with Low Balance**

```promql
jpm_account_balance < 1000
```

### **Calculate Total Balance by Currency**

```promql
sum(jpm_account_balance) by (currency)
```

### **API Error Rate Percentage**

```promql
(
  rate(jpm_api_errors_total[5m]) / 
  rate(jpm_api_calls_total[5m])
) * 100
```

### **Average API Response Time**

```promql
rate(jpm_api_duration_seconds_sum[5m]) / 
rate(jpm_api_duration_seconds_count[5m])
```

### **Token Refresh Success Rate**

```promql
(
  sum(rate(jpm_token_refresh_total{status="success"}[5m])) /
  sum(rate(jpm_token_refresh_total[5m]))
) * 100
```

---

## 🚨 Alerting Rules

Create `alerts.yml`:

```yaml
groups:
  - name: jpmorgan_api_alerts
    interval: 30s
    rules:
      # Alert if API error rate exceeds 5%
      - alert: HighAPIErrorRate
        expr: |
          (
            rate(jpm_api_errors_total[5m]) / 
            rate(jpm_api_calls_total[5m])
          ) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API error rate detected"
          description: "API error rate is {{ $value | humanizePercentage }}"

      # Alert if no successful API calls in 5 minutes
      - alert: APICallsStalled
        expr: |
          (time() - jpm_api_last_success_timestamp) > 300
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "No successful API calls"
          description: "No successful API calls for {{ $value }} seconds"

      # Alert if token expires in less than 5 minutes
      - alert: TokenExpiringSoon
        expr: |
          (jpm_token_expiry_timestamp - time()) < 300
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "OAuth2 token expiring soon"
          description: "Token expires in {{ $value }} seconds"

      # Alert if API response time p95 exceeds 5 seconds
      - alert: SlowAPIResponse
        expr: |
          histogram_quantile(0.95, rate(jpm_api_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow API response time"
          description: "p95 response time is {{ $value }}s"

      # Alert if account balance drops below threshold
      - alert: LowAccountBalance
        expr: |
          jpm_account_balance < 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low account balance"
          description: "Account {{ $labels.accountName }} ({{ $labels.accountId }}) has balance {{ $value }}"
```

Add to `prometheus.yml`:

```yaml
rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

---

## 🔧 Advanced Configuration

### **Custom Scrape Intervals**

For high-frequency monitoring:

```yaml
scrape_configs:
  - job_name: 'jpmorgan-api-frequent'
    static_configs:
      - targets: ['localhost:4000']
    metrics_path: '/metrics'
    scrape_interval: 10s  # Scrape every 10 seconds
```

### **Metric Relabeling**

Add custom labels:

```yaml
scrape_configs:
  - job_name: 'jpmorgan-api'
    static_configs:
      - targets: ['localhost:4000']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'jpm_.*'
        target_label: 'service'
        replacement: 'jpmorgan-api'
```

### **Recording Rules**

Pre-calculate expensive queries:

```yaml
groups:
  - name: jpmorgan_recording_rules
    interval: 30s
    rules:
      - record: jpm:api_success_rate:5m
        expr: |
          rate(jpm_api_calls_total{status="success"}[5m]) / 
          rate(jpm_api_calls_total[5m])

      - record: jpm:total_balance:sum
        expr: sum(jpm_account_balance)

      - record: jpm:api_p95_latency:5m
        expr: |
          histogram_quantile(0.95, rate(jpm_api_duration_seconds_bucket[5m]))
```

---

## 📊 Grafana Dashboard Features

### **Variables**

The dashboard includes these variables:

- **$DS_PROMETHEUS**: Prometheus datasource selector
- **$account_id**: Filter by account ID (add manually if needed)
- **$endpoint**: Filter by API endpoint (add manually if needed)

### **Annotations**

Add deployment markers:

1. Dashboard Settings → Annotations
2. Add annotation query:
   ```promql
   ALERTS{alertname="DeploymentStarted"}
   ```

### **Alerts**

Configure Grafana alerts:

1. Edit any panel
2. Go to Alert tab
3. Create alert rule
4. Set notification channel

---

## 🧪 Testing the Integration

### **1. Generate Test Data**

```bash
# Make API calls to generate metrics
curl http://localhost:4000/api/jpmorgan/balances
curl http://localhost:4000/api/jpmorgan/accounts
curl http://localhost:4000/api/jpmorgan/transactions
```

### **2. Verify Metrics**

```bash
# Check metrics endpoint
curl http://localhost:4000/metrics

# Should see output like:
# jpm_account_balance{accountId="123",accountName="Operating",accountType="CHECKING",currency="USD"} 50000
# jpm_api_calls_total{endpoint="balances",status="success"} 5
# jpm_token_expiry_timestamp 1704123456
```

### **3. Query Prometheus**

```bash
# Query via API
curl 'http://localhost:9090/api/v1/query?query=jpm_account_balance'

# Or use Prometheus UI
# http://localhost:9090/graph
```

### **4. View in Grafana**

1. Open dashboard
2. Verify all panels show data
3. Check time range (last 6 hours by default)
4. Refresh dashboard

---

## 🐛 Troubleshooting

### **No Metrics Showing**

1. Check NestJS server is running:
   ```bash
   curl http://localhost:4000/metrics
   ```

2. Verify Prometheus is scraping:
   - Open http://localhost:9090/targets
   - Check target status

3. Check Prometheus logs:
   ```bash
   docker logs prometheus
   ```

### **Grafana Shows "No Data"**

1. Verify Prometheus datasource:
   - Configuration → Data Sources
   - Test connection

2. Check query syntax in panel
3. Verify time range includes data
4. Check Prometheus has data:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=jpm_account_balance'
   ```

### **Metrics Not Updating**

1. Check scrape interval in prometheus.yml
2. Verify metrics endpoint returns fresh data
3. Check for errors in NestJS logs
4. Restart Prometheus if needed

---

## 📚 Additional Resources

### **Prometheus**
- Official Docs: https://prometheus.io/docs/
- Query Examples: https://prometheus.io/docs/prometheus/latest/querying/examples/
- Best Practices: https://prometheus.io/docs/practices/naming/

### **Grafana**
- Official Docs: https://grafana.com/docs/
- Dashboard Best Practices: https://grafana.com/docs/grafana/latest/best-practices/
- Alert Rules: https://grafana.com/docs/grafana/latest/alerting/

### **prom-client (Node.js)**
- GitHub: https://github.com/siimon/prom-client
- Metrics Types: https://github.com/siimon/prom-client#metric-types

---

## 🎯 Next Steps

1. **Customize Dashboard**: Add panels for your specific needs
2. **Set Up Alerts**: Configure alerting for critical metrics
3. **Add More Metrics**: Extend metrics service with custom metrics
4. **Performance Tuning**: Adjust scrape intervals and retention
5. **High Availability**: Set up Prometheus federation for scale

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Prometheus/Grafana logs
3. Verify configuration files
4. Test metrics endpoint directly

---

**Last Updated:** January 2024  
**Version:** 1.0.0  
**Status:** Production Ready ✅
