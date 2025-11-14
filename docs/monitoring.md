# Monitoring and Alerting Guide - JPMorgan Financial APIs

## Overview

This guide covers the comprehensive monitoring and alerting setup for the JPMorgan Financial APIs platform, including metrics collection, visualization, alerting rules, and incident response.

## Metrics Collection

### Application Metrics

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

scrape_configs:
  - job_name: 'jpmorgan-financial-apis'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgresql:9187']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
    scrape_interval: 30s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
```

#### Application Metrics Implementation

```python
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import time
import psutil

# HTTP Request Metrics
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Business Metrics
ACTIVE_USERS = Gauge('active_users', 'Number of active users')

API_REQUESTS_PER_USER = Counter(
    'api_requests_per_user_total',
    'Total API requests per user',
    ['user_id']
)

# Database Metrics
DB_CONNECTIONS_ACTIVE = Gauge(
    'db_connections_active',
    'Number of active database connections'
)

DB_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Cache Metrics
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')

# External API Metrics
EXTERNAL_API_REQUESTS = Counter(
    'external_api_requests_total',
    'Requests to external APIs',
    ['api_name', 'status']
)

EXTERNAL_API_DURATION = Histogram(
    'external_api_duration_seconds',
    'External API call duration',
    ['api_name'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# System Metrics
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()

    def collect_system_metrics(self):
        """Collect system-level metrics"""
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        CPU_USAGE.set(psutil.cpu_percent(interval=1))

    def collect_application_metrics(self):
        """Collect application-specific metrics"""
        # Calculate uptime
        uptime = time.time() - self.start_time

        # Update business metrics
        # These would be updated based on application logic
        pass

    def get_metrics(self):
        """Get all metrics for Prometheus"""
        self.collect_system_metrics()
        self.collect_application_metrics()
        return generate_latest()

# Flask integration
def metrics_endpoint():
    collector = MetricsCollector()
    return collector.get_metrics()

# Middleware for automatic metrics collection
class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        start_time = time.time()

        def custom_start_response(status, headers, exc_info=None):
            # Record request metrics
            status_code = int(status.split()[0])
            method = environ['REQUEST_METHOD']
            endpoint = environ['PATH_INFO']

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()

            duration = time.time() - start_time
            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            return start_response(status, headers, exc_info)

        return self.app(environ, custom_start_response)
```

### Infrastructure Metrics

#### Kubernetes Metrics

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: monitoring

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: monitoring
```

#### Database Metrics

```yaml
# PostgreSQL exporter configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-exporter
  template:
    metadata:
      labels:
        app: postgres-exporter
    spec:
      containers:
      - name: postgres-exporter
        image: prometheuscommunity/postgres-exporter:latest
        ports:
        - containerPort: 9187
        env:
        - name: DATA_SOURCE_NAME
          value: "postgresql://postgres_exporter:password@postgresql:5432/postgres?sslmode=disable"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 200m
            memory: 512Mi
```

## Visualization

### Grafana Dashboards

#### System Overview Dashboard

```json
{
  "dashboard": {
    "title": "JPMorgan APIs - System Overview",
    "tags": ["jpmorgan", "apis", "system"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status_code=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "Error rate %"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_activity_count",
            "legendFormat": "Active connections"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) * 100",
            "legendFormat": "Cache hit rate %"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory GB"
          }
        ]
      }
    ]
  }
}
```

#### Business Metrics Dashboard

```json
{
  "dashboard": {
    "title": "JPMorgan APIs - Business Metrics",
    "tags": ["jpmorgan", "apis", "business"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_users",
            "legendFormat": "Active users"
          }
        ]
      },
      {
        "title": "API Usage by Endpoint",
        "type": "table",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[1h])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Top Users by Request Volume",
        "type": "table",
        "targets": [
          {
            "expr": "topk(10, sum(rate(api_requests_per_user_total[1h])) by (user_id))",
            "legendFormat": "{{user_id}}"
          }
        ]
      },
      {
        "title": "External API Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(external_api_requests_total[5m])",
            "legendFormat": "{{api_name}}"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Alert Rules

```yaml
groups:
  - name: api_performance
    rules:
      - alert: HighRequestLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
          service: api
        annotations:
          summary: "High API request latency detected"
          description: "95th percentile request latency is {{ $value }}s (threshold: 2s)"
          runbook_url: "https://docs.jpmorgan.com/runbooks/high-latency"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"[5][0-9][0-9]"}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          service: api
        annotations:
          summary: "High API error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
          runbook_url: "https://docs.jpmorgan.com/runbooks/high-error-rate"

      - alert: DatabaseConnectionPoolExhausted
        expr: db_connections_active / db_connections_max > 0.9
        for: 2m
        labels:
          severity: warning
          service: database
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Connection pool utilization is {{ $value | humanizePercentage }}"

  - name: infrastructure
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 90
        for: 5m
        labels:
          severity: warning
          service: infrastructure
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

      - alert: HighMemoryUsage
        expr: memory_usage_bytes / memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          service: infrastructure
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

      - alert: DiskSpaceLow
        expr: (disk_total_bytes - disk_free_bytes) / disk_total_bytes > 0.85
        for: 5m
        labels:
          severity: warning
          service: infrastructure
        annotations:
          summary: "Low disk space"
          description: "Disk usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

  - name: business
    rules:
      - alert: LowActiveUsers
        expr: active_users < 10
        for: 15m
        labels:
          severity: info
          service: business
        annotations:
          summary: "Low active user count"
          description: "Only {{ $value }} active users (threshold: 10)"

      - alert: HighExternalAPILatency
        expr: histogram_quantile(0.95, rate(external_api_duration_seconds_bucket{api_name="jpmorgan"}[5m])) > 10
        for: 5m
        labels:
          severity: warning
          service: external
        annotations:
          summary: "High external API latency"
          description: "JPMorgan API latency is {{ $value }}s (threshold: 10s)"

  - name: security
    rules:
      - alert: MultipleFailedAuthentications
        expr: rate(authentication_failures_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
          service: security
        annotations:
          summary: "Multiple authentication failures"
          description: "{{ $value }} authentication failures in 5 minutes"

      - alert: UnusualTrafficPattern
        expr: rate(http_requests_total[1m]) > rate(http_requests_total[1h]) * 2
        for: 5m
        labels:
          severity: info
          service: security
        annotations:
          summary: "Unusual traffic pattern detected"
          description: "Request rate increased significantly"
```

### Alert Manager Configuration

```yaml
global:
  smtp_smarthost: 'smtp.jpmorgan.com:587'
  smtp_from: 'alerts@jpmorgan.com'
  smtp_auth_username: 'alerts@jpmorgan.com'
  smtp_auth_password: 'password'

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'jpmorgan-alerts'
  routes:
  - match:
      severity: critical
    receiver: 'jpmorgan-critical'
    continue: true
  - match:
      service: security
    receiver: 'jpmorgan-security'
    continue: true

receivers:
- name: 'jpmorgan-alerts'
  email_configs:
  - to: 'devops@jpmorgan.com'
    subject: '{{ template "email.subject" . }}'
    body: '{{ template "email.body" . }}'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'
    channel: '#alerts'
    title: '{{ template "slack.title" . }}'
    text: '{{ template "slack.text" . }}'

- name: 'jpmorgan-critical'
  email_configs:
  - to: 'oncall@jpmorgan.com'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'
    channel: '#critical-alerts'
  pagerduty_configs:
  - service_key: 'xxxxx'

- name: 'jpmorgan-security'
  email_configs:
  - to: 'security@jpmorgan.com'
    subject: 'SECURITY: {{ .GroupLabels.alertname }}'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'
    channel: '#security'
```

## Log Aggregation

### Fluentd Configuration

```yaml
# fluent.conf
<source>
  @type tail
  path /var/log/containers/*jpmorgan*.log
  pos_file /var/log/fluentd-containers.log.pos
  tag kubernetes.*
  <parse>
    @type json
    time_key time
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

<filter kubernetes.**>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    cluster "jpmorgan-apis"
    namespace "jpmorgan-apis"
  </record>
</filter>

<match kubernetes.**>
  @type elasticsearch
  host elasticsearch.jpmorgan-apis.svc.cluster.local
  port 9200
  logstash_format true
  logstash_prefix jpmorgan-apis
  <buffer>
    @type file
    path /var/log/fluentd-buffers/kubernetes.system.buffer
    flush_mode interval
    retry_type exponential_backoff
    flush_interval 5s
    retry_forever
    retry_max_interval 30
    chunk_limit_size 2M
    queue_limit_length 8
    overflow_action block
  </buffer>
</match>
```

### Elasticsearch Index Templates

```json
{
  "index_patterns": ["jpmorgan-apis-*"],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index.codec": "best_compression",
    "refresh_interval": "30s"
  },
  "mappings": {
    "properties": {
      "@timestamp": {
        "type": "date"
      },
      "level": {
        "type": "keyword"
      },
      "message": {
        "type": "text",
        "analyzer": "standard"
      },
      "service": {
        "type": "keyword"
      },
      "user_id": {
        "type": "keyword"
      },
      "request_id": {
        "type": "keyword"
      },
      "endpoint": {
        "type": "keyword"
      },
      "method": {
        "type": "keyword"
      },
      "status_code": {
        "type": "integer"
      },
      "response_time": {
        "type": "float"
      },
      "ip_address": {
        "type": "ip"
      },
      "user_agent": {
        "type": "text"
      }
    }
  }
}
```

## Distributed Tracing

### Jaeger Configuration

```yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: jpmorgan-apis
spec:
  strategy: production
  collector:
    options:
      collector.otlp.enabled: true
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
        index-prefix: jaeger
  ui:
    options:
      dependencies:
        menuEnabled: false
      tracking:
        gaID: UA-000000-2
```

### Application Tracing

```python
from jaeger_client import Config
from flask_opentracing import FlaskTracing

def init_tracing(app):
    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'local_agent': {
                'reporting_host': 'jaeger-agent.jpmorgan-apis.svc.cluster.local',
                'reporting_port': 6831,
            },
            'logging': True,
        },
        service_name='jpmorgan-financial-apis',
    )

    jaeger_tracer = config.initialize_tracer()
    tracing = FlaskTracing(jaeger_tracer, True, app)

    return tracing

# Usage in Flask routes
@app.route('/api/accounts')
@tracer.trace()
def get_accounts():
    with tracer.trace('get_accounts') as span:
        span.set_tag('user.id', get_current_user_id())
        span.set_tag('operation', 'list_accounts')

        # Your business logic here
        accounts = get_accounts_from_db()

        span.set_tag('accounts.count', len(accounts))
        return jsonify(accounts)
```

## Incident Response

### Alert Classification

```python
class AlertClassifier:
    def __init__(self):
        self.severity_levels = {
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }

    def classify_alert(self, alert):
        """Classify alert based on rules"""
        alert_name = alert.get('labels', {}).get('alertname', '')
        severity = alert.get('labels', {}).get('severity', 'info')

        # Business impact assessment
        impact = self._assess_business_impact(alert)

        # Urgency calculation
        urgency = self._calculate_urgency(severity, impact)

        # Response time calculation
        response_time = self._calculate_response_time(urgency)

        return {
            'severity': severity,
            'impact': impact,
            'urgency': urgency,
            'response_time_minutes': response_time,
            'escalation_required': urgency >= 4
        }

    def _assess_business_impact(self, alert):
        """Assess business impact of alert"""
        alert_name = alert.get('labels', {}).get('alertname', '')

        impact_rules = {
            'HighErrorRate': 'high',
            'DatabaseDown': 'critical',
            'SecurityBreach': 'critical',
            'HighRequestLatency': 'medium',
            'LowDiskSpace': 'low'
        }

        return impact_rules.get(alert_name, 'medium')

    def _calculate_urgency(self, severity, impact):
        """Calculate urgency based on severity and impact"""
        severity_score = self.severity_levels.get(severity, 1)
        impact_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        impact_score = impact_scores.get(impact, 2)

        return max(severity_score, impact_score)

    def _calculate_response_time(self, urgency):
        """Calculate response time in minutes"""
        response_times = {
            1: 240,  # 4 hours for info
            2: 60,   # 1 hour for warning
            3: 30,   # 30 minutes for error
            4: 15    # 15 minutes for critical
        }

        return response_times.get(urgency, 60)
```

### Automated Response

```python
class IncidentResponder:
    def __init__(self, k8s_client, notification_service):
        self.k8s = k8s_client
        self.notification = notification_service

    def handle_alert(self, alert):
        """Handle incoming alert"""
        alert_name = alert.get('labels', {}).get('alertname', '')

        # Classify alert
        classification = AlertClassifier().classify_alert(alert)

        # Log incident
        self._log_incident(alert, classification)

        # Execute automated response
        self._execute_automated_response(alert_name, classification)

        # Notify if required
        if classification['escalation_required']:
            self._notify_oncall_team(alert, classification)

    def _execute_automated_response(self, alert_name, classification):
        """Execute automated response based on alert type"""
        responses = {
            'HighCPUUsage': self._scale_up_pods,
            'DatabaseConnectionPoolExhausted': self._restart_database_connections,
            'HighErrorRate': self._enable_circuit_breaker,
            'LowDiskSpace': self._cleanup_disk_space
        }

        response_func = responses.get(alert_name)
        if response_func:
            try:
                response_func()
                self._log_response(alert_name, 'success')
            except Exception as e:
                self._log_response(alert_name, 'failed', str(e))

    def _scale_up_pods(self):
        """Scale up application pods"""
        self.k8s.apps_v1.patch_namespaced_deployment_scale(
            name='jpmorgan-financial-apis',
            namespace='jpmorgan-apis',
            body={'spec': {'replicas': 10}}
        )

    def _restart_database_connections(self):
        """Restart database connection pool"""
        # Implementation would restart connection pool
        pass

    def _enable_circuit_breaker(self):
        """Enable circuit breaker for failing services"""
        # Implementation would enable circuit breaker
        pass

    def _cleanup_disk_space(self):
        """Clean up disk space"""
        # Implementation would clean up temporary files
        pass

    def _log_incident(self, alert, classification):
        """Log incident details"""
        # Implementation would log to incident management system
        pass

    def _log_response(self, alert_name, status, error=None):
        """Log automated response result"""
        # Implementation would log response action
        pass

    def _notify_oncall_team(self, alert, classification):
        """Notify on-call team"""
        message = f"""
        🚨 CRITICAL ALERT

        Alert: {alert.get('labels', {}).get('alertname')}
        Severity: {classification['severity']}
        Impact: {classification['impact']}

        Description: {alert.get('annotations', {}).get('description', '')}

        Response Time: {classification['response_time_minutes']} minutes

        Please investigate immediately.
        """

        self.notification.send_sms(message)
        self.notification.send_slack(message, channel='#oncall')
```

## Best Practices

### Monitoring Strategy

1. **Define Key Metrics**: Focus on business and technical KPIs
2. **Set Appropriate Thresholds**: Avoid alert fatigue
3. **Implement Alert Escalation**: Different severity levels
4. **Regular Review**: Update monitoring as system evolves
5. **Documentation**: Document all alerts and responses

### Alert Management

1. **Alert Classification**: Categorize by severity and impact
2. **Deduplication**: Avoid duplicate alerts
3. **Correlation**: Group related alerts
4. **Suppression**: Suppress during maintenance
5. **Feedback Loop**: Learn from false positives

### Dashboard Design

1. **User-Centric**: Design for different user roles
2. **Real-time Data**: Show current system state
3. **Historical Trends**: Include time-based comparisons
4. **Actionable Insights**: Provide actionable information
5. **Mobile Friendly**: Ensure mobile accessibility

### Log Management

1. **Structured Logging**: Use consistent log format
2. **Log Levels**: Appropriate log level usage
3. **Retention Policies**: Define log retention periods
4. **Search and Analysis**: Enable efficient log searching
5. **Security**: Secure log storage and access

---

**Last Updated**: November 2024
**Version**: 1.0.0
