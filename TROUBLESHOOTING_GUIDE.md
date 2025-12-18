# Troubleshooting Guide - JPMorgan Financial APIs

## Overview

This guide provides solutions to common issues encountered when deploying, configuring, and operating the JPMorgan Financial APIs system. Issues are organized by category for quick reference.

## Quick Health Check

Before troubleshooting specific issues, run this quick health check:

```bash
# Check all services
docker-compose -f docker-compose.production.yml ps

# Check application health
curl -f http://localhost:8000/health

# Check database connectivity
docker-compose -f docker-compose.production.yml exec postgres pg_isready

# Check Redis connectivity
docker-compose -f docker-compose.production.yml exec redis redis-cli ping

# Check Prometheus metrics
curl -f http://localhost:9090/-/healthy

# Check Grafana
curl -f http://localhost:3000/api/health
```

## Database Issues

### PostgreSQL Connection Failed

**Symptoms:**
- Application logs show "connection refused" or "connection timeout"
- Health check fails with database errors
- `pg_isready` command fails

**Solutions:**

1. **Check if PostgreSQL is running:**
   ```bash
   docker-compose -f docker-compose.production.yml ps postgres
   ```

2. **Check PostgreSQL logs:**
   ```bash
   docker-compose -f docker-compose.production.yml logs postgres
   ```

3. **Verify connection string:**
   ```bash
   # Check environment variables
   docker-compose -f docker-compose.production.yml exec app env | grep DATABASE_URL
   ```

4. **Test database connectivity from application:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.database import test_connection
   test_connection()
   "
   ```

5. **Restart PostgreSQL:**
   ```bash
   docker-compose -f docker-compose.production.yml restart postgres
   ```

### Database Migration Issues

**Symptoms:**
- Application fails to start with migration errors
- Tables not created properly
- Schema version conflicts

**Solutions:**

1. **Check migration status:**
   ```bash
   docker-compose -f docker-compose.production.yml exec postgres psql -U jpmorgan -d jpmorgan_api -c "SELECT * FROM alembic_version;"
   ```

2. **Run migrations manually:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.database import init_database
   init_database()
   "
   ```

3. **Reset database (development only):**
   ```bash
   docker-compose -f docker-compose.production.yml exec postgres psql -U jpmorgan -d jpmorgan_api -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   ```

### High Database CPU/Memory Usage

**Symptoms:**
- Database pod shows high resource usage
- Queries are slow
- Connection pool exhausted

**Solutions:**

1. **Check active connections:**
   ```bash
   docker-compose -f docker-compose.production.yml exec postgres psql -U jpmorgan -d jpmorgan_api -c "
   SELECT count(*) as connections FROM pg_stat_activity;
   "
   ```

2. **Check long-running queries:**
   ```bash
   docker-compose -f docker-compose.production.yml exec postgres psql -U jpmorgan -d jpmorgan_api -c "
   SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE state = 'active' AND now() - pg_stat_activity.query_start > interval '1 minute';
   "
   ```

3. **Check query performance:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.telemetry_handler import get_query_performance_stats
   import json
   print(json.dumps(get_query_performance_stats(), indent=2))
   "
   ```

4. **Optimize connection pool settings:**
   ```yaml
   # In docker-compose.production.yml
   postgres:
     environment:
       POSTGRES_MAX_CONNECTIONS: 50  # Adjust based on load
   ```

## Redis/Cache Issues

### Redis Connection Failed

**Symptoms:**
- Application logs show Redis connection errors
- Caching not working
- Session management fails

**Solutions:**

1. **Check Redis status:**
   ```bash
   docker-compose -f docker-compose.production.yml exec redis redis-cli ping
   ```

2. **Check Redis logs:**
   ```bash
   docker-compose -f docker-compose.production.yml logs redis
   ```

3. **Verify Redis configuration:**
   ```bash
   docker-compose -f docker-compose.production.yml exec redis redis-cli config get maxmemory
   ```

4. **Test Redis connectivity:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   import redis
   r = redis.from_url('redis://redis:6379')
   print('Redis ping:', r.ping())
   "
   ```

### Cache Performance Issues

**Symptoms:**
- Slow response times
- High cache miss rates
- Memory usage spikes

**Solutions:**

1. **Check cache hit rates:**
   ```bash
   # Via Prometheus metrics
   curl http://localhost:9090/api/v1/query?query=cache_hit_ratio
   ```

2. **Clear cache if corrupted:**
   ```bash
   docker-compose -f docker-compose.production.yml exec redis redis-cli FLUSHALL
   ```

3. **Adjust cache TTL settings:**
   ```python
   # In app.py
   @cache_database_query(expiration=600)  # Increase TTL
   def get_expensive_metrics(hours=24):
       return database.get_metrics_summary(hours)
   ```

## JP Morgan API Issues

### Authentication Failed

**Symptoms:**
- "Invalid token" errors
- 401 Unauthorized responses
- Token refresh failures

**Solutions:**

1. **Check API credentials:**
   ```bash
   # Verify environment variables are set
   docker-compose -f docker-compose.production.yml exec app env | grep JPMORGAN
   ```

2. **Test token generation:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.jpmorgan_client import JPMorganClient
   client = JPMorganClient()
   token = client.get_access_token()
   print('Token obtained:', bool(token))
   "
   ```

3. **Check token expiration:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.jpmorgan_client import JPMorganClient
   client = JPMorganClient()
   print('Token status:', client.get_token_status())
   "
   ```

4. **Verify API endpoints:**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
        https://api.payments.jpmorgan.com/accounts
   ```

### API Rate Limiting

**Symptoms:**
- 429 Too Many Requests errors
- Intermittent API failures
- Slow response times

**Solutions:**

1. **Check rate limit headers:**
   ```bash
   curl -v -H "Authorization: Bearer YOUR_TOKEN" \
        https://api.payments.jpmorgan.com/accounts 2>&1 | grep -i rate
   ```

2. **Implement backoff strategy:**
   ```python
   # In jpmorgan_client.py
   import time
   from requests.adapters import HTTPAdapter
   from urllib3.util.retry import Retry

   retry_strategy = Retry(
       total=3,
       backoff_factor=2,
       status_forcelist=[429, 500, 502, 503, 504]
   )
   adapter = HTTPAdapter(max_retries=retry_strategy)
   ```

3. **Monitor API usage:**
   ```bash
   # Check Prometheus metrics
   curl http://localhost:9090/api/v1/query?query=jpmorgan_api_requests_total
   ```

## Application Performance Issues

### High CPU Usage

**Symptoms:**
- Application pod shows high CPU usage
- Slow response times
- Request timeouts

**Solutions:**

1. **Profile application performance:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   import cProfile
   from src.telemetry_handler import telemetry_handler
   cProfile.run('telemetry_handler.get_metrics(hours=24)', 'profile.stats')
   "
   ```

2. **Check thread usage:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   import threading
   print('Active threads:', threading.active_count())
   for t in threading.enumerate():
       print(f'  {t.name}: {t.is_alive()}')
   "
   ```

3. **Enable async processing:**
   ```python
   # In app.py, ensure async processing is enabled
   import asyncio
   from .async_utils import process_batch_async

   async def process_telemetry_batch(data):
       return await process_batch_async(data)
   ```

### High Memory Usage

**Symptoms:**
- Application pod shows high memory usage
- Out of memory errors
- Frequent garbage collection

**Solutions:**

1. **Check memory usage:**
   ```bash
   docker stats $(docker-compose -f docker-compose.production.yml ps -q app)
   ```

2. **Profile memory usage:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   import tracemalloc
   tracemalloc.start()
   # Run some operations
   from src.telemetry_handler import telemetry_handler
   result = telemetry_handler.get_metrics(hours=24)
   current, peak = tracemalloc.get_traced_memory()
   print(f'Current memory usage: {current / 1024 / 1024:.1f} MB')
   print(f'Peak memory usage: {peak / 1024 / 1024:.1f} MB')
   "
   ```

3. **Optimize batch processing:**
   ```python
   # Reduce batch size in config
   TELEMETRY_BATCH_SIZE = 50  # Instead of 100
   ```

4. **Enable memory profiling:**
   ```python
   # In app.py
   import gc
   import psutil

   @app.before_request
   def memory_check():
       process = psutil.Process()
       memory_usage = process.memory_info().rss / 1024 / 1024
       if memory_usage > 500:  # MB
           gc.collect()
           app.logger.warning(f'High memory usage: {memory_usage:.1f} MB')
   ```

### Slow API Responses

**Symptoms:**
- API response times > 1 second
- Timeout errors
- Poor user experience

**Solutions:**

1. **Check response times:**
   ```bash
   # Use curl with timing
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
   ```

   curl-format.txt:
   ```
   time_namelookup:  %{time_namelookup}\n
   time_connect: %{time_connect}\n
   time_appconnect: %{time_appconnect}\n
   time_pretransfer: %{time_pretransfer}\n
   time_redirect: %{time_redirect}\n
   time_starttransfer: %{time_starttransfer}\n
   ----------\n
   time_total: %{time_total}\n
   ```

2. **Check database query performance:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.telemetry_handler import get_query_performance_stats
   stats = get_query_performance_stats()
   for query, metrics in stats.items():
       if metrics['avg_time'] > 1.0:
           print(f'SLOW QUERY: {query} - {metrics[\"avg_time\"]}s avg')
   "
   ```

3. **Optimize caching:**
   ```python
   # Increase cache TTL for frequently accessed data
   @cache_database_query(expiration=1800)  # 30 minutes
   def get_telemetry_metrics(hours=24):
       return telemetry_handler.get_metrics(hours)
   ```

4. **Enable connection pooling:**
   ```python
   # In telemetry_handler.py
   self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
       minconn=5,
       maxconn=20,  # Adjust based on load
       dsn=self.db_url
   )
   ```

## ML/Anomaly Detection Issues

### GPU Not Available

**Symptoms:**
- ML models running on CPU
- Slow anomaly detection
- High CPU usage during ML operations

**Solutions:**

1. **Check GPU availability:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.ml_model import AnomalyDetector
   detector = AnomalyDetector()
   print('GPU stats:', detector.get_gpu_stats())
   "
   ```

2. **Install GPU drivers in container:**
   ```dockerfile
   # In Dockerfile
   FROM nvidia/cuda:11.8-runtime-ubuntu20.04

   # Install Python and dependencies
   RUN apt-get update && apt-get install -y python3 python3-pip
   ```

3. **Enable GPU in docker-compose:**
   ```yaml
   app:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

### Model Training Failures

**Symptoms:**
- ML training fails
- Model not saved
- Anomaly detection not working

**Solutions:**

1. **Check training data:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.data_processor import prepare_for_ml
   import numpy as np

   # Test with sample data
   sample_data = [{'feature1': 1.0, 'feature2': 2.0}]
   features, df = prepare_for_ml(sample_data)
   print('Feature matrix shape:', features.shape)
   "
   ```

2. **Check model persistence:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app ls -la /app/models/
   ```

3. **Retrain model:**
   ```bash
   docker-compose -f docker-compose.production.yml exec app python -c "
   from src.ml_model import AnomalyDetector
   from src.data_processor import prepare_for_ml

   # Load training data
   # ... load your training data ...

   detector = AnomalyDetector()
   success = detector.train(features)
   print('Model training:', 'SUCCESS' if success else 'FAILED')
   "
   ```

## Monitoring and Alerting Issues

### Prometheus Metrics Not Available

**Symptoms:**
- Grafana dashboards show no data
- Metrics endpoint returns errors
- Monitoring alerts not working

**Solutions:**

1. **Check Prometheus status:**
   ```bash
   docker-compose -f docker-compose.production.yml ps prometheus
   curl http://localhost:9090/-/healthy
   ```

2. **Check metrics endpoint:**
   ```bash
   curl http://localhost:8000/metrics
   ```

3. **Verify Prometheus configuration:**
   ```yaml
   # In prometheus.yml
   scrape_configs:
     - job_name: 'jpmorgan-api'
       static_configs:
         - targets: ['app:8000']
   ```

4. **Check Grafana data sources:**
   ```bash
   curl http://localhost:3000/api/datasources
   ```

### Alert Manager Not Working

**Symptoms:**
- Alerts not being sent
- Alert manager logs show errors
- No alert notifications

**Solutions:**

1. **Check AlertManager status:**
   ```bash
   docker-compose -f docker-compose.production.yml ps alertmanager
   curl http://localhost:9093/-/healthy
   ```

2. **Check alert rules:**
   ```yaml
   # In alert_rules.yml
   groups:
   - name: jpmorgan-api
     rules:
     - alert: HighErrorRate
       expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
       for: 5m
       labels:
         severity: critical
       annotations:
         summary: "High error rate detected"
   ```

3. **Test alert delivery:**
   ```bash
   # Send test alert
   curl -X POST http://localhost:9093/api/v2/alerts \
        -H "Content-Type: application/json" \
        -d '[{"labels":{"alertname":"TestAlert","severity":"warning"},"annotations":{"summary":"Test alert"}}]'
   ```

## Networking and Security Issues

### SSL/TLS Certificate Issues

**Symptoms:**
- HTTPS not working
- Certificate validation errors
- Mixed content warnings

**Solutions:**

1. **Check certificate validity:**
   ```bash
   openssl s_client -connect localhost:443 -servername yourdomain.com < /dev/null 2>/dev/null | openssl x509 -noout -dates
   ```

2. **Renew certificates:**
   ```bash
   # Using certbot
   certbot renew

   # Or with cert-manager (Kubernetes)
   kubectl get certificates
   kubectl describe certificate your-cert
   ```

3. **Check certificate configuration:**
   ```nginx
   # In nginx.conf
   server {
       listen 443 ssl http2;
       server_name yourdomain.com;

       ssl_certificate /etc/ssl/certs/yourdomain.com.crt;
       ssl_certificate_key /etc/ssl/private/yourdomain.com.key;

       # SSL configuration
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
   }
   ```

### Firewall/Network Issues

**Symptoms:**
- Services can't communicate
- External access blocked
- Intermittent connectivity

**Solutions:**

1. **Check network connectivity:**
   ```bash
   # Test internal connectivity
   docker-compose -f docker-compose.production.yml exec app ping postgres
   docker-compose -f docker-compose.production.yml exec app ping redis
   ```

2. **Check firewall rules:**
   ```bash
   # Linux
   sudo ufw status
   sudo iptables -L

   # Windows
   netsh advfirewall show currentprofile
   ```

3. **Verify port bindings:**
   ```bash
   netstat -tlnp | grep :8000
   netstat -tlnp | grep :5432
   ```

## Log Analysis

### Application Logs

**Common log patterns and solutions:**

1. **Database connection errors:**
   ```
   ERROR - Connection to database failed: FATAL: too many connections
   ```
   Solution: Increase max_connections in PostgreSQL or implement connection pooling.

2. **Memory errors:**
   ```
   ERROR - MemoryError: Out of memory
   ```
   Solution: Increase container memory limits or optimize memory usage.

3. **API rate limit errors:**
   ```
   WARNING - Rate limit exceeded for JP Morgan API
   ```
   Solution: Implement exponential backoff or increase rate limits.

### System Logs

**Check system-level logs:**

```bash
# Docker logs
docker-compose -f docker-compose.production.yml logs -f app

# System logs
journalctl -u docker -f

# Kubernetes logs (if applicable)
kubectl logs -f deployment/jpmorgan-api
```

### Log Levels

**Adjust log levels for troubleshooting:**

```python
# In config.py
LOG_LEVEL = 'DEBUG'  # Change to INFO for production

# In app.py
app.logger.setLevel(logging.DEBUG)
telemetry_logger.get_logger().setLevel(logging.DEBUG)
```

## Emergency Procedures

### Complete System Restart

```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Clean up volumes (CAUTION: destroys data)
docker volume prune -f

# Restart services
docker-compose -f docker-compose.production.yml up -d

# Check health
curl http://localhost:8000/health
```

### Database Recovery

```bash
# Create backup
docker-compose -f docker-compose.production.yml exec postgres pg_dump -U jpmorgan jpmorgan_api > backup.sql

# Restore backup
docker-compose -f docker-compose.production.yml exec -T postgres psql -U jpmorgan jpmorgan_api < backup.sql
```

### Configuration Rollback

```bash
# Backup current config
cp docker-compose.production.yml docker-compose.production.yml.backup

# Restore previous config
cp docker-compose.production.yml.bak docker-compose.production.yml

# Restart services
docker-compose -f docker-compose.production.yml up -d
```

## Performance Troubleshooting Checklist

- [ ] Check system resources (CPU, memory, disk)
- [ ] Review application logs for errors
- [ ] Monitor database query performance
- [ ] Check cache hit rates
- [ ] Verify network connectivity
- [ ] Test API endpoints individually
- [ ] Monitor external API dependencies
- [ ] Check for memory leaks
- [ ] Review configuration settings
- [ ] Test with different load patterns

## Getting Help

If you can't resolve an issue:

1. **Collect diagnostic information:**
   ```bash
   # System info
   uname -a
   docker --version
   docker-compose --version

   # Service status
   docker-compose -f docker-compose.production.yml ps

   # Logs
   docker-compose -f docker-compose.production.yml logs --tail=100 > logs.txt
   ```

2. **Check the documentation:**
   - [README.md](README.md) - Setup and configuration
   - [API Reference](docs/api_reference.md) - Endpoint documentation
   - [Architecture Guide](docs/architecture.md) - System design

3. **Contact support:**
   - Create an issue on GitHub
   - Email: support@your-org.com
   - Include diagnostic information and steps to reproduce

---

**Remember:** Always test changes in a development environment before applying to production!
