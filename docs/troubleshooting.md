# Troubleshooting Guide - JPMorgan Financial APIs

## Overview

This guide provides solutions for common issues encountered with the JPMorgan Financial APIs platform.

## Authentication Issues

### OAuth2 Token Retrieval Failures

**Symptom**: 400 Bad Request when requesting access tokens

**Possible Causes**:
- Invalid client credentials
- Incorrect token URL
- Network connectivity issues
- JPMorgan API service downtime

**Solutions**:

1. **Verify Credentials**:
   ```bash
   # Check environment variables
   echo $TOKEN_CLIENT_ID
   echo $TOKEN_CLIENT_SECRET

   # Test with curl
   curl -X POST "https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -u "$TOKEN_CLIENT_ID:$TOKEN_CLIENT_SECRET" \
     -d "grant_type=client_credentials"
   ```

2. **Check Token URL**:
   ```python
   from config import config
   print(f"Token URL: {config.TOKEN_URL}")
   # Should be: https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token
   ```

3. **Network Connectivity**:
   ```bash
   # Test connectivity to JPMorgan
   curl -I https://id.payments.jpmorgan.com

   # Check DNS resolution
   nslookup id.payments.jpmorgan.com
   ```

### Invalid Token Errors

**Symptom**: 401 Unauthorized responses

**Solutions**:

1. **Check Token Expiration**:
   ```python
   import time
   from src.token_manager import token_manager

   # Check if token is expired
   if time.time() > token_manager.token_expires_at:
       print("Token expired, refreshing...")
       token_manager.get_token()
   ```

2. **Validate Token Format**:
   ```python
   import jwt

   token = token_manager.access_token
   try:
       decoded = jwt.decode(token, options={"verify_signature": False})
       print(f"Token expires: {decoded.get('exp')}")
   except Exception as e:
       print(f"Invalid token format: {e}")
   ```

## Database Connection Issues

### Connection Pool Exhaustion

**Symptom**: Database connection errors, slow queries

**Solutions**:

1. **Check Connection Pool Status**:
   ```bash
   # Monitor database connections
   kubectl exec -it deployment/postgresql -- psql -U jpmorgan_user -d jpmorgan_financial_apis -c "SELECT * FROM pg_stat_activity;"

   # Check application connection pool
   kubectl logs deployment/jpmorgan-financial-apis | grep "connection pool"
   ```

2. **Adjust Connection Pool Settings**:
   ```yaml
   # Update config
   DATABASE_CONNECTION_POOL_SIZE: "20"
   DATABASE_CONNECTION_POOL_MAX_OVERFLOW: "30"
   DATABASE_CONNECTION_POOL_TIMEOUT: "30"
   ```

3. **Restart Application**:
   ```bash
   kubectl rollout restart deployment/jpmorgan-financial-apis
   ```

### Database Migration Failures

**Symptom**: Migration scripts fail during deployment

**Solutions**:

1. **Check Migration Status**:
   ```bash
   # View migration history
   kubectl exec -it deployment/postgresql -- psql -U jpmorgan_user -d jpmorgan_financial_apis -c "SELECT * FROM alembic_version;"

   # Run migrations manually
   python scripts/postgresql_migration.py --check
   ```

2. **Fix Migration Conflicts**:
   ```bash
   # Downgrade and re-run
   python scripts/postgresql_migration.py --downgrade
   python scripts/postgresql_migration.py
   ```

## Redis/Caching Issues

### Redis Connection Failures

**Symptom**: Cache operations failing, slow performance

**Solutions**:

1. **Check Redis Cluster Status**:
   ```bash
   # Connect to Redis
   kubectl exec -it redis-cluster-0 -- redis-cli cluster nodes

   # Test basic operations
   kubectl exec -it redis-cluster-0 -- redis-cli ping
   ```

2. **Verify Application Configuration**:
   ```python
   from config import config
   print(f"Redis URL: {config.REDIS_URL}")

   # Test connection
   import redis
   r = redis.from_url(config.REDIS_URL)
   r.ping()
   ```

3. **Restart Redis Cluster**:
   ```bash
   kubectl rollout restart statefulset/redis-cluster
   ```

### Cache Invalidation Issues

**Symptom**: Stale data served from cache

**Solutions**:

1. **Clear Cache Manually**:
   ```bash
   # Clear all cache
   kubectl exec -it redis-cluster-0 -- redis-cli FLUSHALL

   # Clear specific keys
   kubectl exec -it redis-cluster-0 -- redis-cli KEYS "jpmorgan:*" | xargs redis-cli DEL
   ```

2. **Update Cache TTL**:
   ```python
   # Adjust cache expiration
   CACHE_TTL = 300  # 5 minutes
   ```

## Application Performance Issues

### High CPU Usage

**Symptom**: Pods consuming excessive CPU

**Solutions**:

1. **Profile Application**:
   ```bash
   # Use kubectl top
   kubectl top pods

   # Check resource limits
   kubectl describe pod <pod-name>
   ```

2. **Enable Profiling**:
   ```python
   import cProfile

   def profile_function(func):
       def wrapper(*args, **kwargs):
           profiler = cProfile.Profile()
           profiler.enable()
           result = func(*args, **kwargs)
           profiler.disable()
           profiler.print_stats(sort='cumulative')
           return result
       return wrapper
   ```

3. **Scale Resources**:
   ```yaml
   resources:
     requests:
       cpu: 1000m
       memory: 2Gi
     limits:
       cpu: 2000m
       memory: 4Gi
   ```

### Memory Leaks

**Symptom**: Increasing memory usage over time

**Solutions**:

1. **Monitor Memory Usage**:
   ```bash
   # Check memory usage
   kubectl top pods --containers

   # Monitor garbage collection
   kubectl logs deployment/jpmorgan-financial-apis | grep "GC"
   ```

2. **Enable Memory Profiling**:
   ```python
   import tracemalloc

   tracemalloc.start()

   # Check memory usage
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory usage: {current / 1024 / 1024} MB")
   print(f"Peak memory usage: {peak / 1024 / 1024} MB")
   ```

3. **Optimize Memory Usage**:
   ```python
   # Use generators for large datasets
   def get_large_dataset():
       for item in large_list:
           yield item  # Instead of return large_list
   ```

### Slow API Responses

**Symptom**: API endpoints responding slowly

**Solutions**:

1. **Check Database Query Performance**:
   ```sql
   -- Enable query logging
   SET log_statement = 'all';
   SET log_duration = 'on';

   -- Analyze slow queries
   SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
   ```

2. **Add Database Indexes**:
   ```sql
   CREATE INDEX CONCURRENTLY idx_accounts_user_id ON accounts(user_id);
   CREATE INDEX CONCURRENTLY idx_transactions_account_id ON transactions(account_id);
   ```

3. **Implement Caching**:
   ```python
   @app.cache.memoize(timeout=300)
   def get_account_balance(account_id):
       # Cache expensive operations
       return database_query(account_id)
   ```

## Kubernetes Issues

### Pod Crashes

**Symptom**: Pods restarting frequently

**Solutions**:

1. **Check Pod Logs**:
   ```bash
   # View recent logs
   kubectl logs deployment/jpmorgan-financial-apis --previous

   # Stream logs
   kubectl logs deployment/jpmorgan-financial-apis -f
   ```

2. **Check Resource Limits**:
   ```bash
   # Describe pod
   kubectl describe pod <pod-name>

   # Check events
   kubectl get events --sort-by=.metadata.creationTimestamp
   ```

3. **Debug Container**:
   ```bash
   # Execute into container
   kubectl exec -it deployment/jpmorgan-financial-apis -- /bin/bash

   # Check application status
   ps aux | grep python
   ```

### Service Mesh Issues

**Symptom**: Traffic not routing correctly through Istio

**Solutions**:

1. **Check Istio Configuration**:
   ```bash
   # List virtual services
   kubectl get virtualservice -n jpmorgan-apis

   # Check destination rules
   kubectl get destinationrule -n jpmorgan-apis
   ```

2. **Verify Sidecar Injection**:
   ```bash
   # Check if sidecar is injected
   kubectl get pods -o jsonpath='{.items[*].spec.containers[*].name}'
   ```

3. **Test Service Communication**:
   ```bash
   # Use istioctl
   istioctl proxy-status

   # Check traffic policies
   kubectl get peerauthentication -n jpmorgan-apis
   ```

## Network Issues

### Load Balancer Problems

**Symptom**: Traffic not reaching application

**Solutions**:

1. **Check Load Balancer Status**:
   ```bash
   # AWS Load Balancer
   aws elbv2 describe-load-balancers --names jpmorgan-api-lb

   # Kubernetes service
   kubectl get svc jpmorgan-financial-apis
   ```

2. **Verify Health Checks**:
   ```bash
   # Check health endpoint
   curl http://localhost:8000/health

   # Check load balancer health
   kubectl describe svc jpmorgan-financial-apis
   ```

### DNS Resolution Issues

**Symptom**: Domain name not resolving

**Solutions**:

1. **Check DNS Configuration**:
   ```bash
   # Test DNS resolution
   nslookup api.jpmorgan.com

   # Check DNS records
   dig api.jpmorgan.com
   ```

2. **Update DNS Records**:
   ```bash
   # AWS Route 53
   aws route53 list-resource-record-sets --hosted-zone-id Z123456789

   # Update if necessary
   aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://dns-update.json
   ```

## Monitoring and Alerting Issues

### Missing Metrics

**Symptom**: Metrics not appearing in Prometheus/Grafana

**Solutions**:

1. **Check Service Monitors**:
   ```bash
   # List service monitors
   kubectl get servicemonitor -n jpmorgan-apis

   # Check Prometheus targets
   kubectl port-forward svc/prometheus 9090:9090
   # Visit http://localhost:9090/targets
   ```

2. **Verify Metrics Endpoint**:
   ```bash
   # Check metrics endpoint
   curl http://localhost:9090/metrics

   # Check application metrics
   kubectl exec -it deployment/jpmorgan-financial-apis -- curl http://localhost:8000/metrics
   ```

### Alert Not Firing

**Symptom**: Expected alerts not triggering

**Solutions**:

1. **Check Alert Rules**:
   ```bash
   # View alert rules
   kubectl get prometheusrules -n monitoring

   # Check alert manager
   kubectl port-forward svc/alertmanager 9093:9093
   ```

2. **Test Alert Conditions**:
   ```bash
   # Query Prometheus
   kubectl port-forward svc/prometheus 9090:9090
   # Visit http://localhost:9090/graph
   # Query: up{job="jpmorgan-financial-apis"}
   ```

## Security Issues

### SSL/TLS Problems

**Symptom**: SSL certificate errors

**Solutions**:

1. **Check Certificate Status**:
   ```bash
   # Test SSL connection
   openssl s_client -connect api.jpmorgan.com:443

   # Check certificate expiry
   echo | openssl s_client -connect api.jpmorgan.com:443 2>/dev/null | openssl x509 -noout -dates
   ```

2. **Renew Certificates**:
   ```bash
   # Using cert-manager
   kubectl get certificate -n jpmorgan-apis

   # Manual renewal
   kubectl apply -f k8s/certificates.yml
   ```

### Access Control Issues

**Symptom**: Unauthorized access or permission errors

**Solutions**:

1. **Check RBAC Configuration**:
   ```bash
   # List roles and bindings
   kubectl get roles,rolebindings -n jpmorgan-apis

   # Check service account permissions
   kubectl auth can-i get pods --as=system:serviceaccount:jpmorgan-apis:default
   ```

2. **Verify Network Policies**:
   ```bash
   # List network policies
   kubectl get networkpolicy -n jpmorgan-apis

   # Test connectivity
   kubectl run test-pod --image=busybox --rm -it -- wget --timeout=5 api.jpmorgan.com
   ```

## Performance Tuning

### Database Optimization

**Solutions**:

1. **Analyze Query Performance**:
   ```sql
   -- Find slow queries
   SELECT query, total_time, calls, mean_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;

   -- Check table statistics
   ANALYZE VERBOSE accounts;
   ```

2. **Optimize Configuration**:
   ```yaml
   # PostgreSQL config
   shared_buffers: 256MB
   effective_cache_size: 1GB
   work_mem: 4MB
   maintenance_work_mem: 64MB
   ```

### Application Optimization

**Solutions**:

1. **Enable Gzip Compression**:
   ```python
   from flask import Flask
   from flask_compress import Compress

   app = Flask(__name__)
   Compress(app)
   ```

2. **Implement Connection Pooling**:
   ```python
   from sqlalchemy import create_engine

   engine = create_engine(
       config.DATABASE_URL,
       pool_size=10,
       max_overflow=20,
       pool_timeout=30,
       pool_recycle=3600
   )
   ```

3. **Add Response Caching**:
   ```python
   from flask_caching import Cache

   cache = Cache(app, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': config.REDIS_URL})

   @app.route('/api/accounts')
   @cache.cached(timeout=300)
   def get_accounts():
       return jsonify(accounts)
   ```

## Emergency Procedures

### Service Outage

1. **Assess Impact**:
   ```bash
   # Check service status
   kubectl get pods -n jpmorgan-apis
   kubectl get svc -n jpmorgan-apis
   ```

2. **Check Monitoring**:
   - Review Grafana dashboards
   - Check alert history
   - Monitor error rates

3. **Execute Recovery**:
   ```bash
   # Scale up if needed
   kubectl scale deployment jpmorgan-financial-apis --replicas=20

   # Restart services
   kubectl rollout restart deployment/jpmorgan-financial-apis

   # Check logs for errors
   kubectl logs deployment/jpmorgan-financial-apis --tail=100
   ```

### Data Recovery

1. **Identify Data Loss**:
   ```bash
   # Check database status
   kubectl exec -it deployment/postgresql -- pg_isready

   # Verify data integrity
   kubectl exec -it deployment/postgresql -- psql -U jpmorgan_user -d jpmorgan_financial_apis -c "SELECT COUNT(*) FROM accounts;"
   ```

2. **Restore from Backup**:
   ```bash
   # List available backups
   aws s3 ls s3://jpmorgan-backups/

   # Restore latest backup
   aws s3 cp s3://jpmorgan-backups/latest_backup.sql - | kubectl exec -i deployment/postgresql -- psql -U jpmorgan_user jpmorgan_financial_apis
   ```

### Communication

1. **Internal Communication**:
   - Update incident response channel
   - Notify development team
   - Escalate to management if needed

2. **External Communication**:
   - Update status page
   - Notify customers via email/SMS
   - Post updates on social media

---

**Last Updated**: November 2024
**Version**: 1.0.0
