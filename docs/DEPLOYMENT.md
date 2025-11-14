# Deployment Runbooks - JPMorgan Financial APIs

## Overview

This document provides comprehensive deployment procedures for the JPMorgan Financial APIs across different environments.

## Pre-deployment Checklist

### Environment Preparation
- [ ] Kubernetes cluster version 1.24+ available
- [ ] PostgreSQL database provisioned and accessible
- [ ] Redis cluster configured
- [ ] Docker registry access configured
- [ ] SSL certificates obtained and installed
- [ ] DNS records configured
- [ ] Monitoring stack (Prometheus/Grafana) deployed
- [ ] Load balancer configured

### Application Configuration
- [ ] Environment variables set (TOKEN_CLIENT_ID, TOKEN_CLIENT_SECRET, etc.)
- [ ] Database connection strings configured
- [ ] Redis connection URLs set
- [ ] Secret keys generated and stored securely
- [ ] API keys and credentials validated
- [ ] Configuration files validated

### Security Validation
- [ ] Network policies applied
- [ ] RBAC permissions configured
- [ ] Security contexts set on pods
- [ ] Secrets management configured
- [ ] Audit logging enabled

## Development Environment Deployment

### Local Docker Deployment

```bash
# 1. Clone repository
git clone https://github.com/jpmorgan/jpmorgan-financial-apis.git
cd jpmorgan-financial-apis

# 2. Configure environment
cp .env.example .env
# Edit .env with development values

# 3. Start services
docker-compose up -d

# 4. Verify deployment
curl http://localhost:8000/health
```

### Local Kubernetes Deployment

```bash
# 1. Set up local cluster (Kind/Minikube)
kind create cluster --name jpmorgan-dev

# 2. Deploy dependencies
kubectl apply -f k8s/postgresql.yml
kubectl apply -f k8s/redis.yml

# 3. Wait for dependencies
kubectl wait --for=condition=ready pod -l app=postgresql
kubectl wait --for=condition=ready pod -l app=redis

# 4. Deploy application
kubectl apply -f k8s/jpmorgan-financial-apis.yml

# 5. Verify deployment
kubectl get pods
kubectl logs deployment/jpmorgan-financial-apis
```

## Staging Environment Deployment

### Automated Deployment

```bash
# 1. Run staging deployment script
./deploy_staging.sh

# 2. Monitor deployment
kubectl get pods -n staging --watch

# 3. Run smoke tests
python tests/smoke_test.py --env staging

# 4. Validate functionality
curl -H "Authorization: Bearer <token>" https://staging-api.jpmorgan.com/api/v1/accounts
```

### Manual Deployment Steps

```bash
# 1. Build and push images
docker build -t jpmorgan.azurecr.io/jpmorgan-financial-apis:staging .
docker push jpmorgan.azurecr.io/jpmorgan-financial-apis:staging

# 2. Update deployment
kubectl set image deployment/jpmorgan-financial-apis app=jpmorgan.azurecr.io/jpmorgan-financial-apis:staging -n staging

# 3. Wait for rollout
kubectl rollout status deployment/jpmorgan-financial-apis -n staging

# 4. Run integration tests
python tests/integration_test.py --env staging
```

## Production Environment Deployment

### Blue-Green Deployment Strategy

```bash
# 1. Deploy to blue environment
kubectl apply -f k8s/blue-deployment.yml

# 2. Wait for blue environment to be ready
kubectl wait --for=condition=ready pod -l app=jpmorgan-financial-apis,env=blue

# 3. Run comprehensive tests on blue
python tests/e2e_test.py --env blue

# 4. Switch traffic to blue (update ingress/service)
kubectl apply -f k8s/ingress-blue.yml

# 5. Monitor blue environment
# ... monitor for 30 minutes

# 6. If successful, decommission green
kubectl delete -f k8s/green-deployment.yml
```

### Rolling Update Deployment

```bash
# 1. Update image tag
kubectl set image deployment/jpmorgan-financial-apis app=jpmorgan.azurecr.io/jpmorgan-financial-apis:v2.1.0

# 2. Monitor rollout progress
kubectl rollout status deployment/jpmorgan-financial-apis

# 3. Check rollout history
kubectl rollout history deployment/jpmorgan-financial-apis

# 4. Verify new version
curl -H "X-Version: v2.1.0" https://api.jpmorgan.com/health
```

### Canary Deployment

```bash
# 1. Deploy canary version
kubectl apply -f k8s/canary-deployment.yml

# 2. Route 10% traffic to canary
kubectl apply -f k8s/istio-canary-routing.yml

# 3. Monitor canary metrics
# Check error rates, latency, etc.

# 4. Gradually increase traffic
kubectl apply -f k8s/istio-canary-25percent.yml
kubectl apply -f k8s/istio-canary-50percent.yml
kubectl apply -f k8s/istio-canary-100percent.yml

# 5. Promote canary to stable
kubectl apply -f k8s/stable-deployment.yml
kubectl delete -f k8s/canary-deployment.yml
```

## Multi-Region Deployment

### Primary Region Deployment

```bash
# 1. Deploy to us-east-1
kubectl apply -f k8s/multi-region-deployment.yml --context=us-east-1

# 2. Configure Route 53 latency-based routing
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://route53-primary.json

# 3. Enable cross-region replication
kubectl apply -f k8s/database-replication.yml
```

### Secondary Region Deployment

```bash
# 1. Deploy to us-west-2
kubectl apply -f k8s/multi-region-deployment.yml --context=us-west-2

# 2. Configure read replica
kubectl apply -f k8s/database-read-replica.yml

# 3. Set up cross-region load balancing
kubectl apply -f k8s/cross-region-load-balancer.yml
```

### Failover Procedures

```bash
# 1. Detect primary region failure
# Monitor health checks and alerts

# 2. Promote secondary region
kubectl apply -f k8s/failover-promote-secondary.yml --context=us-west-2

# 3. Update DNS to point to secondary
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch file://route53-failover.json

# 4. Scale up secondary region
kubectl scale deployment jpmorgan-financial-apis --replicas=10 --context=us-west-2
```

## Database Deployment and Migration

### PostgreSQL Deployment

```bash
# 1. Deploy PostgreSQL cluster
kubectl apply -f k8s/postgresql-cluster.yml

# 2. Wait for cluster readiness
kubectl wait --for=condition=ready pod -l app=postgresql

# 3. Initialize database
kubectl exec -it deployment/postgresql-primary -- psql -c "CREATE DATABASE jpmorgan_financial_apis;"

# 4. Run migrations
python scripts/postgresql_migration.py
```

### Database Backup and Restore

```bash
# 1. Create backup
kubectl exec deployment/postgresql-primary -- pg_dump -U jpmorgan_user jpmorgan_financial_apis > backup.sql

# 2. Store backup in S3
aws s3 cp backup.sql s3://jpmorgan-backups/$(date +%Y%m%d_%H%M%S)_backup.sql

# 3. Restore from backup
kubectl exec -i deployment/postgresql-primary -- psql -U jpmorgan_user jpmorgan_financial_apis < backup.sql
```

## Redis Cluster Deployment

### Single Region Redis

```bash
# 1. Deploy Redis cluster
kubectl apply -f k8s/redis-cluster.yml

# 2. Initialize cluster
kubectl apply -f k8s/redis-cluster-init.yml

# 3. Verify cluster status
kubectl exec -it redis-cluster-0 -- redis-cli cluster nodes
```

### Multi-Region Redis

```bash
# 1. Deploy Redis in primary region
kubectl apply -f k8s/redis-cluster.yml --context=us-east-1

# 2. Deploy Redis in secondary region
kubectl apply -f k8s/redis-cluster.yml --context=us-west-2

# 3. Configure cross-region replication
kubectl apply -f k8s/redis-cross-region-replication.yml
```

## Monitoring and Observability Setup

### Prometheus and Grafana

```bash
# 1. Deploy monitoring stack
kubectl apply -f monitoring/prometheus.yml
kubectl apply -f monitoring/grafana.yml

# 2. Configure service monitors
kubectl apply -f k8s/service-monitors.yml

# 3. Import dashboards
kubectl exec -it deployment/grafana -- grafana-cli admin reset-admin-password

# 4. Access Grafana
kubectl port-forward svc/grafana 3000:3000
```

### Distributed Tracing

```bash
# 1. Deploy Jaeger
kubectl apply -f monitoring/jaeger.yml

# 2. Configure application tracing
export JAEGER_AGENT_HOST=jaeger-agent.jpmorgan-apis.svc.cluster.local
export JAEGER_AGENT_PORT=6831

# 3. Access Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686
```

## Security Hardening

### Network Policies

```bash
# 1. Apply restrictive network policies
kubectl apply -f k8s/network-policies.yml

# 2. Configure service mesh security
kubectl apply -f k8s/istio-security.yml

# 3. Set up mutual TLS
kubectl apply -f k8s/istio-mtls.yml
```

### Secrets Management

```bash
# 1. Create secrets from environment
kubectl create secret generic jpmorgan-secrets \
  --from-env-file=.env.production \
  --namespace=jpmorgan-apis

# 2. Configure external secret management
kubectl apply -f k8s/external-secrets.yml

# 3. Set up secret rotation
kubectl apply -f k8s/secret-rotation.yml
```

## Rollback Procedures

### Application Rollback

```bash
# 1. Check rollout history
kubectl rollout history deployment/jpmorgan-financial-apis

# 2. Rollback to previous version
kubectl rollout undo deployment/jpmorgan-financial-apis

# 3. Verify rollback
kubectl rollout status deployment/jpmorgan-financial-apis
```

### Database Rollback

```bash
# 1. Restore from backup
kubectl exec -i deployment/postgresql -- psql -U jpmorgan_user jpmorgan_financial_apis < previous_backup.sql

# 2. Re-run migrations if needed
python scripts/postgresql_migration.py --rollback
```

### Configuration Rollback

```bash
# 1. Restore previous configmap
kubectl apply -f k8s/configmap-previous.yml

# 2. Restart affected pods
kubectl rollout restart deployment/jpmorgan-financial-apis
```

## Post-Deployment Validation

### Automated Validation

```bash
# 1. Run production validation script
./scripts/production-validation.sh

# 2. Check all health endpoints
curl https://api.jpmorgan.com/health
curl https://api.jpmorgan.com/health/database
curl https://api.jpmorgan.com/health/redis

# 3. Run API contract tests
python tests/api_contract_testing.py
```

### Manual Validation Checklist

- [ ] Application pods running and healthy
- [ ] Database connections working
- [ ] Redis cluster operational
- [ ] Load balancer routing traffic correctly
- [ ] SSL certificates valid
- [ ] Monitoring dashboards accessible
- [ ] Alerting configured and working
- [ ] Backup jobs scheduled
- [ ] Log aggregation working
- [ ] API endpoints responding correctly

## Scaling Procedures

### Horizontal Scaling

```bash
# 1. Scale application pods
kubectl scale deployment jpmorgan-financial-apis --replicas=10

# 2. Scale database read replicas
kubectl scale deployment/postgresql-read-replica --replicas=3

# 3. Scale Redis cluster
kubectl scale statefulset redis-cluster --replicas=9
```

### Vertical Scaling

```bash
# 1. Update resource requests/limits
kubectl apply -f k8s/scaled-resources.yml

# 2. Resize database instances
kubectl apply -f k8s/database-scaled.yml

# 3. Monitor performance improvements
kubectl top pods
```

## Maintenance Procedures

### Zero-Downtime Updates

```bash
# 1. Update with rolling strategy
kubectl apply -f k8s/rolling-update.yml

# 2. Monitor update progress
kubectl rollout status deployment/jpmorgan-financial-apis

# 3. Verify no service disruption
curl -f https://api.jpmorgan.com/health
```

### Scheduled Maintenance

```bash
# 1. Enable maintenance mode
kubectl set env deployment/jpmorgan-financial-apis MAINTENANCE_MODE=true

# 2. Wait for active requests to complete
sleep 300

# 3. Perform maintenance
# ... maintenance tasks ...

# 4. Disable maintenance mode
kubectl set env deployment/jpmorgan-financial-apis MAINTENANCE_MODE=false
```

---

**Last Updated**: November 2024
**Version**: 1.0.0
