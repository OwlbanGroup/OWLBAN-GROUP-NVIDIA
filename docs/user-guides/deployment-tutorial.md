# Deployment Tutorial - JPMorgan Financial APIs

This tutorial provides step-by-step instructions for deploying the JPMorgan Financial APIs to various environments, from local development to production Kubernetes clusters.

## Prerequisites

Before deploying, ensure you have:

- Docker and Docker Compose installed
- Kubernetes cluster (for production deployments)
- Helm 3.x (for Helm-based deployments)
- Valid JPMorgan API credentials
- PostgreSQL and Redis instances
- kubectl configured for your cluster

## Local Development Deployment

### Using Docker Compose

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jpmorgan/jpmorgan-financial-apis.git
   cd jpmorgan-financial-apis
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your JPMorgan credentials and database settings
   ```

3. **Start the services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify deployment**:
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

### Manual Local Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up databases**:
   ```bash
   # PostgreSQL
   createdb jpmorgan_financial_apis

   # Redis (if not using Docker)
   redis-server
   ```

3. **Run the application**:
   ```bash
   python app_final.py
   ```

## Production Deployment

### Kubernetes Deployment

1. **Prepare your cluster**:
   ```bash
   # Create namespace
   kubectl create namespace jpmorgan-apis

   # Set context
   kubectl config set-context --current --namespace=jpmorgan-apis
   ```

2. **Deploy databases**:
   ```bash
   # PostgreSQL
   kubectl apply -f k8s/postgresql/

   # Redis Cluster
   kubectl apply -f k8s/redis/
   ```

3. **Deploy the application**:
   ```bash
   # Using the complete deployment script
   ./deploy_production_complete.sh

   # Or apply manifests individually
   kubectl apply -f k8s/deployment.yml
   kubectl apply -f k8s/service.yml
   kubectl apply -f k8s/ingress.yml
   ```

4. **Configure Istio (optional)**:
   ```bash
   # Install Istio
   istioctl install --set profile=demo -y

   # Enable injection
   kubectl label namespace jpmorgan-apis istio-injection=enabled

   # Apply Istio configurations
   kubectl apply -f k8s/istio/
   ```

### Helm Deployment

1. **Add the Helm repository**:
   ```bash
   helm repo add jpmorgan https://charts.jpmorgan.com
   helm repo update
   ```

2. **Install the chart**:
   ```bash
   helm install jpmorgan-apis jpmorgan/financial-apis \
     --namespace jpmorgan-apis \
     --create-namespace \
     --set image.tag=v2.0.0 \
     --set postgresql.enabled=true \
     --set redis.enabled=true
   ```

3. **Upgrade deployment**:
   ```bash
   helm upgrade jpmorgan-apis jpmorgan/financial-apis \
     --set image.tag=v2.1.0
   ```

## Multi-Region Deployment

### Primary Region Setup

```bash
# Deploy to primary region
kubectl apply -f k8s/multi-region-deployment.yml --context=primary-us-east

# Verify deployment
kubectl get pods --context=primary-us-east
```

### Secondary Region Setup

```bash
# Deploy to secondary region
kubectl apply -f k8s/multi-region-deployment.yml --context=secondary-us-west

# Configure cross-region replication
kubectl apply -f k8s/replication.yml
```

### Global Load Balancing

```bash
# Deploy global load balancer
kubectl apply -f k8s/global-lb.yml

# Configure DNS
# Point your domain to the global load balancer IP
```

## GPU-Enabled Deployment

For GPU workloads:

1. **Ensure GPU nodes**:
   ```bash
   kubectl get nodes -l accelerator=nvidia-gpu
   ```

2. **Deploy with GPU support**:
   ```bash
   kubectl apply -f k8s/gpu-deployment.yml
   ```

3. **Monitor GPU usage**:
   ```bash
   kubectl logs -f deployment/jpmorgan-apis-gpu | grep gpu
   ```

## Monitoring and Observability

### Prometheus and Grafana Setup

1. **Deploy monitoring stack**:
   ```bash
   kubectl apply -f k8s/monitoring/
   ```

2. **Access dashboards**:
   ```bash
   # Port forward Grafana
   kubectl port-forward svc/grafana 3000:3000

   # Open http://localhost:3000
   # Default credentials: admin/admin
   ```

### Distributed Tracing

```bash
# Deploy Jaeger
kubectl apply -f k8s/jaeger/

# Access tracing UI
kubectl port-forward svc/jaeger-query 16686:16686
```

## Scaling Configuration

### Horizontal Pod Autoscaler

```bash
# Apply HPA
kubectl apply -f k8s/hpa.yml

# Check scaling
kubectl get hpa
```

### Custom Metrics Autoscaling

```bash
# Deploy metrics server
kubectl apply -f k8s/metrics-server/

# Configure custom metrics
kubectl apply -f k8s/custom-metrics/
```

## Backup and Recovery

### Database Backup

```bash
# Create backup job
kubectl apply -f k8s/backup-job.yml

# Manual backup
kubectl exec -it deployment/postgresql -- pg_dump -U jpmorgan_user jpmorgan_financial_apis > backup.sql
```

### Application Backup

```bash
# Backup configurations
kubectl get configmap,secret -o yaml > config-backup.yml

# Backup persistent volumes
kubectl apply -f k8s/volume-backup.yml
```

## Troubleshooting Deployment Issues

### Common Issues

#### Pod Startup Failures
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name> --previous
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/postgresql -- psql -U jpmorgan_user -d jpmorgan_financial_apis -c "SELECT 1;"

# Check connection pool
kubectl logs deployment/jpmorgan-apis | grep "connection"
```

#### Service Mesh Issues
```bash
# Check Istio proxy status
kubectl exec -it <pod-name> -c istio-proxy -- pilot-agent request GET server_info

# Check service mesh configuration
istioctl proxy-status
```

### Health Checks

```bash
# Application health
curl https://api.jpmorgan.com/health

# Kubernetes health
kubectl get componentstatuses

# Database health
kubectl exec -it deployment/postgresql -- pg_isready -U jpmorgan_user -d jpmorgan_financial_apis
```

## Security Considerations

### Certificate Management

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create certificate
kubectl apply -f k8s/certificates.yml
```

### Network Policies

```bash
# Apply network policies
kubectl apply -f k8s/network-policies.yml
```

## Performance Optimization

### Resource Limits

```yaml
# deployment.yml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Database Optimization

```bash
# Enable connection pooling
kubectl apply -f k8s/pgbouncer/

# Configure read replicas
kubectl apply -f k8s/postgresql-replicas/
```

## Rollback Procedures

### Application Rollback

```bash
# Rollback deployment
kubectl rollout undo deployment/jpmorgan-financial-apis

# Rollback to specific version
kubectl rollout undo deployment/jpmorgan-financial-apis --to-revision=2
```

### Database Rollback

```bash
# Restore from backup
kubectl apply -f k8s/restore-job.yml
```

## Next Steps

- Review the [Production Readiness Checklist](../production-readiness.md)
- Set up [Monitoring Dashboards](../monitoring.md)
- Configure [Incident Response](../incident-response.md) procedures
- Review [Capacity Planning](../capacity-planning.md) guidelines

---

**Last Updated**: November 2024
**Version**: 1.0.0
