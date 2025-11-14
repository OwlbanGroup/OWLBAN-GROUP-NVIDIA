# JPMorgan Financial APIs - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the JPMorgan Financial APIs to production environments. The deployment includes full infrastructure setup, monitoring, security, and scalability configurations.

## Prerequisites

### System Requirements
- **Kubernetes Cluster**: Version 1.24+ with GPU support
- **Docker**: Version 20.10+ with NVIDIA Container Toolkit
- **Helm**: Version 3.8+
- **kubectl**: Version 1.24+
- **istioctl**: Version 1.16+ (for service mesh)
- **NVIDIA GPU**: A100, V100, or RTX series with CUDA 11.8+

### Infrastructure Requirements
- **PostgreSQL**: External or managed PostgreSQL instance
- **Redis**: Redis Cluster for distributed caching
- **Load Balancer**: AWS ALB, GCP Load Balancer, or NGINX Ingress
- **Storage**: S3, GCS, or Azure Blob Storage
- **Monitoring**: Prometheus, Grafana, Elasticsearch stack

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/jpmorgan/jpmorgan-financial-apis.git
cd jpmorgan-financial-apis

# Copy and configure environment file
cp .env.production.example .env.production
# Edit .env.production with your actual values
```

### 2. Pre-deployment Checks

```bash
# Run pre-deployment validation
./deploy_production_complete.sh check
```

### 3. Full Production Deployment

```bash
# Execute complete production deployment
./deploy_production_complete.sh
```

## Detailed Deployment Steps

### Step 1: Infrastructure Setup

#### Kubernetes Cluster Preparation

```bash
# Create namespace
kubectl create namespace jpmorgan-apis

# Apply RBAC policies
kubectl apply -f k8s/rbac.yml

# Setup secrets from environment file
kubectl create secret generic jpmorgan-secrets \
  --from-env-file=.env.production \
  --namespace=jpmorgan-apis
```

#### Database Setup

```bash
# Deploy PostgreSQL with replication
kubectl apply -f k8s/database-replication.yml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n jpmorgan-apis --timeout=300s

# Run database migration
python scripts/postgresql_migration.py
```

#### Redis Cluster Setup

```bash
# Deploy Redis Cluster
kubectl apply -f k8s/redis-cluster.yml
```

### Step 2: Service Mesh Installation

```bash
# Install Istio
istioctl install --set profile=demo -y

# Apply service mesh configurations
kubectl apply -f k8s/istio-service-mesh.yml
```

### Step 3: Monitoring Stack Deployment

```bash
# Deploy Prometheus, Grafana, and Elasticsearch
./deploy_production_complete.sh monitoring

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n jpmorgan-apis
# Open http://localhost:3000 (admin/admin)
```

### Step 4: Application Deployment

```bash
# Build and push Docker images
./deploy_production_complete.sh build

# Deploy application with auto-scaling
kubectl apply -f k8s/hpa.yml

# Deploy multi-GPU configuration
kubectl apply -f k8s/multi-gpu-config.yml

# Deploy load balancer
kubectl apply -f k8s/load-balancer.yml
```

### Step 5: Post-deployment Validation

```bash
# Run comprehensive tests
python comprehensive_e2e_test.py

# Health checks
./scripts/health_check_production.sh

# Load testing
locust -f load-testing/locustfile.py --host=https://api.jpmorgan.com
```

## Configuration Files

### Environment Variables (.env.production)

```bash
# Application
FLASK_ENV=production
SECRET_KEY=your_strong_secret_key

# Database
DATABASE_TYPE=postgresql
DATABASE_HOST=postgresql
DATABASE_PASSWORD=your_db_password

# Authentication
TOKEN_CLIENT_ID=your_jpmorgan_client_id
TOKEN_CLIENT_SECRET=your_jpmorgan_client_secret

# GPU Configuration
CUDA_VISIBLE_DEVICES=all
GPU_MEMORY_FRACTION=0.8

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

### Docker Compose Override

For local testing with production-like setup:

```bash
# Start production-like environment
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

## Monitoring and Observability

### Accessing Dashboards

```bash
# Grafana
kubectl port-forward svc/grafana 3000:3000 -n jpmorgan-apis

# Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n jpmorgan-apis

# Kibana
kubectl port-forward svc/kibana 5601:5601 -n jpmorgan-apis
```

### Key Metrics to Monitor

- **API Performance**: Response times, throughput, error rates
- **GPU Utilization**: Memory usage, compute utilization, temperature
- **Database**: Connection pools, query performance, replication lag
- **Infrastructure**: CPU, memory, disk I/O, network traffic

## Scaling Configuration

### Horizontal Pod Autoscaling

```bash
# Check HPA status
kubectl get hpa -n jpmorgan-apis

# Scale manually if needed
kubectl scale deployment jpmorgan-financial-apis --replicas=5 -n jpmorgan-apis
```

### GPU Scaling

The system automatically scales GPU workers based on:
- CPU utilization > 75%
- Memory utilization > 85%
- GPU utilization > 80%

## Backup and Recovery

### Automated Backups

```bash
# Database backups run daily at 2 AM
kubectl get cronjob -n jpmorgan-apis

# Manual backup
kubectl create job --from=cronjob/database-backup manual-backup-001 -n jpmorgan-apis
```

### Disaster Recovery

```bash
# Switch to backup region
kubectl apply -f k8s/disaster-recovery.yml

# Restore from backup
./scripts/restore_from_backup.sh
```

## Security Considerations

### Network Security

- All traffic encrypted with TLS 1.3
- Mutual TLS enabled via Istio
- Network policies restrict pod-to-pod communication

### Access Control

- JWT token-based authentication
- Role-based access control (RBAC)
- API rate limiting and throttling

### Secrets Management

- Kubernetes secrets for sensitive data
- Automatic secret rotation
- Audit logging for secret access

## Troubleshooting

### Common Issues

#### Application Not Starting
```bash
# Check pod status
kubectl get pods -n jpmorgan-apis

# View logs
kubectl logs -f deployment/jpmorgan-financial-apis -n jpmorgan-apis

# Check events
kubectl get events -n jpmorgan-apis --sort-by=.metadata.creationTimestamp
```

#### GPU Issues
```bash
# Check GPU status
kubectl describe node | grep -A 10 -B 10 gpu

# Verify GPU allocation
kubectl describe pod -l component=ml-worker -n jpmorgan-apis
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/postgresql-primary -n jpmorgan-apis -- psql -U jpmorgan_user -d jpmorgan_financial_apis

# Check connection pool
kubectl logs -f deployment/jpmorgan-financial-apis -n jpmorgan-apis | grep "connection pool"
```

### Performance Tuning

#### Memory Optimization
```yaml
# Adjust memory limits in deployment
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

#### GPU Memory Management
```bash
# Adjust GPU memory fraction
export GPU_MEMORY_FRACTION=0.7
```

## Maintenance Procedures

### Rolling Updates

```bash
# Update application image
kubectl set image deployment/jpmorgan-financial-apis app=jpmorgan.azurecr.io/jpmorgan-financial-apis:v2.0.0 -n jpmorgan-apis

# Monitor rollout
kubectl rollout status deployment/jpmorgan-financial-apis -n jpmorgan-apis
```

### Log Rotation

```bash
# Check log sizes
kubectl exec -it deployment/jpmorgan-financial-apis -n jpmorgan-apis -- du -h /app/logs/

# Rotate logs manually
kubectl exec -it deployment/jpmorgan-financial-apis -n jpmorgan-apis -- logrotate /etc/logrotate.d/jpmorgan
```

## Support and Contact

### Monitoring Alerts

The system sends alerts for:
- High error rates (>5%)
- Resource exhaustion (>90% utilization)
- Service downtime
- Security incidents

### Emergency Contacts

- **Production Issues**: production-support@jpmorgan.com
- **Security Incidents**: security@jpmorgan.com
- **Infrastructure**: infra@jpmorgan.com

## Compliance and Auditing

### GDPR Compliance
- Data encryption at rest and in transit
- Automatic data deletion after retention period
- Audit logging for all data access

### SOC 2 Compliance
- Access controls and authentication
- Change management procedures
- Incident response protocols

---

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Secrets created in Kubernetes
- [ ] Database migration completed
- [ ] SSL certificates installed
- [ ] Monitoring stack deployed
- [ ] Load balancer configured
- [ ] Health checks passing
- [ ] Backup procedures tested
- [ ] Security policies applied
- [ ] Documentation updated

For additional support, refer to the [Troubleshooting Guide](docs/troubleshooting.md) or contact the platform team.
