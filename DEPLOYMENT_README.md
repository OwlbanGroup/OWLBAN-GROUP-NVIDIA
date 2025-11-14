# 🚀 JPMorgan Financial APIs - Production Deployment Guide

## Overview

This guide provides the complete production deployment procedure for the JPMorgan Financial APIs platform. The deployment is designed for enterprise-grade reliability, scalability, and security.

## Prerequisites

### System Requirements
- **Kubernetes Cluster**: 1.24+ with GPU support
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **kubectl**: 1.24+ configured for your cluster
- **Helm**: 3.8+ (optional, for Helm-based deployment)
- **Environment Variables**: All required secrets and configuration

### Required Environment Variables
```bash
# OAuth Credentials
TOKEN_CLIENT_ID=your_jpmorgan_client_id
TOKEN_CLIENT_SECRET=your_jpmorgan_client_secret

# Application Secrets
SECRET_KEY=your_256_bit_secret_key

# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/database

# Redis Configuration
REDIS_URL=redis://host:6379/0

# Optional: GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
```

## Quick Start Deployment

### 1. Prepare Environment
```bash
# Navigate to project directory
cd jpmorgan_financial_apis

# Set required environment variables
export TOKEN_CLIENT_ID="0369026e-0d67-4454-8a13-a0129a5cd3f6"
export TOKEN_CLIENT_SECRET="piAKagzhmiQFFnGbdwvDkCz0mvdC1IBGIzdYl6bLch-vegBy4HmhXNATJwLNFfmGYlWeIDH3eHTF6q0KNcJoqg"
export SECRET_KEY="your-secure-session-secret-key-256-bits"
export DATABASE_URL="postgresql://jpmorgan_user:secure_password@postgresql:5432/jpmorgan_financial_apis"
export REDIS_URL="redis://redis-cluster:6379/0"
```

### 2. Run Production Deployment
```bash
# Execute the final deployment script
./deploy_production_final.sh deploy
```

### 3. Verify Deployment
```bash
# Check deployment status
./deploy_production_final.sh status

# View application logs
./deploy_production_final.sh logs
```

## Detailed Deployment Steps

### Step 1: Pre-deployment Validation
The deployment script automatically performs:
- ✅ Environment variable validation
- ✅ Kubernetes cluster connectivity check
- ✅ Required tool availability check
- ✅ Namespace existence verification
- ✅ Docker image availability check

### Step 2: Backup Creation
- Creates timestamped backup of current state
- Exports Kubernetes resources (deployments, services, configmaps)
- Creates database backup if PostgreSQL is running
- Stores backups in `backups/backup_YYYYMMDD_HHMMSS/`

### Step 3: Compliance & Security Checks
- Runs automated compliance validation (GDPR, SOC 2, Security)
- Performs security scanning for vulnerabilities
- Validates configuration security
- Generates compliance report

### Step 4: Infrastructure Deployment
Deploys in order:
1. **PostgreSQL Database** - Primary with read replicas
2. **Redis Cluster** - Distributed caching with failover
3. **Istio Service Mesh** - Traffic management and security
4. **Monitoring Stack** - Prometheus, Grafana, AlertManager

### Step 5: Application Deployment
- Builds and pushes Docker image (if needed)
- Creates Kubernetes secrets and configmaps
- Deploys application with rolling update strategy
- Configures horizontal pod autoscaling
- Sets up health checks and readiness probes

### Step 6: Post-deployment Validation
- Waits for deployment rollout completion
- Performs health check validation
- Tests API endpoints
- Runs basic functionality tests
- Validates monitoring integration

## Deployment Configurations

### Production Environment Variables
Create a `.env.production` file:
```bash
# Application Configuration
FLASK_ENV=production
SECRET_KEY=your-256-bit-secret-here
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://jpmorgan_user:secure_password@postgresql.jpmorgan-apis.svc.cluster.local:5432/jpmorgan_financial_apis
DATABASE_SSL_MODE=require
DATABASE_CONNECTION_POOL_SIZE=20
DATABASE_CONNECTION_POOL_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://redis-cluster.jpmorgan-apis.svc.cluster.local:6379/0

# OAuth Configuration
TOKEN_CLIENT_ID=0369026e-0d67-4454-8a13-a0129a5cd3f6
TOKEN_CLIENT_SECRET=piAKagzhmiQFFnGbdwvDkCz0mvdC1IBGIzdYl6bLch-vegBy4HmhXNATJwLNFfmGYlWeIDH3eHTF6q0KNcJoqg
TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token
TOKEN_SCOPE=openid profile

# GPU Configuration (if applicable)
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.8

# Monitoring Configuration
METRICS_ENABLED=true
TELEMETRY_ENABLED=true
```

### Kubernetes Namespace Setup
```bash
# Create namespace
kubectl create namespace jpmorgan-apis

# Create service account
kubectl create serviceaccount jpmorgan-apis-sa -n jpmorgan-apis

# Apply RBAC policies
kubectl apply -f k8s/rbac.yml
```

### SSL/TLS Configuration
```yaml
# Certificate management with cert-manager
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: jpmorgan-apis-tls
  namespace: jpmorgan-apis
spec:
  secretName: jpmorgan-apis-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.jpmorgan.com
  - api.jpmorgan-finance.com
```

## Monitoring & Observability

### Accessing Monitoring Dashboards
```bash
# Port forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Access at http://localhost:3000
# Default credentials: admin/admin

# Port forward Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Access at http://localhost:9090
```

### Key Metrics to Monitor
- **Application Health**: Response times, error rates, throughput
- **Database Performance**: Connection pools, query latency, lock waits
- **Cache Efficiency**: Hit rates, memory usage, eviction rates
- **Infrastructure**: CPU, memory, disk, network utilization
- **External APIs**: JPMorgan API response times and success rates

## Troubleshooting Deployment Issues

### Common Issues

#### Deployment Stuck in Pending
```bash
# Check pod status
kubectl get pods -n jpmorgan-apis

# Check pod events
kubectl describe pod <pod-name> -n jpmorgan-apis

# Check resource availability
kubectl get nodes --show-labels
```

#### Database Connection Failures
```bash
# Test database connectivity
kubectl exec -it deployment/postgresql -n jpmorgan-apis -- psql -U jpmorgan_user -d jpmorgan_financial_apis

# Check database logs
kubectl logs -f deployment/postgresql -n jpmorgan-apis
```

#### Application Startup Failures
```bash
# Check application logs
kubectl logs -f deployment/jpmorgan-financial-apis -n jpmorgan-apis

# Check configuration
kubectl get configmap -n jpmorgan-apis
kubectl get secrets -n jpmorgan-apis
```

### Rollback Procedures

#### Automatic Rollback
If deployment fails, the script automatically initiates rollback:
```bash
# Manual rollback if needed
./deploy_production_final.sh rollback
```

#### Manual Rollback Steps
1. Rollback deployment: `kubectl rollout undo deployment/jpmorgan-financial-apis -n jpmorgan-apis`
2. Restore from backup if data corruption suspected
3. Verify application functionality
4. Update monitoring alerts

## Performance Optimization

### Resource Allocation
```yaml
# Optimized resource requests and limits
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Auto-scaling Configuration
```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: jpmorgan-apis-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: jpmorgan-financial-apis
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Security Considerations

### Network Policies
```yaml
# Restrict pod-to-pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-server-policy
spec:
  podSelector:
    matchLabels:
      app: jpmorgan-financial-apis
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
```

### Secrets Management
- All sensitive data stored in Kubernetes secrets
- Environment variables used for non-sensitive configuration
- Secrets rotated regularly via CI/CD pipeline
- Audit logging enabled for secret access

## Maintenance Procedures

### Regular Maintenance Tasks
- **Daily**: Monitor dashboards, review alerts, check logs
- **Weekly**: Update dependencies, review security scans
- **Monthly**: Performance optimization, capacity planning
- **Quarterly**: Security audits, compliance reviews

### Backup Strategy
- **Database**: Daily automated backups with 30-day retention
- **Configuration**: Version controlled in Git
- **Infrastructure**: Infrastructure as Code with Terraform
- **Logs**: Aggregated and archived for 90 days

## Support and Contact

### Emergency Contacts
- **Production Issues**: production-support@jpmorgan.com
- **Security Incidents**: security@jpmorgan.com
- **Infrastructure**: infra@jpmorgan.com
- **On-call Engineer**: +1-800-JPM-HELP

### Documentation Resources
- [API Documentation](docs/api.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Performance Tuning](docs/performance-tuning.md)
- [Security Best Practices](docs/security.md)

## Deployment Checklist

### Pre-deployment
- [ ] Environment variables configured
- [ ] Kubernetes cluster access verified
- [ ] Required tools installed
- [ ] Backup strategy confirmed
- [ ] Rollback plan documented

### During Deployment
- [ ] Pre-deployment checks passed
- [ ] Infrastructure deployed successfully
- [ ] Application deployed without errors
- [ ] Health checks passing
- [ ] Monitoring configured

### Post-deployment
- [ ] Application accessible
- [ ] API endpoints responding
- [ ] Monitoring dashboards working
- [ ] Alerting configured
- [ ] Logs being collected
- [ ] Performance baselines established

---

**Deployment Script**: `deploy_production_final.sh`
**Version**: 1.0.0
**Last Updated**: November 2024
**Supported Platforms**: Kubernetes 1.24+, Docker 20.10+
