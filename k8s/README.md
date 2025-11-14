# Kubernetes Deployment for JPMorgan Financial APIs

This directory contains Kubernetes manifests for deploying the JPMorgan Financial APIs application in a production environment.

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to access the cluster
- Docker registry access
- NGINX Ingress Controller installed
- cert-manager for SSL certificates (optional)

## Components

- **api**: Main Flask application with 3 replicas
- **postgres**: PostgreSQL database for persistent storage
- **redis**: Redis cache for session management
- **ingress**: NGINX ingress for external access with SSL

## Deployment Steps

1. **Build and push Docker image:**
   ```bash
   docker build -t jpmorgan/financial-apis:latest .
   docker push jpmorgan/financial-apis:latest
   ```

2. **Update secrets with real values:**
   Edit `secret.yaml` and replace base64 encoded values with actual secrets.

3. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   kubectl apply -f postgres-deployment.yaml
   kubectl apply -f redis-deployment.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f ingress.yaml
   ```

4. **Verify deployment:**
   ```bash
   kubectl get pods
   kubectl get services
   kubectl get ingress
   ```

## Configuration

### Environment Variables

Key environment variables are managed through ConfigMap and Secret:

- **ConfigMap**: Non-sensitive configuration (log levels, URLs, etc.)
- **Secret**: Sensitive data (passwords, API keys, tokens)

### Database

- PostgreSQL is used for persistent storage of telemetry data
- Connection pooling is configured for optimal performance
- Persistent volume claims ensure data persistence

### Redis

- Used for caching and session management
- Password-protected for security
- Persistent storage for data durability

## Scaling

The API deployment is configured with 3 replicas by default. Scale as needed:

```bash
kubectl scale deployment jpmorgan-api --replicas=5
```

## Monitoring

- Health checks are configured for all services
- Readiness and liveness probes ensure high availability
- Resource limits prevent resource exhaustion

## Security

- Secrets are stored in Kubernetes secrets
- SSL/TLS termination at ingress level
- Network policies should be applied for additional security

## Troubleshooting

1. **Check pod status:**
   ```bash
   kubectl describe pod <pod-name>
   ```

2. **View logs:**
   ```bash
   kubectl logs <pod-name>
   ```

3. **Check service endpoints:**
   ```bash
   kubectl get endpoints
   ```

## Backup and Recovery

- Database PVCs provide persistence
- Regular backups should be configured for PostgreSQL
- Redis persistence is enabled with AOF

## Next Steps

- Implement Helm charts for easier deployment
- Add service mesh (Istio) for advanced traffic management
- Set up monitoring stack (Prometheus, Grafana)
- Configure horizontal pod autoscaling
