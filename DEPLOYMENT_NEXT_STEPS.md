# Deployment Next Steps
## JPMorgan Financial APIs - Production Deployment

## Current Status

### ✅ Working Services (7/8)
- PostgreSQL Database (Healthy)
- Redis Cache (Healthy)
- Prometheus Monitoring (Healthy)
- Grafana Dashboard (Healthy)
- Node Exporter (Running)
- AlertManager (Running)
- NGINX Reverse Proxy (Running but Unhealthy)

### ⚠️ Critical Issue
- **API Container**: Restarting continuously due to database connection error

## Root Cause Analysis

The API container is failing with this error:
```
psycopg2.OperationalError: could not translate host name "ssw0rd2024@postgresql" to address
```

**Problem**: The DATABASE_URL contains a password with special characters (`@` symbol in `SecureP@ssw0rd2024`), which is causing psycopg2 to incorrectly parse the connection string.

**Current DATABASE_URL**:
```
postgresql://jpmorgan_prod:SecureP@ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
```

The `@` in the password is being interpreted as the username/password separator, causing the parser to think `ssw0rd2024@postgresql` is the hostname.

## Immediate Fix Required

### Option 1: URL-Encode the Password (Recommended)
Replace `@` with `%40` in the password:

```yaml
environment:
  - DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
```

### Option 2: Use Individual Connection Parameters
Instead of DATABASE_URL, use separate environment variables:

```yaml
environment:
  - DB_HOST=postgresql
  - DB_PORT=5432
  - DB_NAME=jpmorgan_financial_apis_prod
  - DB_USER=jpmorgan_prod
  - DB_PASSWORD=SecureP@ssw0rd2024
```

Then update `telemetry_handler.py` to construct the DSN from these parameters.

### Option 3: Change the Password (Quick Fix)
Change the password to one without special characters:

```yaml
POSTGRES_PASSWORD: SecurePassword2024
DATABASE_URL: postgresql://jpmorgan_prod:SecurePassword2024@postgresql:5432/jpmorgan_financial_apis_prod
```

## Step-by-Step Fix Instructions

### Using PowerShell (Windows)

**Step 1: Stop the containers**
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml down
```

**Step 2: Edit docker-compose.production.yml**
Update the DATABASE_URL in the `app` service section (line 54):

```yaml
environment:
  - DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
```

**Step 3: Rebuild and restart**
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml up -d --build
```

**Step 4: Monitor the API container**
```powershell
cd jpmorgan_financial_apis; docker logs -f jpmorgan-api-prod
```

**Step 5: Verify all services are healthy**
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml ps
```

## Additional Issues to Address

### 1. NGINX Health Check Failing
The NGINX container is running but marked as unhealthy. This might be due to:
- Missing SSL certificates
- Configuration syntax errors
- Port binding issues

**Check NGINX logs:**
```powershell
cd jpmorgan_financial_apis; docker logs jpmorgan-nginx-prod
```

### 2. Docker Compose Version Warning
Remove the obsolete `version` attribute from docker-compose.production.yml (line 1).

### 3. Environment File
Ensure `.env.production` exists and contains necessary environment variables.

## Verification Steps

After applying the fix, verify:

1. **API Health Check**
```powershell
curl http://localhost:8000/health
```

2. **Database Connection**
```powershell
cd jpmorgan_financial_apis; docker exec -it jpmorgan-api-prod python -c "import psycopg2; conn = psycopg2.connect('postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod'); print('Connected successfully'); conn.close()"
```

3. **All Services Status**
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml ps
```

4. **Access Monitoring Dashboards**
- Grafana: http://localhost:3000 (admin/SecureGrafanaP@ss2024)
- Prometheus: http://localhost:9090
- AlertManager: http://localhost:9093

## Post-Deployment Tasks

1. **Set up SSL/TLS certificates** for NGINX
2. **Configure domain names** and DNS
3. **Set up automated backups** for PostgreSQL
4. **Configure log rotation** for application logs
5. **Set up monitoring alerts** in AlertManager
6. **Perform load testing** to verify performance
7. **Document API endpoints** and usage
8. **Set up CI/CD pipeline** for automated deployments

## Rollback Plan

If issues persist:

```powershell
# Stop all containers
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml down

# Remove volumes (WARNING: This deletes data)
docker volume rm jpmorgan_financial_apis_postgres_data
docker volume rm jpmorgan_financial_apis_redis_data

# Restart from scratch
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml up -d
```

## Support Commands

### View all container logs
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml logs
```

### View specific container logs
```powershell
cd jpmorgan_financial_apis; docker logs jpmorgan-api-prod
cd jpmorgan_financial_apis; docker logs jpmorgan-nginx-prod
cd jpmorgan_financial_apis; docker logs jpmorgan-postgres-prod
```

### Restart specific service
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml restart app
```

### Execute command in container
```powershell
cd jpmorgan_financial_apis; docker exec -it jpmorgan-api-prod /bin/sh
```

### Check database
```powershell
cd jpmorgan_financial_apis; docker exec -it jpmorgan-postgres-prod psql -U jpmorgan_prod -d jpmorgan_financial_apis_prod
```

## Contact & Escalation

If issues persist after applying these fixes:
1. Check application logs in `./logs` directory
2. Review Docker container logs
3. Verify network connectivity between containers
4. Check resource usage (CPU, memory, disk)
5. Consult the PRODUCTION_DEPLOYMENT_GUIDE.md for detailed troubleshooting

---

**Last Updated**: 2025-01-17
**Status**: Critical Fix Required
**Priority**: HIGH
