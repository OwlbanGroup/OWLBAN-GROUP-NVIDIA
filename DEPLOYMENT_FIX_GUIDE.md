# JPMorgan Financial APIs - Deployment Fix Guide

## Overview

This guide provides step-by-step instructions to fix the critical deployment issues and get the JPMorgan Financial APIs production environment running successfully.

## Issues Identified

1. **Database Connection Error**: API container restarting due to password parsing issue
2. **Missing .env.production**: Environment configuration file not present
3. **NGINX SSL Configuration**: SSL certificates missing, causing NGINX health check failures
4. **Docker Compose Warning**: Obsolete version attribute in docker-compose.production.yml

## Solutions Implemented

### 1. Fixed Database Connection
- **Problem**: Password contained `@` symbol causing parsing errors
- **Solution**: URL-encoded the password in DATABASE_URL (`SecureP%40ssw0rd2024`)
- **Location**: `docker-compose.production.yml` line 58

### 2. Created Environment Configuration
- **File**: `.env.production.example`
- **Purpose**: Template for all required environment variables
- **Action Required**: Copy to `.env.production` and customize values

### 3. NGINX Configuration
- **Created**: `nginx/nginx.conf.no-ssl` for testing without SSL
- **Created**: `scripts/generate_ssl_certs.sh` for SSL certificate generation
- **Benefit**: Can deploy without SSL initially, add SSL later

### 4. Automated Deployment Script
- **File**: `scripts/fix_deployment.ps1`
- **Features**:
  - Automatic .env.production creation
  - NGINX configuration switching
  - Container management
  - Health check verification
  - Detailed status reporting

## Quick Start - Automated Deployment

### Prerequisites
- Docker Desktop installed and running
- PowerShell (Windows) or Bash (Linux/Mac)
- At least 4GB RAM available
- Ports 80, 443, 3000, 5432, 6379, 8000, 9090, 9093, 9100 available

### Step 1: Run the Fix Script

**Windows (PowerShell):**
```powershell
cd C:\Users\bizle\Desktop\jpmorgan_financial_apis
.\scripts\fix_deployment.ps1
```

**Linux/Mac (Bash):**
```bash
cd ~/jpmorgan_financial_apis
bash scripts/fix_deployment.sh  # Create this if needed
```

### Step 2: Monitor Deployment

The script will:
1. ✓ Check/create .env.production
2. ✓ Configure NGINX (with or without SSL)
3. ✓ Create required directories
4. ✓ Stop existing containers
5. ✓ Build and start new containers
6. ✓ Wait for services to initialize
7. ✓ Verify health checks
8. ✓ Display service URLs

### Step 3: Verify Deployment

Access these URLs to confirm everything is working:

- **API Health**: http://localhost:8000/health
- **API Root**: http://localhost:8000/
- **Dashboard**: http://localhost:8000/dashboard
- **Swagger Docs**: http://localhost:8000/swagger
- **Grafana**: http://localhost:3000 (admin/SecureGrafanaP@ss2024)
- **Prometheus**: http://localhost:9090

## Manual Deployment (Alternative)

If you prefer manual deployment or the script fails:

### Step 1: Create .env.production

```bash
cp .env.production.example .env.production
```

Edit `.env.production` and set at minimum:
```env
DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your_secure_random_string_here
LOG_LEVEL=INFO
FLASK_ENV=production
ALLOW_MISSING_TOKENS=true
```

### Step 2: Configure NGINX (No SSL)

```bash
cp nginx/nginx.conf.no-ssl nginx/nginx.conf
```

### Step 3: Create Required Directories

```bash
mkdir -p logs logs/nginx backups models nginx/ssl
```

### Step 4: Stop Existing Containers

```bash
docker-compose -f docker-compose.production.yml down
```

### Step 5: Build and Start

```bash
docker-compose -f docker-compose.production.yml up -d --build
```

### Step 6: Monitor Logs

```bash
# Watch all services
docker-compose -f docker-compose.production.yml logs -f

# Watch specific service
docker logs jpmorgan-api-prod -f
```

### Step 7: Check Status

```bash
docker-compose -f docker-compose.production.yml ps
```

All services should show "Up" and "healthy" status.

## Adding SSL/TLS (Optional)

### Option 1: Self-Signed Certificates (Testing)

**Git Bash or WSL:**
```bash
bash scripts/generate_ssl_certs.sh
```

**Manual (OpenSSL):**
```bash
cd nginx/ssl
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr -subj "/C=US/ST=NY/L=NYC/O=JPMorgan/CN=localhost"
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
```

### Option 2: Let's Encrypt (Production)

1. Install Certbot
2. Obtain certificates:
```bash
certbot certonly --standalone -d yourdomain.com
```
3. Copy certificates to `nginx/ssl/`
4. Update `nginx/nginx.conf` with proper domain

### Switch to SSL Configuration

```bash
cp nginx/nginx.conf.backup nginx/nginx.conf  # Restore SSL config
docker-compose -f docker-compose.production.yml restart nginx
```

## Troubleshooting

### API Container Keeps Restarting

**Check logs:**
```bash
docker logs jpmorgan-api-prod --tail 100
```

**Common causes:**
1. Database connection error
   - Verify DATABASE_URL in .env.production
   - Ensure password is URL-encoded: `SecureP%40ssw0rd2024`
   - Check PostgreSQL is running: `docker logs jpmorgan-postgres-prod`

2. Missing environment variables
   - Check .env.production exists
   - Verify SECRET_KEY is set

3. Port conflicts
   - Ensure port 8000 is not in use
   - Check with: `netstat -ano | findstr :8000` (Windows)

### NGINX Health Check Failing

**Check logs:**
```bash
docker logs jpmorgan-nginx-prod --tail 50
```

**Common causes:**
1. SSL certificates missing
   - Use nginx.conf.no-ssl for testing
   - Generate certificates with generate_ssl_certs.sh

2. Configuration syntax error
   - Test config: `docker exec jpmorgan-nginx-prod nginx -t`

3. Upstream not available
   - Verify API container is running
   - Check API health: `curl http://localhost:8000/health`

### Database Connection Issues

**Test connection:**
```bash
docker exec -it jpmorgan-postgres-prod psql -U jpmorgan_prod -d jpmorgan_financial_apis_prod
```

**Check if database is ready:**
```bash
docker exec jpmorgan-postgres-prod pg_isready -U jpmorgan_prod
```

**Verify password:**
- In docker-compose.production.yml: `SecureP@ssw0rd2024`
- In DATABASE_URL: `SecureP%40ssw0rd2024` (URL-encoded)

### Port Already in Use

**Find process using port (Windows):**
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Find process using port (Linux/Mac):**
```bash
lsof -i :8000
kill -9 <PID>
```

### Out of Memory

**Check Docker resources:**
- Docker Desktop → Settings → Resources
- Increase memory to at least 4GB
- Increase CPU to at least 2 cores

### Clean Start (Nuclear Option)

**⚠️ WARNING: This deletes all data!**

```bash
# Stop all containers
docker-compose -f docker-compose.production.yml down

# Remove volumes
docker volume rm jpmorgan_financial_apis_postgres_data
docker volume rm jpmorgan_financial_apis_redis_data
docker volume rm jpmorgan_financial_apis_prometheus_data
docker volume rm jpmorgan_financial_apis_grafana_data
docker volume rm jpmorgan_financial_apis_alertmanager_data

# Remove images
docker-compose -f docker-compose.production.yml down --rmi all

# Start fresh
docker-compose -f docker-compose.production.yml up -d --build
```

## Verification Checklist

After deployment, verify:

- [ ] All 8 containers are running
- [ ] PostgreSQL health check passes
- [ ] Redis health check passes  
- [ ] API responds to health check
- [ ] NGINX is accessible
- [ ] Grafana dashboard loads
- [ ] Prometheus is accessible
- [ ] Can create a test user
- [ ] Can authenticate and get token
- [ ] Can access protected endpoints

## Performance Monitoring

### Check Container Stats
```bash
docker stats
```

### View Metrics
- **Prometheus**: http://localhost:9090/targets
- **Grafana**: http://localhost:3000/dashboards

### Check Logs
```bash
# Application logs
docker logs jpmorgan-api-prod -f

# Database logs
docker logs jpmorgan-postgres-prod -f

# NGINX access logs
docker exec jpmorgan-nginx-prod tail -f /var/log/nginx/access.log
```

## Production Recommendations

Before going live:

1. **Security**
   - [ ] Change all default passwords
   - [ ] Generate strong SECRET_KEY
   - [ ] Set up proper SSL certificates
   - [ ] Configure firewall rules
   - [ ] Enable authentication on all services

2. **Monitoring**
   - [ ] Configure Grafana alerts
   - [ ] Set up log aggregation
   - [ ] Configure backup schedules
   - [ ] Set up uptime monitoring

3. **Scaling**
   - [ ] Configure load balancing
   - [ ] Set up database replication
   - [ ] Configure Redis clustering
   - [ ] Implement auto-scaling

4. **Backup**
   - [ ] Automated database backups
   - [ ] Configuration backups
   - [ ] Disaster recovery plan
   - [ ] Test restore procedures

## Support

For additional help:

1. Check logs: `docker-compose -f docker-compose.production.yml logs`
2. Review DEPLOYMENT_NEXT_STEPS.md
3. Check TODO_DEPLOYMENT_FIX.md for progress
4. Consult PRODUCTION_DEPLOYMENT_GUIDE.md

## Summary

The deployment fix addresses all critical issues:

✅ Database connection fixed with URL-encoded password  
✅ Environment configuration template created  
✅ NGINX configuration for non-SSL testing  
✅ Automated deployment script  
✅ Comprehensive troubleshooting guide  

Run `.\scripts\fix_deployment.ps1` to deploy automatically!

---

**Last Updated**: 2025-01-17  
**Version**: 1.0.0  
**Status**: Ready for Deployment
