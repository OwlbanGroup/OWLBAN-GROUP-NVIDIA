# Live Production Deployment Guide

**Date:** December 5, 2025  
**Version:** 1.0.0  
**Status:** Ready for Deployment

---

## 📋 OVERVIEW

This guide provides step-by-step instructions for deploying the JPMorgan Financial APIs to a live production environment.

---

## ✅ PRE-DEPLOYMENT CHECKLIST

### 1. System Requirements
- [ ] Docker installed (version 20.10+)
- [ ] Docker Compose installed (version 2.0+)
- [ ] Python 3.12+ installed
- [ ] Git installed
- [ ] Minimum 4GB RAM available
- [ ] Minimum 20GB disk space available
- [ ] Ports available: 80, 443, 3000, 5432, 6379, 8000, 9090, 9093, 9100

### 2. Security Requirements
- [ ] SSL/TLS certificates obtained (for HTTPS)
- [ ] Firewall rules configured
- [ ] VPN access configured (if required)
- [ ] Secrets management system ready
- [ ] Backup system configured

### 3. Configuration Requirements
- [ ] `.env.production` file created with actual credentials
- [ ] Database credentials secured
- [ ] API keys obtained (JPMorgan, etc.)
- [ ] Domain name configured
- [ ] DNS records updated

### 4. Backup Requirements
- [ ] Backup storage location configured
- [ ] Backup retention policy defined
- [ ] Disaster recovery plan documented
- [ ] Rollback procedure tested

---

## 🚀 DEPLOYMENT OPTIONS

### Option 1: Automated Deployment (Recommended)

**Quick Start:**
```powershell
# Run automated deployment
.\DEPLOY_TO_LIVE_PRODUCTION.ps1

# With options
.\DEPLOY_TO_LIVE_PRODUCTION.ps1 -Environment production -SkipTests

# Dry run (test without deploying)
.\DEPLOY_TO_LIVE_PRODUCTION.ps1 -DryRun
```

**What it does:**
1. ✅ Checks prerequisites
2. ✅ Creates backup of current deployment
3. ✅ Stops existing services
4. ✅ Updates configuration
5. ✅ Runs pre-deployment tests
6. ✅ Deploys services
7. ✅ Runs post-deployment tests
8. ✅ Provides deployment summary

---

### Option 2: Manual Deployment

#### Step 1: Prepare Environment
```powershell
# Navigate to project directory
cd C:\Users\bizle\Desktop\jpmorgan_financial_apis

# Create backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "backups\deployment_$timestamp"

# Backup configuration
Copy-Item .env.production "backups\deployment_$timestamp\"
Copy-Item docker-compose.production.yml "backups\deployment_$timestamp\"

# Backup database
docker exec jpmorgan-postgres-prod pg_dump -U jpmorgan_prod jpmorgan_financial_apis_prod > "backups\deployment_$timestamp\database.sql"
```

#### Step 2: Update Configuration
```powershell
# Edit .env.production with your credentials
notepad .env.production

# Required variables:
# - SECRET_KEY (generate new: New-Guid)
# - DATABASE_PASSWORD (secure password)
# - TOKEN_CLIENT_ID (JPMorgan API)
# - TOKEN_CLIENT_SECRET (JPMorgan API)
# - JPMORGAN_API_KEY (JPMorgan API)
```

#### Step 3: Stop Existing Services
```powershell
# Stop all services
docker-compose -f docker-compose.production.yml stop

# Verify stopped
docker ps
```

#### Step 4: Deploy Services
```powershell
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Build and start services
docker-compose -f docker-compose.production.yml up -d --build

# Wait for services to start
Start-Sleep -Seconds 30

# Check service status
docker-compose -f docker-compose.production.yml ps
```

#### Step 5: Verify Deployment
```powershell
# Run verification script
.\FINAL_PRODUCTION_VERIFICATION.ps1

# Or manual verification
curl http://localhost:8000/health
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health
```

---

## 🔧 CONFIGURATION

### Environment Variables (.env.production)

```bash
# Environment
FLASK_ENV=production
FLASK_DEBUG=0
TESTING=0

# Security
SECRET_KEY=<generate-new-guid>
TOKEN_CLIENT_ID=<your-jpmorgan-client-id>
TOKEN_CLIENT_SECRET=<your-jpmorgan-client-secret>

# Database
DATABASE_URL=postgresql://jpmorgan_prod:<password>@postgres:5432/jpmorgan_financial_apis_prod
DATABASE_PASSWORD=<secure-password>

# Redis
REDIS_URL=redis://redis:6379/0

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Audit Logging
AUDIT_LOG_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=90

# API Settings
API_BASE_URL=https://api.yourdomain.com
ALLOWED_ORIGINS=https://yourdomain.com

# JPMorgan API
JPMORGAN_ENVIRONMENT=production
JPMORGAN_API_KEY=<your-api-key>
```

### Generate Secure Secrets

```powershell
# Generate SECRET_KEY
$secretKey = New-Guid
Write-Host "SECRET_KEY=$secretKey"

# Generate DATABASE_PASSWORD
$dbPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
Write-Host "DATABASE_PASSWORD=$dbPassword"
```

---

## 🔒 SECURITY HARDENING

### 1. Change Default Passwords

```powershell
# Grafana admin password
docker exec jpmorgan-grafana-prod grafana-cli admin reset-admin-password <NewSecurePassword>

# PostgreSQL password
docker exec jpmorgan-postgres-prod psql -U jpmorgan_prod -c "ALTER USER jpmorgan_prod WITH PASSWORD '<NewSecurePassword>';"
```

### 2. Configure Firewall

```powershell
# Allow only necessary ports
# HTTP/HTTPS
New-NetFirewallRule -DisplayName "JPMorgan API HTTP" -Direction Inbound -LocalPort 80 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "JPMorgan API HTTPS" -Direction Inbound -LocalPort 443 -Protocol TCP -Action Allow

# Monitoring (restrict to internal network)
New-NetFirewallRule -DisplayName "Grafana" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow -RemoteAddress 192.168.0.0/16
New-NetFirewallRule -DisplayName "Prometheus" -Direction Inbound -LocalPort 9090 -Protocol TCP -Action Allow -RemoteAddress 192.168.0.0/16
```

### 3. Enable HTTPS

```powershell
# Generate self-signed certificate (for testing)
python scripts/setup_https.py --action generate-self-signed --domain yourdomain.com

# Or use Let's Encrypt (for production)
# Install certbot and run:
# certbot certonly --standalone -d yourdomain.com
```

### 4. Configure Rate Limiting

Edit `config.py`:
```python
# Production rate limits
RATE_LIMIT_PER_DAY = 10000
RATE_LIMIT_PER_HOUR = 1000
RATE_LIMIT_PER_MINUTE = 100
```

---

## 📊 MONITORING SETUP

### 1. Access Grafana

```
URL: http://localhost:3000
Default Username: admin
Default Password: admin (change immediately!)
```

### 2. Import Dashboards

```powershell
# Import Prometheus dashboard
.\import_prometheus_dashboard.ps1

# Or manually:
# 1. Login to Grafana
# 2. Go to Dashboards > Import
# 3. Upload prometheus_dashboard.json
```

### 3. Configure Alerts

```powershell
# Edit alerts.yml
notepad alerts.yml

# Restart AlertManager
docker-compose -f docker-compose.production.yml restart alertmanager
```

---

## 🧪 POST-DEPLOYMENT TESTING

### 1. Run Verification Script

```powershell
.\FINAL_PRODUCTION_VERIFICATION.ps1
```

### 2. Run Security Tests

```powershell
.\RUN_SECURITY_TESTS.ps1
```

### 3. Manual Testing

```powershell
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/

# Test authentication
$headers = @{"Authorization"="Bearer test_token"}
Invoke-WebRequest -Uri http://localhost:8000/user/profile -Headers $headers

# Test audit logging
Invoke-WebRequest -Uri http://localhost:8000/audit/logs -Headers $headers
```

### 4. Load Testing

```powershell
# Install Apache Bench (if not installed)
# choco install apache-httpd

# Run load test
ab -n 1000 -c 10 http://localhost:8000/health

# Expected results:
# - Requests per second: > 100
# - Time per request: < 100ms
# - Failed requests: 0
```

---

## 🔄 ROLLBACK PROCEDURE

### If Deployment Fails:

```powershell
# 1. Stop new deployment
docker-compose -f docker-compose.production.yml down

# 2. Restore configuration
$backupDir = "backups\deployment_<timestamp>"
Copy-Item "$backupDir\.env.production" .env.production -Force
Copy-Item "$backupDir\docker-compose.production.yml" docker-compose.production.yml -Force

# 3. Restore database
docker exec -i jpmorgan-postgres-prod psql -U jpmorgan_prod jpmorgan_financial_apis_prod < "$backupDir\database.sql"

# 4. Restart services
docker-compose -f docker-compose.production.yml up -d

# 5. Verify rollback
.\FINAL_PRODUCTION_VERIFICATION.ps1
```

---

## 📝 MAINTENANCE

### Daily Tasks
- [ ] Check Grafana dashboards
- [ ] Review Prometheus alerts
- [ ] Check application logs
- [ ] Verify backup completion
- [ ] Monitor resource usage

### Weekly Tasks
- [ ] Review security logs
- [ ] Analyze performance metrics
- [ ] Update documentation
- [ ] Test disaster recovery
- [ ] Review audit logs

### Monthly Tasks
- [ ] Update dependencies
- [ ] Security audit
- [ ] Capacity planning review
- [ ] Backup restoration test
- [ ] Team training

---

## 🆘 TROUBLESHOOTING

### Services Won't Start

```powershell
# Check logs
docker-compose -f docker-compose.production.yml logs -f

# Check specific service
docker logs jpmorgan-api-prod

# Restart specific service
docker-compose -f docker-compose.production.yml restart app
```

### Database Connection Issues

```powershell
# Check PostgreSQL status
docker exec jpmorgan-postgres-prod pg_isready -U jpmorgan_prod

# Check database logs
docker logs jpmorgan-postgres-prod

# Connect to database
docker exec -it jpmorgan-postgres-prod psql -U jpmorgan_prod jpmorgan_financial_apis_prod
```

### High Memory Usage

```powershell
# Check resource usage
docker stats

# Restart services
docker-compose -f docker-compose.production.yml restart

# Clear Docker cache
docker system prune -a
```

---

## 📞 SUPPORT

### Documentation
- Deployment Guide: `LIVE_PRODUCTION_DEPLOYMENT_GUIDE.md`
- Security Guide: `SECURITY_AND_E2E_IMPLEMENTATION_COMPLETE.md`
- Test Results: `FINAL_SECURITY_AND_E2E_TEST_RESULTS.md`
- Next Steps: `NEXT_STEPS_ROADMAP.md`

### Monitoring
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- API Docs: http://localhost:8000/api/docs

### Logs
- Application: `docker logs jpmorgan-api-prod`
- Database: `docker logs jpmorgan-postgres-prod`
- Deployment: `logs/deployment_<timestamp>.log`

---

## ✅ DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] Prerequisites checked
- [ ] Configuration updated
- [ ] Secrets secured
- [ ] Backup created
- [ ] Tests passed

### Deployment
- [ ] Services stopped
- [ ] Images pulled
- [ ] Services started
- [ ] Health checks passed
- [ ] Monitoring active

### Post-Deployment
- [ ] Verification tests passed
- [ ] Security tests passed
- [ ] Performance acceptable
- [ ] Monitoring configured
- [ ] Documentation updated

### Security
- [ ] Passwords changed
- [ ] Firewall configured
- [ ] HTTPS enabled
- [ ] Rate limiting active
- [ ] Audit logging enabled

---

**Deployment Guide Version:** 1.0.0  
**Last Updated:** December 5, 2025  
**Status:** Ready for Production Deployment

🚀 **Ready to deploy to live production!** 🚀
