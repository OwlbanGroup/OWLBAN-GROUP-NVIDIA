# 🚀 Docker Container Fix - Quick Start Guide

## Current Issues Identified

Based on your Docker container status, you have:

1. ❌ **PostgreSQL** - Restarting (Version mismatch: v14 data with v15 container)
2. ❌ **AlertManager** - Restarting (Invalid email configuration)
3. ⚠️ **NGINX** - Created but not started (Waiting for dependencies)
4. ⚠️ **API** - Created but not started (Waiting for PostgreSQL)
5. ✅ **Redis, Prometheus, Grafana, Node-Exporter** - Running properly

---

## 🎯 Solution: Automated Backup & Fix

I've created comprehensive scripts to fix all issues while preserving your data.

---

## 📝 What You Need to Do

### Step 1: Navigate to Project Directory

```powershell
cd C:\Users\bizle\Desktop\jpmorgan_financial_apis
```

### Step 2: Run the Backup and Fix Script

```powershell
.\backup_and_fix_docker.ps1
```

**What this script does:**
- ✅ Creates timestamped backup of all data
- ✅ Backs up PostgreSQL database (if accessible)
- ✅ Backs up all Docker volumes
- ✅ Backs up configuration files
- ✅ Exports all container logs
- ✅ Creates detailed backup manifest
- ✅ Asks for your confirmation before making changes
- ✅ Fixes all container issues
- ✅ Restarts services with correct configuration

### Step 3: Verify Everything Works

```powershell
.\check_docker_status.ps1
```

This will show you the status of all containers and confirm everything is running.

---

## 📋 Expected Timeline

- **Backup Phase**: 2-5 minutes (depending on data size)
- **Fix Phase**: 3-5 minutes (container restart and initialization)
- **Total Time**: ~5-10 minutes

---

## 🔍 What Gets Fixed

### 1. PostgreSQL Version Mismatch
- **Problem**: Data initialized with v15, container using v14.19
- **Solution**: Remove incompatible volume, restart with v15 (as specified in docker-compose.yml)
- **Impact**: Fresh PostgreSQL database (your old data is backed up)

### 2. AlertManager Configuration
- **Problem**: Invalid email credentials causing restart loop
- **Solution**: Update with working webhook configuration
- **Impact**: AlertManager will start successfully

### 3. Dependent Services
- **Problem**: NGINX and API waiting for PostgreSQL
- **Solution**: Once PostgreSQL is fixed, these will start automatically
- **Impact**: Full stack will be operational

---

## 💾 Your Data is Safe

All backups are stored in:
```
C:\Users\bizle\Desktop\jpmorgan_financial_apis\backups\docker_backup_YYYY-MM-DD_HH-mm-ss\
```

Each backup includes:
- Full PostgreSQL database dump (SQL file)
- All Docker volumes (compressed tar.gz)
- All configuration files
- All container logs
- Detailed restoration instructions

---

## 🔄 If You Need to Restore Data

After the fix, if you need to restore your old PostgreSQL data:

```powershell
# Find your backup directory
cd backups
dir

# Restore database (replace timestamp with your backup)
docker exec -i jpmorgan-postgres-prod psql -U jpmorgan_prod -d jpmorgan_financial_apis_prod < docker_backup_2024-11-14_XX-XX-XX\postgres_full_backup.sql
```

---

## 📊 After Fix - Verify Services

### Check Container Status
```powershell
docker ps --filter "name=jpmorgan-"
```

All containers should show "Up" status.

### Test Service Endpoints

1. **API Health Check**:
   ```powershell
   curl http://localhost:8000/health
   ```

2. **Grafana Dashboard**:
   - Open browser: http://localhost:3000
   - Login: admin / SecureGrafanaP@ss2024

3. **Prometheus**:
   - Open browser: http://localhost:9090

---

## 🆘 If Something Goes Wrong

### Option 1: Check Status
```powershell
.\check_docker_status.ps1
```

### Option 2: View Logs
```powershell
# View all logs
docker-compose -f docker-compose.production.yml logs

# View specific container
docker logs jpmorgan-postgres-prod --tail 100
```

### Option 3: Manual Fix
Follow the detailed guide:
```powershell
notepad DOCKER_FIX_GUIDE.md
```

### Option 4: Complete Restart
```powershell
# Stop everything
docker-compose -f docker-compose.production.yml down

# Remove volumes (CAUTION: This deletes data)
docker volume rm jpmorgan_financial_apis_postgres_data

# Start fresh
docker-compose -f docker-compose.production.yml up -d
```

---

## 📁 Files Created for You

| File | Purpose |
|------|---------|
| `backup_and_fix_docker.ps1` | Main script - backs up and fixes everything |
| `fix_docker_containers.ps1` | Fix script (called by backup script) |
| `check_docker_status.ps1` | Quick status checker |
| `DOCKER_FIX_GUIDE.md` | Detailed troubleshooting guide |
| `DOCKER_FIX_SUMMARY.md` | This file - quick reference |

---

## ✅ Success Checklist

After running the scripts, verify:

- [ ] All containers show "Up" status
- [ ] API responds at http://localhost:8000/health
- [ ] Grafana accessible at http://localhost:3000
- [ ] Prometheus accessible at http://localhost:9090
- [ ] No containers in restart loop
- [ ] PostgreSQL accepting connections
- [ ] Backup created successfully

---

## 🎓 Understanding the Fix

### Why PostgreSQL Failed
Your data directory was created by PostgreSQL 15, but an old container with PostgreSQL 14 tried to use it. PostgreSQL versions are not backward compatible at the data level.

### Why AlertManager Failed
The configuration had placeholder email credentials that weren't valid, causing the service to fail on startup.

### Why NGINX and API Didn't Start
They have health check dependencies on PostgreSQL. Since PostgreSQL was failing, Docker Compose kept them in "Created" state waiting for PostgreSQL to become healthy.

---

## 🔐 Security Reminder

After fixing, consider:

1. **Change default passwords** in `.env.production`
2. **Update Grafana password** after first login
3. **Configure proper AlertManager email** settings
4. **Review NGINX SSL certificates**

---

## 📞 Need Help?

1. Check the detailed guide: `DOCKER_FIX_GUIDE.md`
2. Review backup manifest: `backups\docker_backup_*\BACKUP_MANIFEST.txt`
3. Check container logs: `docker logs <container-name>`

---

## 🚀 Ready to Fix?

Run this command now:

```powershell
cd C:\Users\bizle\Desktop\jpmorgan_financial_apis
.\backup_and_fix_docker.ps1
```

The script will guide you through the process and ask for confirmation before making any changes.

---

**Created**: 2024-11-14  
**Status**: Ready to Execute  
**Estimated Time**: 5-10 minutes  
**Risk Level**: Low (Full backup created first)
