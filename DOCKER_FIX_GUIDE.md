# Docker Container Fix Guide
## JPMorgan Financial APIs - Production Deployment

---

## 🔍 Problem Summary

Your Docker containers are experiencing the following issues:

1. **PostgreSQL Version Mismatch**: Data initialized with PostgreSQL 15, but container trying to use v14.19
2. **AlertManager**: Restarting due to invalid email configuration
3. **NGINX & API**: Not starting because they depend on PostgreSQL
4. **Multiple Old Containers**: Leftover containers from previous deployments

---

## 🛠️ Solution Overview

This guide provides automated scripts to:
1. **Backup all data** (volumes, configs, logs)
2. **Fix configuration issues**
3. **Clean up incompatible volumes**
4. **Restart services with correct versions**

---

## 📋 Prerequisites

- Docker Desktop running on Windows
- PowerShell 5.1 or higher
- At least 2GB free disk space for backups
- Administrator privileges (for Docker operations)

---

## 🚀 Quick Fix (Automated)

### Step 1: Run the Backup and Fix Script

```powershell
cd jpmorgan_financial_apis
.\backup_and_fix_docker.ps1
```

This script will:
- ✅ Create timestamped backup of all data
- ✅ Export container logs
- ✅ Backup configuration files
- ✅ Ask for confirmation before proceeding with fixes
- ✅ Automatically run the fix script if you confirm

### Step 2: Verify Services

After the fix completes, verify all services are running:

```powershell
docker ps --filter "name=jpmorgan-"
```

Expected output: All containers should show "Up" status

---

## 🔧 Manual Fix (Step-by-Step)

If you prefer to run each step manually:

### 1. Backup Everything

```powershell
# Run backup script only
cd jpmorgan_financial_apis
.\backup_and_fix_docker.ps1
# Choose 'no' when asked to proceed with fixes
```

### 2. Stop All Containers

```powershell
docker stop $(docker ps -a --filter "name=jpmorgan-" -q)
```

### 3. Remove Containers

```powershell
docker rm $(docker ps -a --filter "name=jpmorgan-" -q)
```

### 4. Remove Incompatible PostgreSQL Volume

```powershell
docker volume rm jpmorgan_financial_apis_postgres_data
```

### 5. Fix AlertManager Configuration

The fix script automatically updates `alertmanager.yml` with a working configuration.

### 6. Restart Services

```powershell
docker-compose -f docker-compose.production.yml up -d
```

### 7. Monitor Startup

```powershell
docker-compose -f docker-compose.production.yml logs -f
```

Press `Ctrl+C` to stop following logs.

---

## 📊 Verification Steps

### Check Container Status

```powershell
docker ps -a --filter "name=jpmorgan-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Test Service Endpoints

1. **API Health Check**:
   ```powershell
   curl http://localhost:8000/health
   ```

2. **Grafana Dashboard**:
   - URL: http://localhost:3000
   - Username: `admin`
   - Password: `SecureGrafanaP@ss2024`

3. **Prometheus**:
   - URL: http://localhost:9090

4. **PostgreSQL**:
   ```powershell
   docker exec jpmorgan-postgres-prod psql -U jpmorgan_prod -d jpmorgan_financial_apis_prod -c "SELECT version();"
   ```

### Check Container Logs

For any container that's not running properly:

```powershell
docker logs jpmorgan-<service-name>-prod --tail 100
```

---

## 🔄 Data Restoration

If you need to restore data from a backup:

### Restore PostgreSQL Database

```powershell
# Find your backup directory
cd backups
dir

# Restore from SQL dump
docker exec -i jpmorgan-postgres-prod psql -U jpmorgan_prod -d jpmorgan_financial_apis_prod < docker_backup_YYYY-MM-DD_HH-mm-ss\postgres_full_backup.sql
```

### Restore a Docker Volume

```powershell
# Example: Restore Grafana data
$backupDir = "backups\docker_backup_YYYY-MM-DD_HH-mm-ss"
docker run --rm -v jpmorgan_financial_apis_grafana_data:/target -v ${PWD}\${backupDir}:/backup alpine tar xzf /backup/jpmorgan_financial_apis_grafana_data.tar.gz -C /target
```

---

## 🐛 Troubleshooting

### Container Keeps Restarting

1. Check logs:
   ```powershell
   docker logs <container-name> --tail 100
   ```

2. Check container inspect:
   ```powershell
   docker inspect <container-name> --format='{{.State.Status}}: {{.State.Error}}'
   ```

### PostgreSQL Won't Start

1. Verify volume is removed:
   ```powershell
   docker volume ls | Select-String "postgres"
   ```

2. Check PostgreSQL logs:
   ```powershell
   docker logs jpmorgan-postgres-prod --tail 50
   ```

3. Ensure correct version in docker-compose.yml:
   ```yaml
   postgresql:
     image: postgres:15-alpine  # Should be 15, not 14
   ```

### Port Already in Use

If you get "port already allocated" errors:

1. Find what's using the port:
   ```powershell
   netstat -ano | findstr :<PORT>
   ```

2. Stop the conflicting service or change the port in docker-compose.yml

### Network Issues

Reset Docker networks:

```powershell
docker network prune -f
docker-compose -f docker-compose.production.yml up -d
```

---

## 📁 Backup Locations

All backups are stored in:
```
jpmorgan_financial_apis/backups/docker_backup_YYYY-MM-DD_HH-mm-ss/
```

Each backup contains:
- `postgres_full_backup.sql` - Full database dump
- `*.tar.gz` - Compressed volume backups
- `*_logs.txt` - Container logs
- `BACKUP_MANIFEST.txt` - Detailed backup information
- Configuration file copies

---

## 🔐 Security Notes

After fixing:

1. **Change Default Passwords**:
   - PostgreSQL: Update in `.env.production`
   - Grafana: Change via UI after first login
   - Update docker-compose.production.yml accordingly

2. **Review AlertManager Config**:
   - Update email settings in `alertmanager.yml`
   - Configure proper SMTP credentials

3. **SSL/TLS**:
   - Ensure NGINX SSL certificates are properly configured
   - Check `nginx/ssl/` directory

---

## 📞 Support

If issues persist:

1. Check all container logs:
   ```powershell
   docker-compose -f docker-compose.production.yml logs
   ```

2. Verify Docker Desktop is running properly

3. Ensure sufficient system resources (RAM, disk space)

4. Review the backup manifest for restoration options

---

## ✅ Success Checklist

- [ ] Backup completed successfully
- [ ] All containers showing "Up" status
- [ ] API health endpoint responding
- [ ] Grafana dashboard accessible
- [ ] Prometheus metrics collecting
- [ ] PostgreSQL accepting connections
- [ ] No containers in restart loop
- [ ] Logs show no critical errors

---

## 📝 Notes

- The fix process preserves all your configuration files
- Backups are timestamped and never overwritten
- You can run the backup script multiple times safely
- The PostgreSQL volume is recreated fresh with v15
- All other volumes (Redis, Grafana, etc.) are preserved

---

**Last Updated**: 2024-11-14
**Version**: 1.0
