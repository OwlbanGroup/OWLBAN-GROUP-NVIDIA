# JPMorgan Financial APIs - Windows Deployment Guide

## 🪟 Quick Start for Windows Users

This guide is specifically for deploying the JPMorgan Financial APIs on Windows with Docker Desktop.

---

## 📋 Prerequisites

### Required Software

1. **Docker Desktop for Windows**
   - Download: https://www.docker.com/products/docker-desktop
   - Minimum version: 4.0+
   - Ensure WSL 2 backend is enabled

2. **Windows 10/11**
   - Version 2004 or higher (Build 19041 or higher)
   - WSL 2 enabled

3. **PowerShell 5.1+** (comes with Windows)

---

## 🚀 One-Command Deployment

### Step 1: Ensure Docker Desktop is Running

**Option A: Manual Start**
1. Open Docker Desktop from Start Menu
2. Wait for Docker to show "Docker Desktop is running" in system tray
3. Verify by opening PowerShell and running: `docker version`

**Option B: Automatic Start (using our script)**
The deployment script will automatically start Docker Desktop if it's not running.

### Step 2: Run the Deployment Script

Open PowerShell in the project directory and run:

```powershell
cd jpmorgan_financial_apis
.\quick-deploy-windows.ps1
```

That's it! The script will:
- ✅ Check and start Docker Desktop if needed
- ✅ Create necessary directories
- ✅ Generate SSL certificates
- ✅ Build and start all services
- ✅ Run health checks
- ✅ Open the health endpoint in your browser

---

## 🔧 Manual Deployment (Alternative)

If you prefer to run commands manually:

### 1. Start Docker Desktop

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

Wait 30-60 seconds for Docker to be ready.

### 2. Verify Docker is Running

```powershell
docker version
docker-compose version
```

### 3. Navigate to Project Directory

```powershell
cd C:\Users\YourUsername\Desktop\jpmorgan_financial_apis
```

### 4. Create Directories

```powershell
New-Item -ItemType Directory -Path logs, backups, nginx\ssl, models -Force
```

### 5. Configure Environment

```powershell
# Copy example environment file
Copy-Item .env.production.example .env.production

# Edit with your values
notepad .env.production
```

### 6. Generate SSL Certificates

```powershell
python scripts/setup_https.py --action generate-self-signed --domain localhost --cert-dir nginx/ssl --days 365
```

### 7. Deploy with Docker Compose

```powershell
docker-compose -f docker-compose.production.yml up -d --build
```

### 8. Check Status

```powershell
# View running containers
docker-compose -f docker-compose.production.yml ps

# Check logs
docker-compose -f docker-compose.production.yml logs -f app

# Test health endpoint
Invoke-WebRequest -Uri https://localhost/health -SkipCertificateCheck
```

---

## 🐛 Troubleshooting

### Docker Desktop Won't Start

**Problem:** Docker Desktop fails to start or shows errors

**Solutions:**
1. Check if Hyper-V is enabled:
   ```powershell
   Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V
   ```

2. Enable WSL 2:
   ```powershell
   wsl --install
   wsl --set-default-version 2
   ```

3. Restart your computer after enabling Hyper-V or WSL 2

4. Check Docker Desktop settings:
   - Open Docker Desktop
   - Go to Settings → General
   - Ensure "Use the WSL 2 based engine" is checked

### "docker: command not found" Error

**Problem:** PowerShell can't find Docker commands

**Solutions:**
1. Restart PowerShell after installing Docker Desktop
2. Verify Docker Desktop is running (check system tray)
3. Add Docker to PATH manually:
   ```powershell
   $env:Path += ";C:\Program Files\Docker\Docker\resources\bin"
   ```

### Port Already in Use

**Problem:** Error about ports 80, 443, 5432, etc. already in use

**Solutions:**
1. Check what's using the port:
   ```powershell
   netstat -ano | findstr :80
   netstat -ano | findstr :443
   ```

2. Stop the conflicting service or change ports in `docker-compose.production.yml`

3. Common conflicts:
   - Port 80/443: IIS, Apache, other web servers
   - Port 5432: Local PostgreSQL installation
   - Port 6379: Local Redis installation

### SSL Certificate Errors

**Problem:** Browser shows SSL certificate warnings

**Solution:** This is expected with self-signed certificates. For development:
- Click "Advanced" → "Proceed to localhost (unsafe)"
- Or use `-SkipCertificateCheck` in PowerShell commands

For production, replace with CA-signed certificates.

### Container Won't Start

**Problem:** One or more containers fail to start

**Solutions:**
1. Check logs:
   ```powershell
   docker-compose -f docker-compose.production.yml logs [service-name]
   ```

2. Check container status:
   ```powershell
   docker-compose -f docker-compose.production.yml ps
   ```

3. Restart specific service:
   ```powershell
   docker-compose -f docker-compose.production.yml restart [service-name]
   ```

4. Rebuild and restart:
   ```powershell
   docker-compose -f docker-compose.production.yml up -d --build [service-name]
   ```

### WSL 2 Issues

**Problem:** Docker Desktop requires WSL 2 but it's not working

**Solutions:**
1. Update WSL:
   ```powershell
   wsl --update
   ```

2. Set WSL 2 as default:
   ```powershell
   wsl --set-default-version 2
   ```

3. Install a Linux distribution:
   ```powershell
   wsl --install -d Ubuntu
   ```

4. Restart Docker Desktop after WSL changes

---

## 📊 Accessing Services

Once deployed, access these URLs in your browser:

| Service | URL | Credentials |
|---------|-----|-------------|
| API | https://localhost | N/A |
| Health Check | https://localhost/health | N/A |
| API Documentation | https://localhost/docs | N/A |
| Grafana Dashboard | http://localhost:3000 | admin / SecureGrafanaP@ss2024 |
| Prometheus | http://localhost:9090 | N/A |

**Note:** You'll see SSL warnings for `https://localhost` - this is normal with self-signed certificates.

---

## 🔄 Common Operations

### View Logs

```powershell
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker-compose -f docker-compose.production.yml logs -f app

# Last 100 lines
docker-compose -f docker-compose.production.yml logs --tail=100 app
```

### Restart Services

```powershell
# Restart all
docker-compose -f docker-compose.production.yml restart

# Restart specific service
docker-compose -f docker-compose.production.yml restart app
```

### Stop Services

```powershell
# Stop all (keeps data)
docker-compose -f docker-compose.production.yml stop

# Stop and remove containers (keeps data)
docker-compose -f docker-compose.production.yml down

# Stop and remove everything including volumes (DELETES DATA!)
docker-compose -f docker-compose.production.yml down -v
```

### Update Application

```powershell
# Pull latest changes (if using git)
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.production.yml up -d --build app
```

### Check Resource Usage

```powershell
# View resource usage
docker stats

# View disk usage
docker system df
```

### Clean Up

```powershell
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

---

## 🔒 Security Notes for Windows

1. **Firewall Configuration**
   - Windows Firewall may prompt for Docker
   - Allow Docker Desktop through firewall
   - For production, configure Windows Firewall rules

2. **Antivirus Exclusions**
   - Add Docker directories to antivirus exclusions:
     - `C:\Program Files\Docker`
     - `C:\ProgramData\Docker`
     - Your project directory

3. **File Permissions**
   - Ensure your user has permissions to the project directory
   - Run PowerShell as Administrator if needed

---

## 💡 Tips for Windows Users

1. **Use PowerShell, not CMD**
   - PowerShell has better Docker support
   - Use Windows Terminal for better experience

2. **Enable Developer Mode**
   - Settings → Update & Security → For developers
   - Enable "Developer Mode"

3. **Allocate Resources to Docker**
   - Docker Desktop → Settings → Resources
   - Increase CPU and Memory if needed
   - Recommended: 4 CPUs, 8GB RAM

4. **Use WSL 2 Backend**
   - Much faster than Hyper-V backend
   - Better file system performance

5. **Keep Docker Desktop Updated**
   - Check for updates regularly
   - Docker Desktop → Check for updates

---

## 📞 Getting Help

### Check Logs First

```powershell
# Application logs
docker-compose -f docker-compose.production.yml logs app

# All services logs
docker-compose -f docker-compose.production.yml logs
```

### Verify Docker Status

```powershell
# Docker version
docker version

# Docker info
docker info

# Container status
docker-compose -f docker-compose.production.yml ps
```

### Common Commands Reference

```powershell
# Start deployment
.\quick-deploy-windows.ps1

# Stop all services
docker-compose -f docker-compose.production.yml down

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Restart service
docker-compose -f docker-compose.production.yml restart app

# Check health
Invoke-WebRequest -Uri https://localhost/health -SkipCertificateCheck

# Open in browser
Start-Process https://localhost/docs
```

---

## ✅ Pre-Deployment Checklist

Before deploying, ensure:

- [ ] Docker Desktop is installed and running
- [ ] WSL 2 is enabled (for Docker Desktop)
- [ ] PowerShell 5.1+ is available
- [ ] Project directory is accessible
- [ ] Ports 80, 443, 5432, 6379, 3000, 9090 are available
- [ ] `.env.production` is configured with your values
- [ ] Antivirus exclusions are set (if applicable)
- [ ] Windows Firewall allows Docker

---

## 🎉 Success!

Once deployed successfully, you should see:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                  🎉 DEPLOYMENT SUCCESSFUL! 🎉                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

Your JPMorgan Financial APIs are now running!

**Next Steps:**
1. Test the API: https://localhost/health
2. View documentation: https://localhost/docs
3. Check monitoring: http://localhost:3000
4. Review logs for any warnings
5. Update production secrets in `.env.production`

---

**Need more help?** Check the main [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) for detailed information.
