# PowerShell Commands Reference
## JPMorgan Financial APIs

## Common PowerShell Syntax Differences

### Command Chaining

**❌ WRONG (Bash/Linux syntax):**
```powershell
cd jpmorgan_financial_apis && docker-compose -f docker-compose.production.yml ps
```

**✅ CORRECT (PowerShell syntax):**

**Option 1: Use semicolon (;)**
```powershell
cd jpmorgan_financial_apis; docker-compose -f docker-compose.production.yml ps
```

**Option 2: Use separate lines**
```powershell
cd jpmorgan_financial_apis
docker-compose -f docker-compose.production.yml ps
```

**Option 3: Use the helper script (Recommended)**
```powershell
.\jpmorgan_financial_apis\check_production_status.ps1
```

## Quick Commands

### Check Production Status
```powershell
# Navigate to project directory first
cd jpmorgan_financial_apis

# Then run docker-compose
docker-compose -f docker-compose.production.yml ps
```

Or use the helper script:
```powershell
.\jpmorgan_financial_apis\check_production_status.ps1
```

### View Production Logs
```powershell
cd jpmorgan_financial_apis
docker-compose -f docker-compose.production.yml logs
```

### Start Production Services
```powershell
cd jpmorgan_financial_apis
docker-compose -f docker-compose.production.yml up -d
```

### Stop Production Services
```powershell
cd jpmorgan_financial_apis
docker-compose -f docker-compose.production.yml down
```

### Restart Production Services
```powershell
cd jpmorgan_financial_apis
docker-compose -f docker-compose.production.yml restart
```

## Available Helper Scripts

All scripts should be run from the Desktop directory:

### 1. Check Production Status
```powershell
.\jpmorgan_financial_apis\check_production_status.ps1
```
Shows the status of production Docker Compose containers.

### 2. Check Docker Status
```powershell
.\jpmorgan_financial_apis\check_docker_status.ps1
```
Shows the status of all JPMorgan Docker containers.

### 3. Check Deployment Status
```powershell
.\jpmorgan_financial_apis\check_deployment_status.ps1
```
Comprehensive deployment status check.

### 4. Quick Deploy (Windows)
```powershell
.\jpmorgan_financial_apis\quick-deploy-windows.ps1
```
Quick deployment script for Windows.

### 5. Backup and Fix Docker
```powershell
.\jpmorgan_financial_apis\backup_and_fix_docker.ps1
```
Backup current state and fix Docker issues.

### 6. Fix Docker Containers
```powershell
.\jpmorgan_financial_apis\fix_docker_containers.ps1
```
Fix problematic Docker containers.

## PowerShell Tips

### 1. Execution Policy
If you get an error about execution policy, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Running Scripts
Always use `.\` prefix when running scripts:
```powershell
.\script-name.ps1
```

### 3. Getting Help
View script help:
```powershell
Get-Help .\script-name.ps1
```

### 4. Viewing Output
Pipe to `Out-String` for better formatting:
```powershell
docker ps | Out-String
```

### 5. Error Handling
Check if a command succeeded:
```powershell
if ($?) {
    Write-Host "Success!" -ForegroundColor Green
} else {
    Write-Host "Failed!" -ForegroundColor Red
}
```

## Common Docker Commands (PowerShell)

### List All Containers
```powershell
docker ps -a
```

### List Running Containers
```powershell
docker ps
```

### View Container Logs
```powershell
docker logs <container-name>
```

### Follow Container Logs
```powershell
docker logs -f <container-name>
```

### Execute Command in Container
```powershell
docker exec -it <container-name> bash
```

### Remove Stopped Containers
```powershell
docker container prune
```

### Remove All Containers (Careful!)
```powershell
docker rm -f $(docker ps -aq)
```

## Docker Compose Commands (PowerShell)

### Start Services
```powershell
docker-compose -f docker-compose.production.yml up -d
```

### Stop Services
```powershell
docker-compose -f docker-compose.production.yml down
```

### View Logs
```powershell
docker-compose -f docker-compose.production.yml logs
```

### Follow Logs
```powershell
docker-compose -f docker-compose.production.yml logs -f
```

### Restart Services
```powershell
docker-compose -f docker-compose.production.yml restart
```

### Rebuild and Start
```powershell
docker-compose -f docker-compose.production.yml up -d --build
```

### View Service Status
```powershell
docker-compose -f docker-compose.production.yml ps
```

## Troubleshooting

### Issue: "&&" is not a valid statement separator
**Solution:** Use `;` instead of `&&` or run commands on separate lines.

### Issue: Script execution is disabled
**Solution:** Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Issue: Docker is not running
**Solution:** Start Docker Desktop and wait for it to fully initialize.

### Issue: Permission denied
**Solution:** Run PowerShell as Administrator (right-click → Run as Administrator)

## Quick Reference Card

| Task | Command |
|------|---------|
| Change directory | `cd path\to\directory` |
| Run script | `.\script-name.ps1` |
| Chain commands | `command1; command2` |
| Check Docker status | `docker ps` |
| View logs | `docker logs container-name` |
| Start compose | `docker-compose up -d` |
| Stop compose | `docker-compose down` |

---

**Note:** Always run commands from the correct directory or use full paths to avoid errors.
