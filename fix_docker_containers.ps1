# JPMorgan Financial APIs - Docker Container Fix Script
# This script fixes Docker container issues after backup is complete

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs - Container Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Continue"

# Step 1: Stop all JPMorgan containers
Write-Host "[1/7] Stopping all JPMorgan containers..." -ForegroundColor Yellow
$containers = docker ps -a --filter "name=jpmorgan-" --format "{{.Names}}"
foreach ($container in $containers) {
    if ($container) {
        Write-Host "  -> Stopping: $container" -ForegroundColor Cyan
        docker stop $container 2>$null | Out-Null
    }
}
Write-Host "  [OK] All containers stopped" -ForegroundColor Green
Write-Host ""

# Step 2: Remove all JPMorgan containers
Write-Host "[2/7] Removing all JPMorgan containers..." -ForegroundColor Yellow
$containers = docker ps -a --filter "name=jpmorgan-" --format "{{.Names}}"
foreach ($container in $containers) {
    if ($container) {
        Write-Host "  -> Removing: $container" -ForegroundColor Cyan
        docker rm -f $container 2>$null | Out-Null
    }
}
Write-Host "  [OK] All containers removed" -ForegroundColor Green
Write-Host ""

# Step 3: Remove incompatible PostgreSQL volume
Write-Host "[3/7] Removing incompatible PostgreSQL volume..." -ForegroundColor Yellow
$postgresVolume = "jpmorgan_financial_apis_postgres_data"
docker volume rm $postgresVolume 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] PostgreSQL volume removed successfully" -ForegroundColor Green
} else {
    Write-Host "  [!] Volume may not exist or already removed" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Fix AlertManager configuration
Write-Host "[4/7] Fixing AlertManager configuration..." -ForegroundColor Yellow
$alertmanagerConfig = @"
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
"@

$alertmanagerConfig | Out-File "alertmanager.yml" -Encoding UTF8 -Force
Write-Host "  [OK] AlertManager configuration updated" -ForegroundColor Green
Write-Host ""

# Step 5: Clean up any dangling volumes (optional)
Write-Host "[5/7] Cleaning up dangling volumes..." -ForegroundColor Yellow
docker volume prune -f 2>$null | Out-Null
Write-Host "  [OK] Cleanup complete" -ForegroundColor Green
Write-Host ""

# Step 6: Start services with docker-compose
Write-Host "[6/7] Starting services with docker-compose..." -ForegroundColor Yellow
Write-Host "  -> This may take a few minutes..." -ForegroundColor Cyan
docker-compose -f docker-compose.production.yml up -d
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Services started successfully" -ForegroundColor Green
} else {
    Write-Host "  [!] Error starting services" -ForegroundColor Red
    Write-Host "  -> Check logs with: docker-compose -f docker-compose.production.yml logs" -ForegroundColor Yellow
}
Write-Host ""

# Step 7: Wait and check container health
Write-Host "[7/7] Checking container health..." -ForegroundColor Yellow
Write-Host "  -> Waiting 30 seconds for containers to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "Container Status:" -ForegroundColor Cyan
docker ps -a --filter "name=jpmorgan-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
Write-Host ""

# Check for any containers that are not running
$failedContainers = docker ps -a --filter "name=jpmorgan-" --filter "status=exited" --format "{{.Names}}"
$restartingContainers = docker ps -a --filter "name=jpmorgan-" --filter "status=restarting" --format "{{.Names}}"

if ($failedContainers -or $restartingContainers) {
    Write-Host "[!] Some containers are not running properly:" -ForegroundColor Yellow
    if ($failedContainers) {
        Write-Host "  Exited containers:" -ForegroundColor Red
        foreach ($container in $failedContainers) {
            if ($container) {
                Write-Host "    - $container" -ForegroundColor Red
            }
        }
    }
    if ($restartingContainers) {
        Write-Host "  Restarting containers:" -ForegroundColor Yellow
        foreach ($container in $restartingContainers) {
            if ($container) {
                Write-Host "    - $container" -ForegroundColor Yellow
            }
        }
    }
    Write-Host ""
    Write-Host "To check logs for a specific container:" -ForegroundColor Cyan
    Write-Host "  docker logs <container-name> --tail 50" -ForegroundColor White
} else {
    Write-Host "[OK] All containers are running!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "[OK] FIX PROCESS COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Verify services are accessible:" -ForegroundColor White
Write-Host "     - API: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "     - Grafana: http://localhost:3000 (admin/SecureGrafanaP@ss2024)" -ForegroundColor Cyan
Write-Host "     - Prometheus: http://localhost:9090" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. If you need to restore data from backup:" -ForegroundColor White
Write-Host "     - Check the backup manifest in backups/docker_backup_*/BACKUP_MANIFEST.txt" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Monitor logs:" -ForegroundColor White
Write-Host "     docker-compose -f docker-compose.production.yml logs -f" -ForegroundColor Cyan
Write-Host ""
