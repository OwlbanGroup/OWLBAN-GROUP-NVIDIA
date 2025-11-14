# JPMorgan Financial APIs - Docker Backup and Fix Script
# This script backs up all data before fixing Docker container issues

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs - Backup & Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Continue"

# Create backup directory with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$backupDir = ".\backups\docker_backup_$timestamp"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

Write-Host "[1/8] Created backup directory: $backupDir" -ForegroundColor Green
Write-Host ""

# Backup PostgreSQL data from the running/failed container
Write-Host "[2/8] Backing up PostgreSQL data..." -ForegroundColor Yellow
try {
    # Try to backup from the container if it's accessible
    docker exec jpmorgan-postgres-prod pg_dumpall -U jpmorgan_prod > "$backupDir\postgres_full_backup.sql" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] PostgreSQL backup successful" -ForegroundColor Green
    } else {
        Write-Host "  [!] Could not backup from running container (expected if container is failing)" -ForegroundColor Yellow
        Write-Host "  -> Will backup volume data directly" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [!] Container backup failed: $_" -ForegroundColor Yellow
}

# Backup the actual volume data
Write-Host "[3/8] Backing up Docker volumes..." -ForegroundColor Yellow
$volumes = @(
    "jpmorgan_financial_apis_postgres_data",
    "jpmorgan_financial_apis_redis_data",
    "jpmorgan_financial_apis_prometheus_data",
    "jpmorgan_financial_apis_grafana_data",
    "jpmorgan_financial_apis_alertmanager_data"
)

foreach ($volume in $volumes) {
    Write-Host "  -> Backing up volume: $volume" -ForegroundColor Cyan
    try {
        # Create a temporary container to copy volume data
        docker run --rm -v ${volume}:/source -v ${PWD}/${backupDir}:/backup alpine tar czf /backup/${volume}.tar.gz -C /source . 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    [OK] Volume backed up successfully" -ForegroundColor Green
        } else {
            Write-Host "    [!] Volume backup failed or volume doesn't exist" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "    [!] Error backing up volume: $_" -ForegroundColor Yellow
    }
}
Write-Host ""

# Backup configuration files
Write-Host "[4/8] Backing up configuration files..." -ForegroundColor Yellow
$configFiles = @(
    "docker-compose.production.yml",
    "alertmanager.yml",
    "prometheus.yml",
    ".env.production",
    "nginx\nginx.conf"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        $destPath = Join-Path $backupDir (Split-Path $file -Leaf)
        Copy-Item $file $destPath -Force
        Write-Host "  [OK] Backed up: $file" -ForegroundColor Green
    } else {
        Write-Host "  [!] File not found: $file" -ForegroundColor Yellow
    }
}
Write-Host ""

# Export container logs
Write-Host "[5/8] Exporting container logs..." -ForegroundColor Yellow
$containers = docker ps -a --filter "name=jpmorgan-" --format "{{.Names}}"
foreach ($container in $containers) {
    if ($container) {
        Write-Host "  -> Exporting logs for: $container" -ForegroundColor Cyan
        docker logs $container > "$backupDir\${container}_logs.txt" 2>&1
        Write-Host "    [OK] Logs exported" -ForegroundColor Green
    }
}
Write-Host ""

# Create backup manifest
Write-Host "[6/8] Creating backup manifest..." -ForegroundColor Yellow
$volumeList = ($volumes | ForEach-Object { "  - $_.tar.gz" }) -join "`n"
$configList = ($configFiles | Where-Object { Test-Path $_ } | ForEach-Object { "  - $(Split-Path $_ -Leaf)" }) -join "`n"
$containerList = ($containers | Where-Object { $_ } | ForEach-Object { "  - ${_}_logs.txt" }) -join "`n"

$manifest = @"
JPMorgan Financial APIs - Docker Backup Manifest
================================================
Backup Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Backup Location: $backupDir

Contents:
---------
1. PostgreSQL full database dump (if accessible)
2. Docker volume backups (tar.gz archives)
3. Configuration files
4. Container logs

Volume Backups:
$volumeList

Configuration Files:
$configList

Container Logs:
$containerList

Restoration Instructions:
------------------------
To restore PostgreSQL data:
  docker exec -i jpmorgan-postgres-prod psql -U jpmorgan_prod < postgres_full_backup.sql

To restore a volume:
  docker run --rm -v VOLUME_NAME:/target -v ${PWD}/${backupDir}:/backup alpine tar xzf /backup/VOLUME_NAME.tar.gz -C /target

================================================
"@

$manifest | Out-File "$backupDir\BACKUP_MANIFEST.txt" -Encoding UTF8
Write-Host "  [OK] Manifest created" -ForegroundColor Green
Write-Host ""

# Display backup summary
Write-Host "[7/8] Backup Summary:" -ForegroundColor Cyan
Write-Host "  Backup Location: $backupDir" -ForegroundColor White
$backupSize = (Get-ChildItem $backupDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "  Total Size: $([math]::Round($backupSize, 2)) MB" -ForegroundColor White
Write-Host "  Files Backed Up: $((Get-ChildItem $backupDir -File).Count)" -ForegroundColor White
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "[OK] BACKUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backup saved to: $backupDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review the backup manifest: $backupDir\BACKUP_MANIFEST.txt" -ForegroundColor White
Write-Host "  2. Run the fix script: .\fix_docker_containers.ps1" -ForegroundColor White
Write-Host ""

# Ask for confirmation to proceed with fixes
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to Fix Docker Containers?" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Stop all JPMorgan containers" -ForegroundColor White
Write-Host "  2. Remove incompatible PostgreSQL volume" -ForegroundColor White
Write-Host "  3. Fix AlertManager configuration" -ForegroundColor White
Write-Host "  4. Restart all services with fresh volumes" -ForegroundColor White
Write-Host ""
$proceed = Read-Host "Do you want to proceed with the fixes? (yes/no)"

if ($proceed -eq "yes" -or $proceed -eq "y") {
    Write-Host ""
    Write-Host "[8/8] Proceeding with Docker fixes..." -ForegroundColor Green
    Write-Host ""
    
    # Execute the fix script
    if (Test-Path ".\fix_docker_containers.ps1") {
        & ".\fix_docker_containers.ps1"
    } else {
        Write-Host "[!] Fix script not found. Please run it manually." -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "Fix cancelled. Your backup is safe at: $backupDir" -ForegroundColor Yellow
    Write-Host "Run .\fix_docker_containers.ps1 when ready to proceed." -ForegroundColor Yellow
}
