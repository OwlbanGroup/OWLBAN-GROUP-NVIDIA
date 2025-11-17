# Fix Database Connection Issue
# JPMorgan Financial APIs

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Database Connection Fix Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check if docker-compose.production.yml exists
if (-Not (Test-Path "docker-compose.production.yml")) {
    Write-Host "[ERROR] docker-compose.production.yml not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Stopping containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml down
Write-Host "[OK] Containers stopped" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Backing up docker-compose.production.yml..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "docker-compose.production.yml" "docker-compose.production.yml.backup_$timestamp"
Write-Host "[OK] Backup created: docker-compose.production.yml.backup_$timestamp" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Fixing DATABASE_URL..." -ForegroundColor Yellow
$content = Get-Content "docker-compose.production.yml" -Raw

# URL-encode the @ symbol in the password
$content = $content -replace 'DATABASE_URL=postgresql://jpmorgan_prod:SecureP@ssw0rd2024@postgresql', 'DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql'

# Remove obsolete version attribute
$content = $content -replace "version: '3.8'`n", ""

Set-Content "docker-compose.production.yml" $content
Write-Host "[OK] DATABASE_URL fixed (@ symbol URL-encoded)" -ForegroundColor Green
Write-Host "[OK] Removed obsolete version attribute" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Rebuilding and starting containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml up -d --build
Write-Host "[OK] Containers started" -ForegroundColor Green
Write-Host ""

Write-Host "Step 5: Waiting for services to initialize (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
Write-Host ""

Write-Host "Step 6: Checking container status..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml ps
Write-Host ""

Write-Host "Step 7: Checking API container logs..." -ForegroundColor Yellow
Write-Host "Last 20 lines of API logs:" -ForegroundColor Cyan
docker logs jpmorgan-api-prod --tail 20
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fix Applied Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Monitor API logs: docker logs -f jpmorgan-api-prod" -ForegroundColor White
Write-Host "2. Check health: curl http://localhost:8000/health" -ForegroundColor White
Write-Host "3. View all services: docker-compose -f docker-compose.production.yml ps" -ForegroundColor White
Write-Host ""

Write-Host "If the API is still restarting, wait 1-2 minutes for initialization." -ForegroundColor Yellow
Write-Host "Then check logs again with: docker logs jpmorgan-api-prod" -ForegroundColor Yellow
Write-Host ""
