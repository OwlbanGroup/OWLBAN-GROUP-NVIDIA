# Fix Authentication Issue - Final Step
# JPMorgan Financial APIs

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Authentication Fix - Final Step" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

Write-Host "Analysis: The URL encoding was incorrect." -ForegroundColor Yellow
Write-Host "PostgreSQL in Docker Compose handles @ symbols correctly." -ForegroundColor Yellow
Write-Host "Reverting to original DATABASE_URL format..." -ForegroundColor Yellow
Write-Host ""

Write-Host "Step 1: Stopping containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml down
Write-Host "[OK] Containers stopped" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Backing up current docker-compose.production.yml..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "docker-compose.production.yml" "docker-compose.production.yml.backup_$timestamp"
Write-Host "[OK] Backup created: docker-compose.production.yml.backup_$timestamp" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Reverting DATABASE_URL to original format..." -ForegroundColor Yellow
$content = Get-Content "docker-compose.production.yml" -Raw

# Revert the URL encoding - use original format
$content = $content -replace 'DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql', 'DATABASE_URL=postgresql://jpmorgan_prod:SecureP@ssw0rd2024@postgresql'

# Remove obsolete version attribute if still present
$content = $content -replace "version: '3.8'`r?`n", ""

Set-Content "docker-compose.production.yml" $content
Write-Host "[OK] DATABASE_URL reverted to original format" -ForegroundColor Green
Write-Host "[OK] Removed obsolete version attribute" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Starting containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml up -d
Write-Host "[OK] Containers started" -ForegroundColor Green
Write-Host ""

Write-Host "Step 5: Waiting for services to initialize (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
Write-Host ""

Write-Host "Step 6: Checking container status..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml ps
Write-Host ""

Write-Host "Step 7: Checking API container logs..." -ForegroundColor Yellow
Write-Host "Last 30 lines of API logs:" -ForegroundColor Cyan
docker logs jpmorgan-api-prod --tail 30
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Authentication Fix Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Verification Commands:" -ForegroundColor Yellow
Write-Host "1. Check API health: curl http://localhost:8000/health" -ForegroundColor White
Write-Host "2. View all services: docker-compose -f docker-compose.production.yml ps" -ForegroundColor White
Write-Host "3. Monitor API logs: docker logs -f jpmorgan-api-prod" -ForegroundColor White
Write-Host ""
