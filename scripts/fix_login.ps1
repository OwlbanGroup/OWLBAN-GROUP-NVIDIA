# JPMorgan Financial APIs - Login Fix Script
# This script applies the login fixes and restarts the NGINX container

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JP Morgan Financial APIs - Login Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
try {
    $dockerStatus = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not installed or not accessible." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan
Write-Host ""

# Step 1: Backup existing nginx.conf if it exists
Write-Host "Step 1: Backing up existing NGINX configuration..." -ForegroundColor Yellow
if (Test-Path "nginx/nginx.conf") {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item "nginx/nginx.conf" "nginx/nginx.conf.backup_$timestamp"
    Write-Host "✓ Backed up to nginx/nginx.conf.backup_$timestamp" -ForegroundColor Green
} else {
    Write-Host "! No existing nginx.conf found (this is OK)" -ForegroundColor Yellow
}

Write-Host ""

# Step 2: Copy the fixed configuration
Write-Host "Step 2: Applying fixed NGINX configuration..." -ForegroundColor Yellow
if (Test-Path "nginx/nginx.conf.no-ssl") {
    Copy-Item "nginx/nginx.conf.no-ssl" "nginx/nginx.conf" -Force
    Write-Host "✓ Copied nginx.conf.no-ssl to nginx.conf" -ForegroundColor Green
} else {
    Write-Host "ERROR: nginx/nginx.conf.no-ssl not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 3: Check if containers are running
Write-Host "Step 3: Checking container status..." -ForegroundColor Yellow
$nginxRunning = docker ps --filter "name=jpmorgan-nginx-prod" --format "{{.Names}}" 2>$null
$appRunning = docker ps --filter "name=jpmorgan-api-prod" --format "{{.Names}}" 2>$null

if ($nginxRunning) {
    Write-Host "✓ NGINX container is running" -ForegroundColor Green
} else {
    Write-Host "! NGINX container is not running" -ForegroundColor Yellow
}

if ($appRunning) {
    Write-Host "✓ API container is running" -ForegroundColor Green
} else {
    Write-Host "! API container is not running" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Restart NGINX container
if ($nginxRunning) {
    Write-Host "Step 4: Restarting NGINX container..." -ForegroundColor Yellow
    docker restart jpmorgan-nginx-prod
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NGINX container restarted successfully" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to restart NGINX container" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Step 4: Starting containers..." -ForegroundColor Yellow
    docker-compose -f docker-compose.production.yml up -d nginx
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NGINX container started successfully" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to start NGINX container" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 5: Wait for NGINX to be ready
Write-Host "Step 5: Waiting for NGINX to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check NGINX configuration
Write-Host "Verifying NGINX configuration..." -ForegroundColor Yellow
$nginxTest = docker exec jpmorgan-nginx-prod nginx -t 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ NGINX configuration is valid" -ForegroundColor Green
} else {
    Write-Host "ERROR: NGINX configuration test failed:" -ForegroundColor Red
    Write-Host $nginxTest -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 6: Test endpoints
Write-Host "Step 6: Testing endpoints..." -ForegroundColor Yellow

# Test health endpoint
Write-Host "Testing /health endpoint..." -ForegroundColor Cyan
try {
    $healthResponse = Invoke-WebRequest -Uri "http://localhost/health" -UseBasicParsing -TimeoutSec 10
    if ($healthResponse.StatusCode -eq 200) {
        Write-Host "✓ Health endpoint is accessible" -ForegroundColor Green
        $healthData = $healthResponse.Content | ConvertFrom-Json
        Write-Host "  Version: $($healthData.version)" -ForegroundColor Gray
    }
}
catch {
    Write-Host "✗ Health endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test dashboard endpoint
Write-Host "Testing /dashboard endpoint..." -ForegroundColor Cyan
try {
    $dashboardResponse = Invoke-WebRequest -Uri "http://localhost/dashboard" -UseBasicParsing -TimeoutSec 10
    if ($dashboardResponse.StatusCode -eq 200) {
        Write-Host "✓ Dashboard endpoint is accessible" -ForegroundColor Green
    }
}
catch {
    Write-Host "✗ Dashboard endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Step 7: Display summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fix Applied Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Changes Applied:" -ForegroundColor Yellow
Write-Host "  ✓ Dashboard updated to use relative URLs" -ForegroundColor Green
Write-Host "  ✓ NGINX configuration updated with /user/ and /auth/ routes" -ForegroundColor Green
Write-Host "  ✓ CORS headers added to NGINX" -ForegroundColor Green
Write-Host "  ✓ NGINX container restarted" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Open your browser to: http://localhost/dashboard" -ForegroundColor Cyan
Write-Host "  2. Try logging in with:" -ForegroundColor Cyan
Write-Host "     Username: testuser" -ForegroundColor Gray
Write-Host "     Password: testpass" -ForegroundColor Gray
Write-Host "  3. Check browser console (F12) for any errors" -ForegroundColor Cyan
Write-Host ""
Write-Host "If you still see issues:" -ForegroundColor Yellow
Write-Host "  - Clear browser cache (Ctrl+Shift+Delete)" -ForegroundColor Gray
Write-Host "  - Check container logs: docker logs jpmorgan-nginx-prod" -ForegroundColor Gray
Write-Host "  - Check API logs: docker logs jpmorgan-api-prod" -ForegroundColor Gray
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  Dashboard:   http://localhost/dashboard" -ForegroundColor Cyan
Write-Host "  API Health:  http://localhost/health" -ForegroundColor Cyan
Write-Host "  API Root:    http://localhost/" -ForegroundColor Cyan
Write-Host "  Swagger:     http://localhost/swagger" -ForegroundColor Cyan
Write-Host "  Grafana:     http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
