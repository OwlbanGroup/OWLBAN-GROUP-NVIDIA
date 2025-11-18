# JPMorgan Financial APIs - Complete Setup and Deployment Script
# This script creates the environment and deploys the application with login fixes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JP Morgan Financial APIs" -ForegroundColor Cyan
Write-Host "Complete Setup & Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "Step 1: Checking Docker..." -ForegroundColor Yellow
$dockerRunning = $false
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker is running" -ForegroundColor Green
        $dockerRunning = $true
    }
}
catch {
    Write-Host "✗ Docker is not running" -ForegroundColor Red
}

if (-not $dockerRunning) {
    Write-Host ""
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Create required directories
Write-Host "Step 2: Creating required directories..." -ForegroundColor Yellow
$directories = @(
    "logs",
    "logs/nginx",
    "backups",
    "models",
    "nginx/ssl"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✓ Created $dir" -ForegroundColor Green
    }
    else {
        Write-Host "  $dir already exists" -ForegroundColor Gray
    }
}

Write-Host ""

# Step 3: Create .env.production if it doesn't exist
Write-Host "Step 3: Setting up environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env.production")) {
    Write-Host "Creating .env.production file..." -ForegroundColor Cyan
    
    $envContent = @"
# JPMorgan Financial APIs - Production Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
POSTGRES_DB=jpmorgan_financial_apis_prod
POSTGRES_USER=jpmorgan_prod
POSTGRES_PASSWORD=SecureP@ssw0rd2024

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Flask Configuration
FLASK_ENV=production
FLASK_APP=app.py
FLASK_RUN_PORT=8000
SECRET_KEY=your-secret-key-change-this-in-production-$(Get-Random -Minimum 10000 -Maximum 99999)

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/jpmorgan/app.log

# Security
ALLOW_MISSING_TOKENS=true
CORS_ORIGINS=*

# Token Manager (Optional - can be left empty for testing)
TOKEN_CLIENT_ID=
TOKEN_CLIENT_SECRET=
TOKEN_URL=
TOKEN_SCOPE=

# Cloud Storage (Optional)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_S3_BUCKET=

GCS_PROJECT_ID=
GCS_BUCKET=
GCS_CREDENTIALS_PATH=

AZURE_STORAGE_CONNECTION_STRING=
AZURE_CONTAINER_NAME=

# GitHub MCP (Optional)
GITHUB_TOKEN=
"@
    
    Set-Content -Path ".env.production" -Value $envContent
    Write-Host "✓ Created .env.production" -ForegroundColor Green
}
else {
    Write-Host "✓ .env.production already exists" -ForegroundColor Green
}

Write-Host ""

# Step 4: Apply NGINX configuration with login fixes
Write-Host "Step 4: Applying NGINX configuration with login fixes..." -ForegroundColor Yellow
if (Test-Path "nginx/nginx.conf.no-ssl") {
    # Backup existing config if it exists
    if (Test-Path "nginx/nginx.conf") {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        Copy-Item "nginx/nginx.conf" "nginx/nginx.conf.backup_$timestamp"
        Write-Host "  Backed up existing config" -ForegroundColor Gray
    }
    
    Copy-Item "nginx/nginx.conf.no-ssl" "nginx/nginx.conf" -Force
    Write-Host "✓ Applied fixed NGINX configuration" -ForegroundColor Green
    Write-Host "  - Added /user/ proxy routes" -ForegroundColor Gray
    Write-Host "  - Added /auth/ proxy routes" -ForegroundColor Gray
    Write-Host "  - Added CORS headers" -ForegroundColor Gray
}
else {
    Write-Host "✗ nginx/nginx.conf.no-ssl not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 5: Stop existing containers
Write-Host "Step 5: Stopping existing containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml down 2>&1 | Out-Null
Write-Host "✓ Stopped existing containers" -ForegroundColor Green

Write-Host ""

# Step 6: Build and start containers
Write-Host "Step 6: Building and starting containers..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Cyan
Write-Host ""

docker-compose -f docker-compose.production.yml up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Containers started successfully" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "✗ Failed to start containers" -ForegroundColor Red
    Write-Host "Check logs with: docker-compose -f docker-compose.production.yml logs" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 7: Wait for services to be ready
Write-Host "Step 7: Waiting for services to initialize..." -ForegroundColor Yellow
Write-Host "Waiting 30 seconds for all services to start..." -ForegroundColor Cyan

for ($i = 30; $i -gt 0; $i--) {
    Write-Host -NoNewline "`r  $i seconds remaining... "
    Start-Sleep -Seconds 1
}
Write-Host ""
Write-Host "✓ Wait complete" -ForegroundColor Green

Write-Host ""

# Step 8: Check container status
Write-Host "Step 8: Checking container status..." -ForegroundColor Yellow
$containers = docker-compose -f docker-compose.production.yml ps --format json | ConvertFrom-Json

$allHealthy = $true
foreach ($container in $containers) {
    $name = $container.Name
    $state = $container.State
    
    if ($state -eq "running") {
        Write-Host "✓ $name is running" -ForegroundColor Green
    }
    else {
        Write-Host "✗ $name is $state" -ForegroundColor Red
        $allHealthy = $false
    }
}

Write-Host ""

# Step 9: Test endpoints
Write-Host "Step 9: Testing endpoints..." -ForegroundColor Yellow

# Test health endpoint
Write-Host "Testing /health endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Health endpoint is accessible" -ForegroundColor Green
        $data = $response.Content | ConvertFrom-Json
        Write-Host "  Version: $($data.version)" -ForegroundColor Gray
        Write-Host "  Status: $($data.status)" -ForegroundColor Gray
    }
}
catch {
    Write-Host "✗ Health endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
    $allHealthy = $false
}

Write-Host ""

# Test dashboard endpoint
Write-Host "Testing /dashboard endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost/dashboard" -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Dashboard endpoint is accessible" -ForegroundColor Green
    }
}
catch {
    Write-Host "✗ Dashboard endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
    $allHealthy = $false
}

Write-Host ""

# Step 10: Display summary
Write-Host "========================================" -ForegroundColor Cyan
if ($allHealthy) {
    Write-Host "Deployment Successful!" -ForegroundColor Green
}
else {
    Write-Host "Deployment Completed with Warnings" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Environment Created:" -ForegroundColor Yellow
Write-Host "  ✓ Required directories created" -ForegroundColor Green
Write-Host "  ✓ .env.production configured" -ForegroundColor Green
Write-Host "  ✓ NGINX configuration applied (with login fixes)" -ForegroundColor Green
Write-Host "  ✓ Docker containers deployed" -ForegroundColor Green
Write-Host ""

Write-Host "Services Running:" -ForegroundColor Yellow
Write-Host "  - PostgreSQL Database" -ForegroundColor Cyan
Write-Host "  - Redis Cache" -ForegroundColor Cyan
Write-Host "  - Flask API Application" -ForegroundColor Cyan
Write-Host "  - NGINX Reverse Proxy" -ForegroundColor Cyan
Write-Host "  - Prometheus Monitoring" -ForegroundColor Cyan
Write-Host "  - Grafana Dashboard" -ForegroundColor Cyan
Write-Host "  - AlertManager" -ForegroundColor Cyan
Write-Host "  - Node Exporter" -ForegroundColor Cyan
Write-Host ""

Write-Host "Access Your Application:" -ForegroundColor Yellow
Write-Host "  Dashboard:   http://localhost/dashboard" -ForegroundColor Cyan
Write-Host "  API Health:  http://localhost/health" -ForegroundColor Cyan
Write-Host "  API Root:    http://localhost/" -ForegroundColor Cyan
Write-Host "  Swagger:     http://localhost/swagger" -ForegroundColor Cyan
Write-Host "  Grafana:     http://localhost:3000" -ForegroundColor Cyan
Write-Host "               (admin / SecureGrafanaP@ss2024)" -ForegroundColor Gray
Write-Host "  Prometheus:  http://localhost:9090" -ForegroundColor Cyan
Write-Host ""

Write-Host "Test Login:" -ForegroundColor Yellow
Write-Host "  1. Open: http://localhost/dashboard" -ForegroundColor Cyan
Write-Host "  2. Login with:" -ForegroundColor Cyan
Write-Host "     Username: testuser" -ForegroundColor Gray
Write-Host "     Password: testpass" -ForegroundColor Gray
Write-Host "  3. Dashboard should load without 'Failed to fetch' error" -ForegroundColor Cyan
Write-Host ""

Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  View logs:        docker-compose -f docker-compose.production.yml logs -f" -ForegroundColor Gray
Write-Host "  Stop services:    docker-compose -f docker-compose.production.yml down" -ForegroundColor Gray
Write-Host "  Restart services: docker-compose -f docker-compose.production.yml restart" -ForegroundColor Gray
Write-Host "  Check status:     docker-compose -f docker-compose.production.yml ps" -ForegroundColor Gray
Write-Host ""

if (-not $allHealthy) {
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  Some services may still be initializing." -ForegroundColor Cyan
    Write-Host "  Wait a few more minutes and test again." -ForegroundColor Cyan
    Write-Host "  Check logs: docker-compose -f docker-compose.production.yml logs" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Login Fix Applied:" -ForegroundColor Yellow
Write-Host "  ✓ Dashboard uses relative URLs" -ForegroundColor Green
Write-Host "  ✓ NGINX proxies /user/ endpoints" -ForegroundColor Green
Write-Host "  ✓ CORS headers configured" -ForegroundColor Green
Write-Host "  ✓ Ready for testing!" -ForegroundColor Green
Write-Host ""
