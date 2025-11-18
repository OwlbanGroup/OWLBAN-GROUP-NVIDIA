# JPMorgan Financial APIs - Deployment Fix Script
# This script fixes common deployment issues and restarts the production environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs - Deployment Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent $PSScriptRoot

# Change to project directory
Set-Location $ProjectDir

# Step 1: Check if .env.production exists
Write-Host "[1/8] Checking .env.production file..." -ForegroundColor Yellow
if (-not (Test-Path ".env.production")) {
    Write-Host "  WARNING: .env.production not found. Creating from example..." -ForegroundColor Yellow
    
    if (Test-Path ".env.production.example") {
        Copy-Item ".env.production.example" ".env.production"
        Write-Host "  SUCCESS: Created .env.production from example" -ForegroundColor Green
        Write-Host "  WARNING: Please edit .env.production and set proper values!" -ForegroundColor Yellow
        Write-Host ""
        $response = Read-Host "Press Enter to continue or Ctrl+C to exit and edit the file"
    } else {
        Write-Host "  ERROR: .env.production.example not found!" -ForegroundColor Red
        Write-Host "  Creating minimal .env.production..." -ForegroundColor Yellow
        
        $envContent = @"
# Minimal Production Configuration
DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
REDIS_URL=redis://redis:6379/0
SECRET_KEY=CHANGE_THIS_TO_A_SECURE_RANDOM_STRING
LOG_LEVEL=INFO
FLASK_ENV=production
ALLOW_MISSING_TOKENS=true
"@
        $envContent | Out-File -FilePath ".env.production" -Encoding UTF8
        
        Write-Host "  SUCCESS: Created minimal .env.production" -ForegroundColor Green
    }
} else {
    Write-Host "  SUCCESS: .env.production exists" -ForegroundColor Green
}
Write-Host ""

# Step 2: Check NGINX configuration
Write-Host "[2/8] Checking NGINX configuration..." -ForegroundColor Yellow
if (-not (Test-Path "nginx/ssl/server.crt") -or -not (Test-Path "nginx/ssl/server.key")) {
    Write-Host "  WARNING: SSL certificates not found" -ForegroundColor Yellow
    Write-Host "  Using non-SSL NGINX configuration for testing..." -ForegroundColor Yellow
    
    if (Test-Path "nginx/nginx.conf.no-ssl") {
        Copy-Item "nginx/nginx.conf" "nginx/nginx.conf.backup" -Force -ErrorAction SilentlyContinue
        Copy-Item "nginx/nginx.conf.no-ssl" "nginx/nginx.conf" -Force
        Write-Host "  SUCCESS: Switched to non-SSL NGINX configuration" -ForegroundColor Green
    }
} else {
    Write-Host "  SUCCESS: SSL certificates found" -ForegroundColor Green
}
Write-Host ""

# Step 3: Create required directories
Write-Host "[3/8] Creating required directories..." -ForegroundColor Yellow
$directories = @("logs", "logs/nginx", "backups", "models", "nginx/ssl")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  SUCCESS: Created $dir" -ForegroundColor Green
    }
}
Write-Host ""

# Step 4: Stop existing containers
Write-Host "[4/8] Stopping existing containers..." -ForegroundColor Yellow
try {
    docker-compose -f docker-compose.production.yml down 2>&1 | Out-Null
    Write-Host "  SUCCESS: Containers stopped" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: No containers to stop or error occurred" -ForegroundColor Yellow
}
Write-Host ""

# Step 5: Clean up old containers and volumes (optional)
Write-Host "[5/8] Cleaning up..." -ForegroundColor Yellow
$cleanup = Read-Host "Do you want to remove old volumes? (This will delete data) [y/N]"
if ($cleanup -eq "y" -or $cleanup -eq "Y") {
    Write-Host "  Removing volumes..." -ForegroundColor Yellow
    docker volume rm jpmorgan_financial_apis_postgres_data -ErrorAction SilentlyContinue
    docker volume rm jpmorgan_financial_apis_redis_data -ErrorAction SilentlyContinue
    Write-Host "  SUCCESS: Volumes removed" -ForegroundColor Green
} else {
    Write-Host "  SKIPPED: Volume cleanup" -ForegroundColor Gray
}
Write-Host ""

# Step 6: Build and start containers
Write-Host "[6/8] Building and starting containers..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
try {
    docker-compose -f docker-compose.production.yml up -d --build
    Write-Host "  SUCCESS: Containers started" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to start containers" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 7: Wait for services to be ready
Write-Host "[7/8] Waiting for services to be ready..." -ForegroundColor Yellow
Write-Host "  Waiting 30 seconds for services to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Check service health
Write-Host "  Checking service health..." -ForegroundColor Gray
docker-compose -f docker-compose.production.yml ps
Write-Host ""

# Step 8: Test API health
Write-Host "[8/8] Testing API health..." -ForegroundColor Yellow
$maxRetries = 5
$retryCount = 0
$healthCheckPassed = $false

while ($retryCount -lt $maxRetries -and -not $healthCheckPassed) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            $healthCheckPassed = $true
            Write-Host "  SUCCESS: API health check passed!" -ForegroundColor Green
            Write-Host "  Response: $($response.Content)" -ForegroundColor Gray
        }
    } catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "  WARNING: Health check failed, retrying ($retryCount/$maxRetries)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    }
}

if (-not $healthCheckPassed) {
    Write-Host "  ERROR: API health check failed after $maxRetries attempts" -ForegroundColor Red
    Write-Host ""
    Write-Host "Checking API logs for errors..." -ForegroundColor Yellow
    docker logs jpmorgan-api-prod --tail 50
    Write-Host ""
    Write-Host "WARNING: Deployment completed but API is not responding" -ForegroundColor Yellow
    Write-Host "   Check logs with: docker logs jpmorgan-api-prod" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS: Deployment Fix Completed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Services are running at:" -ForegroundColor Cyan
    Write-Host "  API:        http://localhost:8000" -ForegroundColor White
    Write-Host "  Health:     http://localhost:8000/health" -ForegroundColor White
    Write-Host "  Dashboard:  http://localhost:8000/dashboard" -ForegroundColor White
    Write-Host "  Swagger:    http://localhost:8000/swagger" -ForegroundColor White
    Write-Host "  Grafana:    http://localhost:3000 (admin/SecureGrafanaP@ss2024)" -ForegroundColor White
    Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor White
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Cyan
    Write-Host "  View logs:    docker logs jpmorgan-api-prod -f" -ForegroundColor Gray
    Write-Host "  Check status: docker-compose -f docker-compose.production.yml ps" -ForegroundColor Gray
    Write-Host "  Stop all:     docker-compose -f docker-compose.production.yml down" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "Script completed." -ForegroundColor Cyan
