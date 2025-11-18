# JPMorgan Financial APIs - Simple Deployment Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JP Morgan Financial APIs - Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create directories
Write-Host "Step 1: Creating directories..." -ForegroundColor Yellow
$dirs = @("logs", "logs/nginx", "backups", "models", "nginx/ssl")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created $dir" -ForegroundColor Green
    }
}
Write-Host ""

# Step 2: Create .env.production
Write-Host "Step 2: Creating .env.production..." -ForegroundColor Yellow
if (-not (Test-Path ".env.production")) {
    $env = @"
DATABASE_URL=postgresql://jpmorgan_prod:SecureP%40ssw0rd2024@postgresql:5432/jpmorgan_financial_apis_prod
REDIS_URL=redis://redis:6379/0
FLASK_ENV=production
FLASK_APP=app.py
FLASK_RUN_PORT=8000
SECRET_KEY=change-this-secret-key-$(Get-Random)
LOG_LEVEL=INFO
ALLOW_MISSING_TOKENS=true
CORS_ORIGINS=*
"@
    Set-Content -Path ".env.production" -Value $env
    Write-Host "  Created .env.production" -ForegroundColor Green
}
Write-Host ""

# Step 3: Copy NGINX config
Write-Host "Step 3: Applying NGINX configuration..." -ForegroundColor Yellow
Copy-Item "nginx/nginx.conf.no-ssl" "nginx/nginx.conf" -Force
Write-Host "  Applied fixed NGINX config" -ForegroundColor Green
Write-Host ""

# Step 4: Deploy
Write-Host "Step 4: Deploying containers..." -ForegroundColor Yellow
Write-Host "  Stopping existing containers..." -ForegroundColor Cyan
docker-compose -f docker-compose.production.yml down 2>&1 | Out-Null

Write-Host "  Building and starting containers..." -ForegroundColor Cyan
docker-compose -f docker-compose.production.yml up -d --build

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Waiting 30 seconds for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
Write-Host ""

Write-Host "Access your application:" -ForegroundColor Yellow
Write-Host "  Dashboard:  http://localhost/dashboard" -ForegroundColor Cyan
Write-Host "  API Health: http://localhost/health" -ForegroundColor Cyan
Write-Host "  Grafana:    http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Login credentials:" -ForegroundColor Yellow
Write-Host "  Username: testuser" -ForegroundColor Cyan
Write-Host "  Password: testpass" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check status: docker-compose -f docker-compose.production.yml ps" -ForegroundColor Gray
Write-Host "View logs:    docker-compose -f docker-compose.production.yml logs -f" -ForegroundColor Gray
Write-Host ""
