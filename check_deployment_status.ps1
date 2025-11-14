# Deployment Status Checker
# Run this script to check if the deployment has completed

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs - Deployment Status" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
$dockerVersion = docker --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker is running: $dockerVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Docker is not running!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Checking deployment containers..." -ForegroundColor Yellow

# Navigate to project directory
Set-Location -Path $PSScriptRoot

# Check container status
docker-compose -f docker-compose.production.yml ps

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Checking if containers are running..." -ForegroundColor White
Write-Host "==================================" -ForegroundColor Cyan

$runningContainers = docker ps --filter "name=jpmorgan_financial_apis" --format "{{.Names}}"

if ($runningContainers) {
    Write-Host ""
    Write-Host "✓ Found running containers:" -ForegroundColor Green
    docker ps --filter "name=jpmorgan_financial_apis" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Green
    Write-Host "✓ DEPLOYMENT COMPLETE!" -ForegroundColor Green
    Write-Host "==================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access Points:" -ForegroundColor Cyan
    Write-Host "  • API:        https://localhost" -ForegroundColor White
    Write-Host "  • Health:     https://localhost/health" -ForegroundColor White
    Write-Host "  • Docs:       https://localhost/docs" -ForegroundColor White
    Write-Host "  • Grafana:    http://localhost:3000" -ForegroundColor White
    Write-Host "  • Prometheus: http://localhost:9090" -ForegroundColor White
    Write-Host ""
    Write-Host "Next Step: Run critical-path tests" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "⏳ No running containers found - build still in progress" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The Docker Compose build is still running." -ForegroundColor White
    Write-Host "This typically takes 5-10 minutes for the first build." -ForegroundColor White
    Write-Host ""
    Write-Host "Check the terminal where you ran:" -ForegroundColor White
    Write-Host "  docker-compose -f docker-compose.production.yml up -d --build" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Run this script again in a few minutes to check progress" -ForegroundColor White
Write-Host "==================================" -ForegroundColor Cyan
