# Check Production Docker Compose Status
# JPMorgan Financial APIs

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Production Docker Compose Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to the project directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running or not accessible" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Check if docker-compose.production.yml exists
if (-Not (Test-Path "docker-compose.production.yml")) {
    Write-Host "✗ docker-compose.production.yml not found!" -ForegroundColor Red
    Write-Host "  Current directory: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Found docker-compose.production.yml" -ForegroundColor Green
Write-Host ""

# Show production containers status
Write-Host "Production Containers Status:" -ForegroundColor Cyan
Write-Host ""
docker-compose -f docker-compose.production.yml ps

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Status Check Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Additional helpful commands
Write-Host "Helpful Commands:" -ForegroundColor Yellow
Write-Host "  View logs:    docker-compose -f docker-compose.production.yml logs" -ForegroundColor White
Write-Host "  Start all:    docker-compose -f docker-compose.production.yml up -d" -ForegroundColor White
Write-Host "  Stop all:     docker-compose -f docker-compose.production.yml down" -ForegroundColor White
Write-Host "  Restart all:  docker-compose -f docker-compose.production.yml restart" -ForegroundColor White
Write-Host ""
