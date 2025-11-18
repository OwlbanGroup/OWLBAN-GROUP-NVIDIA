# Wait for Docker Desktop to be ready and start production services
Write-Host "Waiting for Docker Desktop to be ready..." -ForegroundColor Yellow

$maxAttempts = 30
$attempt = 0
$dockerReady = $false

while (-not $dockerReady -and $attempt -lt $maxAttempts) {
    $attempt++
    Write-Host "Attempt $attempt of $maxAttempts..." -ForegroundColor Cyan
    
    try {
        docker ps 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $dockerReady = $true
            Write-Host "Docker Desktop is ready!" -ForegroundColor Green
        } else {
            Write-Host "Docker not ready yet. Waiting 10 seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        }
    } catch {
        Write-Host "Docker not ready yet. Waiting 10 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}

if (-not $dockerReady) {
    Write-Host "Docker Desktop failed to start after $maxAttempts attempts." -ForegroundColor Red
    Write-Host "Please check Docker Desktop manually and ensure it's running." -ForegroundColor Red
    exit 1
}

# Docker is ready, now start production services
Write-Host "`nStarting production services..." -ForegroundColor Green
docker-compose -f docker-compose.production.yml up -d

Write-Host "`nWaiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service status
Write-Host "`nProduction Services Status:" -ForegroundColor Cyan
docker ps --filter "name=jpmorgan-" --format "table {{.Names}}\t{{.Status}}"

Write-Host "`n✅ Production services are starting!" -ForegroundColor Green
Write-Host "`nAccess points:" -ForegroundColor Cyan
Write-Host "  - Grafana Dashboard: http://localhost:3000" -ForegroundColor White
Write-Host "  - API Dashboard: http://localhost:8000" -ForegroundColor White
Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "`nOpening Grafana dashboard in browser..." -ForegroundColor Yellow

Start-Sleep -Seconds 5
Start-Process "http://localhost:3000"
