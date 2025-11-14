# Quick Docker Status Check Script
# JPMorgan Financial APIs

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Docker Container Status Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

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

# List all JPMorgan containers
Write-Host "JPMorgan Containers:" -ForegroundColor Cyan
Write-Host ""
docker ps -a --filter "name=jpmorgan-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Out-String | Write-Host

# Count containers by status
$runningCount = (docker ps --filter "name=jpmorgan-" --format "{{.Names}}").Count
$totalCount = (docker ps -a --filter "name=jpmorgan-" --format "{{.Names}}").Count
$exitedCount = (docker ps -a --filter "name=jpmorgan-" --filter "status=exited" --format "{{.Names}}").Count
$restartingCount = (docker ps -a --filter "name=jpmorgan-" --filter "status=restarting" --format "{{.Names}}").Count
$createdCount = (docker ps -a --filter "name=jpmorgan-" --filter "status=created" --format "{{.Names}}").Count

Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Total Containers: $totalCount" -ForegroundColor White
Write-Host "  Running: $runningCount" -ForegroundColor Green
if ($exitedCount -gt 0) {
    Write-Host "  Exited: $exitedCount" -ForegroundColor Red
}
if ($restartingCount -gt 0) {
    Write-Host "  Restarting: $restartingCount" -ForegroundColor Yellow
}
if ($createdCount -gt 0) {
    Write-Host "  Created (not started): $createdCount" -ForegroundColor Yellow
}
Write-Host ""

# Check for problem containers
if ($exitedCount -gt 0 -or $restartingCount -gt 0 -or $createdCount -gt 0) {
    Write-Host "⚠ Issues Detected!" -ForegroundColor Yellow
    Write-Host ""
    
    if ($exitedCount -gt 0) {
        Write-Host "Exited Containers:" -ForegroundColor Red
        docker ps -a --filter "name=jpmorgan-" --filter "status=exited" --format "  - {{.Names}}" | Write-Host
        Write-Host ""
    }
    
    if ($restartingCount -gt 0) {
        Write-Host "Restarting Containers:" -ForegroundColor Yellow
        docker ps -a --filter "name=jpmorgan-" --filter "status=restarting" --format "  - {{.Names}}" | Write-Host
        Write-Host ""
    }
    
    if ($createdCount -gt 0) {
        Write-Host "Created but Not Started:" -ForegroundColor Yellow
        docker ps -a --filter "name=jpmorgan-" --filter "status=created" --format "  - {{.Names}}" | Write-Host
        Write-Host ""
    }
    
    Write-Host "Recommended Action:" -ForegroundColor Cyan
    Write-Host "  Run the backup and fix script:" -ForegroundColor White
    Write-Host "  .\backup_and_fix_docker.ps1" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "✓ All containers are running properly!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Service URLs:" -ForegroundColor Cyan
    Write-Host "  API Health: http://localhost:8000/health" -ForegroundColor White
    Write-Host "  Grafana: http://localhost:3000" -ForegroundColor White
    Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor White
    Write-Host ""
}

# Check Docker volumes
Write-Host "Docker Volumes:" -ForegroundColor Cyan
docker volume ls --filter "name=jpmorgan" --format "table {{.Name}}\t{{.Driver}}" | Out-String | Write-Host

# Check Docker networks
Write-Host "Docker Networks:" -ForegroundColor Cyan
docker network ls --filter "name=jpmorgan" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}" | Out-String | Write-Host

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Status Check Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
