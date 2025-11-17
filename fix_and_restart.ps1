# Fix and Restart JPMorgan Financial APIs
# This script fixes the missing psycopg2 dependency and restarts all containers

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs - Fix & Restart" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop all containers
Write-Host "[Step 1/4] Stopping all containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml down
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Containers stopped successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to stop containers" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Remove old images (optional but recommended)
Write-Host "[Step 2/4] Removing old API image..." -ForegroundColor Yellow
docker rmi jpmorgan_financial_apis-app -f 2>$null
Write-Host "✓ Old image removed (if existed)" -ForegroundColor Green
Write-Host ""

# Step 3: Rebuild with no cache
Write-Host "[Step 3/4] Rebuilding Docker images (this may take 5-10 minutes)..." -ForegroundColor Yellow
Write-Host "Building with updated requirements.txt that includes psycopg2-binary..." -ForegroundColor Cyan
docker-compose -f docker-compose.production.yml build --no-cache
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Images rebuilt successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to rebuild images" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 4: Start all containers
Write-Host "[Step 4/4] Starting all containers..." -ForegroundColor Yellow
docker-compose -f docker-compose.production.yml up -d
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Containers started successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to start containers" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Wait for containers to initialize
Write-Host "Waiting 10 seconds for containers to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
Write-Host ""

# Check container status
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Container Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker-compose -f docker-compose.production.yml ps
Write-Host ""

# Check API logs
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Container Logs (Last 20 lines)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker logs jpmorgan-api-prod --tail 20
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Dependencies updated (psycopg2-binary added)" -ForegroundColor Green
Write-Host "✓ Docker images rebuilt" -ForegroundColor Green
Write-Host "✓ Containers restarted" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Monitor API logs: docker logs jpmorgan-api-prod -f" -ForegroundColor White
Write-Host "2. Check all containers: docker-compose -f docker-compose.production.yml ps" -ForegroundColor White
Write-Host "3. Test health endpoint: curl -k https://localhost/health" -ForegroundColor White
Write-Host "4. Test API docs: curl -k https://localhost/docs" -ForegroundColor White
Write-Host "5. Access Grafana: http://localhost:3000 (admin/SecureGrafanaP@ss2024)" -ForegroundColor White
Write-Host ""
Write-Host "If API is still restarting, check logs for errors:" -ForegroundColor Yellow
Write-Host "docker logs jpmorgan-api-prod --tail 50" -ForegroundColor White
Write-Host ""
