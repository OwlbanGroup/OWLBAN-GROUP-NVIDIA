################################################################################
# Import Prometheus Dashboard to Grafana
# This script imports a dedicated Prometheus monitoring dashboard
################################################################################

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PROMETHEUS DASHBOARD IMPORT" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$grafanaUrl = "http://localhost:3000"
$username = "admin"
$password = "SecureGrafanaP@ss2024"
$dashboardFile = "prometheus_dashboard.json"

# Create credentials
$pair = "$($username):$($password)"
$encodedCreds = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{
    Authorization = "Basic $encodedCreds"
    "Content-Type" = "application/json"
}

################################################################################
# Step 1: Check Grafana
################################################################################

Write-Host "Step 1: Checking Grafana status..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$grafanaUrl/api/health" -Method Get
    if ($health.database -eq "ok") {
        Write-Host "  ✅ Grafana is running and healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Cannot connect to Grafana" -ForegroundColor Red
    Write-Host "  Make sure Grafana is running: docker-compose -f docker-compose.production.yml ps" -ForegroundColor Yellow
    exit 1
}

################################################################################
# Step 2: Verify Prometheus Data Source
################################################################################

Write-Host ""
Write-Host "Step 2: Verifying Prometheus data source..." -ForegroundColor Cyan
try {
    $datasource = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources/name/Prometheus" -Headers $headers -Method Get
    Write-Host "  ✅ Prometheus data source found (ID: $($datasource.id))" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Prometheus data source not found" -ForegroundColor Yellow
    Write-Host "  Run setup_grafana_dashboard.ps1 first to configure Prometheus" -ForegroundColor Yellow
    exit 1
}

################################################################################
# Step 3: Import Prometheus Dashboard
################################################################################

Write-Host ""
Write-Host "Step 3: Importing Prometheus monitoring dashboard..." -ForegroundColor Cyan

if (Test-Path $dashboardFile) {
    try {
        $dashboardJson = Get-Content $dashboardFile -Raw | ConvertFrom-Json
        
        # Prepare dashboard for import
        $importPayload = @{
            dashboard = $dashboardJson.dashboard
            overwrite = $true
            inputs = @(
                @{
                    name = "DS_PROMETHEUS"
                    type = "datasource"
                    pluginId = "prometheus"
                    value = "Prometheus"
                }
            )
        } | ConvertTo-Json -Depth 20
        
        $result = Invoke-RestMethod -Uri "$grafanaUrl/api/dashboards/db" -Headers $headers -Method Post -Body $importPayload
        
        Write-Host "  ✅ Prometheus dashboard imported successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Dashboard Details:" -ForegroundColor Cyan
        Write-Host "    Title: Prometheus Monitoring Dashboard - JPMorgan APIs" -ForegroundColor White
        Write-Host "    URL: $grafanaUrl$($result.url)" -ForegroundColor White
        Write-Host "    UID: $($result.uid)" -ForegroundColor White
        Write-Host "    Panels: 16 monitoring panels" -ForegroundColor White
        Write-Host ""
        
        $dashboardUrl = "$grafanaUrl$($result.url)"
        
    } catch {
        Write-Host "  ❌ Failed to import dashboard" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  ❌ Dashboard file not found: $dashboardFile" -ForegroundColor Red
    exit 1
}

################################################################################
# Step 4: Verify Dashboard
################################################################################

Write-Host "Step 4: Verifying dashboard..." -ForegroundColor Cyan
try {
    Start-Sleep -Seconds 2
    $dashboards = Invoke-RestMethod -Uri "$grafanaUrl/api/search?query=Prometheus" -Headers $headers -Method Get
    $promDashboard = $dashboards | Where-Object { $_.title -like "*Prometheus Monitoring*" }
    
    if ($promDashboard) {
        Write-Host "  ✅ Dashboard verified and accessible" -ForegroundColor Green
        Write-Host "  Dashboard ID: $($promDashboard.id)" -ForegroundColor White
    }
} catch {
    Write-Host "  ⚠️  Could not verify dashboard" -ForegroundColor Yellow
}

################################################################################
# Summary
################################################################################

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  PROMETHEUS DASHBOARD IMPORT COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Dashboard Access:" -ForegroundColor Cyan
Write-Host "  URL: $dashboardUrl" -ForegroundColor White
Write-Host "  Username: $username" -ForegroundColor White
Write-Host "  Password: $password" -ForegroundColor White
Write-Host ""

Write-Host "Dashboard Features:" -ForegroundColor Cyan
Write-Host "  ✅ Prometheus Status Monitor" -ForegroundColor White
Write-Host "  ✅ All Services Health Status" -ForegroundColor White
Write-Host "  ✅ Target Scrape Monitoring" -ForegroundColor White
Write-Host "  ✅ Scrape Duration Tracking" -ForegroundColor White
Write-Host "  ✅ Query Rate Monitoring" -ForegroundColor White
Write-Host "  ✅ Memory & CPU Usage" -ForegroundColor White
Write-Host "  ✅ TSDB Metrics" -ForegroundColor White
Write-Host "  ✅ Storage Size Tracking" -ForegroundColor White
Write-Host "  ✅ HTTP Request Duration" -ForegroundColor White
Write-Host "  ✅ Rule Evaluation Metrics" -ForegroundColor White
Write-Host ""

Write-Host "16 Monitoring Panels:" -ForegroundColor Cyan
Write-Host "  1. Prometheus Status" -ForegroundColor White
Write-Host "  2. All Services Status" -ForegroundColor White
Write-Host "  3. Prometheus Targets Count" -ForegroundColor White
Write-Host "  4. Scrape Duration" -ForegroundColor White
Write-Host "  5. Scrape Samples Rate" -ForegroundColor White
Write-Host "  6. Query Rate" -ForegroundColor White
Write-Host "  7. Target Scrape Health" -ForegroundColor White
Write-Host "  8. Memory Usage" -ForegroundColor White
Write-Host "  9. CPU Usage" -ForegroundColor White
Write-Host "  10. TSDB Head Series" -ForegroundColor White
Write-Host "  11. TSDB Chunks" -ForegroundColor White
Write-Host "  12. Scrape Duration by Job" -ForegroundColor White
Write-Host "  13. Scrape Samples by Job" -ForegroundColor White
Write-Host "  14. Rule Evaluation Duration" -ForegroundColor White
Write-Host "  15. Storage Size" -ForegroundColor White
Write-Host "  16. HTTP Request Duration (Percentiles)" -ForegroundColor White
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open the dashboard in your browser" -ForegroundColor White
Write-Host "  2. Review all 16 monitoring panels" -ForegroundColor White
Write-Host "  3. Set up alerts for critical metrics" -ForegroundColor White
Write-Host "  4. Customize panel thresholds" -ForegroundColor White
Write-Host ""

# Open dashboard in browser
Write-Host "Opening Prometheus dashboard in browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process $dashboardUrl

Write-Host ""
Write-Host "✅ Prometheus dashboard is now active!" -ForegroundColor Green
Write-Host ""
