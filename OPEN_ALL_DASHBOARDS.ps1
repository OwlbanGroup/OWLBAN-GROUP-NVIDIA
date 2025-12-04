# ============================================
# JPMorgan Financial APIs - Dashboard Launcher
# ============================================
# Opens all monitoring dashboards in your default browser
# Last Updated: December 2025
# ============================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  JPMorgan Financial APIs" -ForegroundColor Yellow
Write-Host "  Dashboard Launcher" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if services are running
Write-Host "Checking service status..." -ForegroundColor Yellow
$services = docker-compose -f docker-compose.production.yml ps --services --filter "status=running"

if ($services) {
    Write-Host "✅ Services are running!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "⚠️  Warning: Some services may not be running" -ForegroundColor Red
    Write-Host "Run: docker-compose -f docker-compose.production.yml up -d" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

# Dashboard URLs
$dashboards = @{
    "Grafana (Main Dashboard)" = "http://localhost:3000";
    "Prometheus (Metrics)" = "http://localhost:9090";
    "API Documentation" = "http://localhost:8000/docs";
    "API Health Check" = "http://localhost:8000/health";
    "AlertManager" = "http://localhost:9093"
}

Write-Host "Opening dashboards..." -ForegroundColor Yellow
Write-Host ""

# Open each dashboard
foreach ($dashboard in $dashboards.GetEnumerator()) {
    Write-Host "  📊 Opening: $($dashboard.Key)" -ForegroundColor Cyan
    Write-Host "     URL: $($dashboard.Value)" -ForegroundColor Gray
    Start-Process $dashboard.Value
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  All Dashboards Opened!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Display credentials
Write-Host "📝 Grafana Login Credentials:" -ForegroundColor Yellow
Write-Host "   Username: admin" -ForegroundColor White
Write-Host "   Password: SecureGrafanaP@ss2024" -ForegroundColor White
Write-Host ""

# Display quick tips
Write-Host "💡 Quick Tips:" -ForegroundColor Yellow
Write-Host "   • Grafana: Main monitoring dashboard" -ForegroundColor Gray
Write-Host "   • Prometheus: Raw metrics and queries" -ForegroundColor Gray
Write-Host "   • API Docs: Interactive API documentation" -ForegroundColor Gray
Write-Host "   • Health Check: System status" -ForegroundColor Gray
Write-Host "   • AlertManager: Alert configuration" -ForegroundColor Gray
Write-Host ""

Write-Host "📚 For more information, see:" -ForegroundColor Yellow
Write-Host "   PROJECT_DASHBOARDS_CENTRAL_HUB.md" -ForegroundColor White
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
