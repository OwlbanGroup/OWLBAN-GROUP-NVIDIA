################################################################################
# Grafana Dashboard Setup Script
# Automatically configures Grafana with JPMorgan Financial APIs dashboard
################################################################################

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  GRAFANA DASHBOARD SETUP" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$grafanaUrl = "http://localhost:3000"
$username = "admin"
$password = "SecureGrafanaP@ss2024"
$dashboardFile = "grafana_dashboard.json"

# Create credentials
$pair = "$($username):$($password)"
$encodedCreds = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($pair))
$headers = @{
    Authorization = "Basic $encodedCreds"
    "Content-Type" = "application/json"
}

################################################################################
# Step 1: Check Grafana is Running
################################################################################

Write-Host "Step 1: Checking Grafana status..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$grafanaUrl/api/health" -Method Get
    if ($health.database -eq "ok") {
        Write-Host "  OK Grafana is running and healthy" -ForegroundColor Green
    } else {
        Write-Host "  ERROR Grafana database issue" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ERROR Cannot connect to Grafana at $grafanaUrl" -ForegroundColor Red
    Write-Host "  Make sure Grafana is running: docker-compose -f docker-compose.production.yml ps" -ForegroundColor Yellow
    exit 1
}

################################################################################
# Step 2: Add Prometheus Data Source
################################################################################

Write-Host ""
Write-Host "Step 2: Configuring Prometheus data source..." -ForegroundColor Cyan

$datasource = @{
    name = "Prometheus"
    type = "prometheus"
    url = "http://prometheus:9090"
    access = "proxy"
    isDefault = $true
    jsonData = @{
        httpMethod = "POST"
        timeInterval = "30s"
    }
} | ConvertTo-Json

try {
    # Check if datasource already exists
    $existing = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources/name/Prometheus" -Headers $headers -Method Get -ErrorAction SilentlyContinue
    
    if ($existing) {
        Write-Host "  INFO Prometheus datasource already exists" -ForegroundColor Yellow
        Write-Host "  Updating datasource..." -ForegroundColor Cyan
        $result = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources/$($existing.id)" -Headers $headers -Method Put -Body $datasource
        Write-Host "  OK Prometheus datasource updated" -ForegroundColor Green
    } else {
        $result = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources" -Headers $headers -Method Post -Body $datasource
        Write-Host "  OK Prometheus datasource added" -ForegroundColor Green
    }
    
    # Test datasource
    Write-Host "  Testing Prometheus connection..." -ForegroundColor Cyan
    $testResult = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources/proxy/$($result.datasource.id)/api/v1/query?query=up" -Headers $headers -Method Get
    if ($testResult.status -eq "success") {
        Write-Host "  OK Prometheus connection successful" -ForegroundColor Green
    }
} catch {
    Write-Host "  ERROR Failed to configure Prometheus datasource" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
}

################################################################################
# Step 3: Import Dashboard
################################################################################

Write-Host ""
Write-Host "Step 3: Importing JPMorgan Financial APIs dashboard..." -ForegroundColor Cyan

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
        } | ConvertTo-Json -Depth 10
        
        $result = Invoke-RestMethod -Uri "$grafanaUrl/api/dashboards/db" -Headers $headers -Method Post -Body $importPayload
        
        Write-Host "  OK Dashboard imported successfully" -ForegroundColor Green
        Write-Host "  Dashboard URL: $grafanaUrl$($result.url)" -ForegroundColor Cyan
        
        # Save dashboard URL for later
        $dashboardUrl = "$grafanaUrl$($result.url)"
        
    } catch {
        Write-Host "  ERROR Failed to import dashboard" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  WARNING Dashboard file not found: $dashboardFile" -ForegroundColor Yellow
    Write-Host "  Creating a basic dashboard instead..." -ForegroundColor Cyan
    
    # Create a simple dashboard
    $basicDashboard = @{
        dashboard = @{
            title = "JPMorgan Financial APIs - Basic Monitoring"
            tags = @("jpmorgan", "financial", "api")
            timezone = "browser"
            panels = @(
                @{
                    id = 1
                    title = "API Health Status"
                    type = "stat"
                    targets = @(
                        @{
                            expr = "up"
                            refId = "A"
                        }
                    )
                    gridPos = @{
                        h = 8
                        w = 6
                        x = 0
                        y = 0
                    }
                },
                @{
                    id = 2
                    title = "Request Rate"
                    type = "graph"
                    targets = @(
                        @{
                            expr = "rate(http_requests_total_final[5m])"
                            refId = "A"
                        }
                    )
                    gridPos = @{
                        h = 8
                        w = 18
                        x = 6
                        y = 0
                    }
                }
            )
            refresh = "30s"
        }
        overwrite = $true
    } | ConvertTo-Json -Depth 10
    
    try {
        $result = Invoke-RestMethod -Uri "$grafanaUrl/api/dashboards/db" -Headers $headers -Method Post -Body $basicDashboard
        Write-Host "  OK Basic dashboard created" -ForegroundColor Green
        $dashboardUrl = "$grafanaUrl$($result.url)"
    } catch {
        Write-Host "  ERROR Failed to create basic dashboard" -ForegroundColor Red
    }
}

################################################################################
# Step 4: Configure Dashboard Settings
################################################################################

Write-Host ""
Write-Host "Step 4: Configuring dashboard settings..." -ForegroundColor Cyan

# Set home dashboard
try {
    $orgPrefs = @{
        homeDashboardId = 0
        theme = "dark"
        timezone = "browser"
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "$grafanaUrl/api/org/preferences" -Headers $headers -Method Put -Body $orgPrefs | Out-Null
    Write-Host "  OK Dashboard preferences configured" -ForegroundColor Green
} catch {
    Write-Host "  WARNING Could not set dashboard preferences" -ForegroundColor Yellow
}

################################################################################
# Step 5: Create Alert Notification Channel
################################################################################

Write-Host ""
Write-Host "Step 5: Setting up alert notifications..." -ForegroundColor Cyan

$notificationChannel = @{
    name = "Email Alerts"
    type = "email"
    isDefault = $true
    sendReminder = $true
    settings = @{
        addresses = "admin@example.com"
        autoResolve = $true
    }
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "$grafanaUrl/api/alert-notifications" -Headers $headers -Method Post -Body $notificationChannel -ErrorAction SilentlyContinue
    Write-Host "  OK Alert notification channel created" -ForegroundColor Green
    Write-Host "  NOTE Update email address in Grafana settings" -ForegroundColor Yellow
} catch {
    Write-Host "  INFO Alert notification may already exist or requires configuration" -ForegroundColor Yellow
}

################################################################################
# Summary
################################################################################

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  GRAFANA SETUP COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Grafana Access:" -ForegroundColor Cyan
Write-Host "  URL: $grafanaUrl" -ForegroundColor White
Write-Host "  Username: $username" -ForegroundColor White
Write-Host "  Password: $password" -ForegroundColor White
Write-Host ""

if ($dashboardUrl) {
    Write-Host "Dashboard:" -ForegroundColor Cyan
    Write-Host "  $dashboardUrl" -ForegroundColor White
    Write-Host ""
}

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open Grafana in your browser" -ForegroundColor White
Write-Host "  2. Review the dashboard" -ForegroundColor White
Write-Host "  3. Customize panels as needed" -ForegroundColor White
Write-Host "  4. Set up alert rules" -ForegroundColor White
Write-Host "  5. Configure notification channels" -ForegroundColor White
Write-Host ""

Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  Read: GRAFANA_DASHBOARD_SETUP_GUIDE.md" -ForegroundColor White
Write-Host ""

# Open Grafana in browser
Write-Host "Opening Grafana in browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 2

if ($dashboardUrl) {
    Start-Process $dashboardUrl
} else {
    Start-Process $grafanaUrl
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
