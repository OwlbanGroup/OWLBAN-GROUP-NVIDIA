################################################################################
# Prometheus Demo Script - Interactive Queries
################################################################################

Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  PROMETHEUS MONITORING DEMO" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""

$prometheusUrl = "http://localhost:9090"

function Invoke-PrometheusQuery {
    param(
        [string]$Query,
        [string]$Description
    )
    
    Write-Host "Query: $Description" -ForegroundColor Cyan
    Write-Host "PromQL: $Query" -ForegroundColor Gray
    
    try {
        $encodedQuery = [System.Uri]::EscapeDataString($Query)
        $url = "$prometheusUrl/api/v1/query?query=$encodedQuery"
        
        $response = Invoke-RestMethod -Uri $url -Method Get
        
        if ($response.status -eq "success") {
            $results = $response.data.result
            
            if ($results.Count -eq 0) {
                Write-Host "  No data available" -ForegroundColor Yellow
            } else {
                foreach ($result in $results) {
                    $metric = $result.metric
                    $value = $result.value[1]
                    
                    $labels = ($metric.PSObject.Properties | Where-Object { $_.Name -ne "__name__" } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", "
                    
                    if ($labels) {
                        Write-Host "  OK $labels = $value" -ForegroundColor Green
                    } else {
                        Write-Host "  OK Value = $value" -ForegroundColor Green
                    }
                }
            }
        } else {
            Write-Host "  ERROR Query failed: $($response.error)" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
}

# Demo 1: Check Service Health
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "1. SERVICE HEALTH MONITORING" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Invoke-PrometheusQuery -Query "up" -Description "All Services Status (1 is up, 0 is down)"

# Demo 2: Count Services
Invoke-PrometheusQuery -Query "count(up == 1)" -Description "Number of Services UP"
Invoke-PrometheusQuery -Query "count(up == 0)" -Description "Number of Services DOWN"

# Demo 3: API Metrics
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "2. API PERFORMANCE METRICS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Invoke-PrometheusQuery -Query "http_requests_total_final" -Description "Total HTTP Requests"
Invoke-PrometheusQuery -Query "rate(http_requests_total_final[5m])" -Description "Request Rate (per second, last 5 min)"
Invoke-PrometheusQuery -Query "active_connections_final" -Description "Active Connections"

# Demo 4: Error Monitoring
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "3. ERROR MONITORING" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Invoke-PrometheusQuery -Query "errors_total_final" -Description "Total Errors"
Invoke-PrometheusQuery -Query "rate(errors_total_final[5m])" -Description "Error Rate (per second, last 5 min)"

# Demo 5: Business Metrics
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "4. BUSINESS METRICS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Invoke-PrometheusQuery -Query "telemetry_events_processed_total_final" -Description "Telemetry Events Processed"
Invoke-PrometheusQuery -Query "anomaly_detections_total_final" -Description "Anomaly Detections"

# Demo 6: System Resources
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "5. SYSTEM RESOURCES" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Invoke-PrometheusQuery -Query "process_resident_memory_bytes" -Description "Memory Usage (bytes)"
Invoke-PrometheusQuery -Query "process_cpu_seconds_total" -Description "CPU Time (seconds)"

# Summary
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host "  DEMO COMPLETE" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "SUCCESS: Prometheus is working!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open Prometheus UI: http://localhost:9090" -ForegroundColor White
Write-Host "  2. Open Grafana: http://localhost:3000" -ForegroundColor White
Write-Host "  3. Read the guide: PROMETHEUS_SETUP_GUIDE.md" -ForegroundColor White
Write-Host ""
Write-Host "Try these queries in Prometheus UI:" -ForegroundColor Cyan
Write-Host "  - up" -ForegroundColor White
Write-Host "  - rate(http_requests_total_final[5m])" -ForegroundColor White
Write-Host "  - active_connections_final" -ForegroundColor White
Write-Host ""

# Open Prometheus in browser
Write-Host "Opening Prometheus UI..." -ForegroundColor Cyan
Start-Process "http://localhost:9090"
Start-Sleep -Seconds 2
Write-Host "Opening Grafana..." -ForegroundColor Cyan
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "Enjoy monitoring your production environment!" -ForegroundColor Green
Write-Host ""
