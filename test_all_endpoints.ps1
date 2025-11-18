# Comprehensive API Testing Script for JPMorgan Financial APIs
# Tests all major endpoints and generates a detailed report

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs - Full Testing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "http://localhost:8000"
$testResults = @()

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null
    )
    
    Write-Host "Testing: $Name..." -NoNewline
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            TimeoutSec = 10
        }
        
        if ($Headers.Count -gt 0) {
            $params.Headers = $Headers
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-WebRequest @params -ErrorAction Stop
        
        $result = @{
            Name = $Name
            Status = "✅ PASS"
            StatusCode = $response.StatusCode
            ResponseTime = "OK"
        }
        
        Write-Host " ✅ PASS ($($response.StatusCode))" -ForegroundColor Green
        
    } catch {
        $statusCode = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { "N/A" }
        $result = @{
            Name = $Name
            Status = "❌ FAIL"
            StatusCode = $statusCode
            Error = $_.Exception.Message
        }
        
        Write-Host " ❌ FAIL ($statusCode)" -ForegroundColor Red
    }
    
    return $result
}

Write-Host "=== Core API Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 1. Health Check
$testResults += Test-Endpoint -Name "Health Check" -Url "$baseUrl/health"

# 2. Metrics
$testResults += Test-Endpoint -Name "Prometheus Metrics" -Url "$baseUrl/metrics"

# 3. Telemetry Metrics
$testResults += Test-Endpoint -Name "Telemetry Metrics" -Url "$baseUrl/telemetry/metrics"

# 4. Data Formats
$testResults += Test-Endpoint -Name "Data Formats" -Url "$baseUrl/data/formats"

# 5. WebSocket Status
$testResults += Test-Endpoint -Name "WebSocket Status" -Url "$baseUrl/ws/status"

Write-Host ""
Write-Host "=== Authentication Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 6. Login (should fail without valid credentials)
$loginBody = '{"username":"testuser","password":"testpass"}'
$testResults += Test-Endpoint -Name "User Login" -Url "$baseUrl/auth/login" -Method "POST" -Body $loginBody

# 7. Auth Me (should fail without token)
$testResults += Test-Endpoint -Name "Auth Me (No Token)" -Url "$baseUrl/auth/me"

# 8. User Registration
$registerBody = '{"username":"newuser","password":"newpass123","email":"test@example.com"}'
$testResults += Test-Endpoint -Name "User Registration" -Url "$baseUrl/auth/register" -Method "POST" -Body $registerBody

Write-Host ""
Write-Host "=== Telemetry Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 9. Post Telemetry (should fail without auth)
$telemetryBody = '{"metric":"test","value":100}'
$testResults += Test-Endpoint -Name "Post Telemetry" -Url "$baseUrl/telemetry" -Method "POST" -Body $telemetryBody

# 10. Batch Telemetry
$batchBody = '{"metrics":[{"name":"test1","value":100},{"name":"test2","value":200}]}'
$testResults += Test-Endpoint -Name "Batch Telemetry" -Url "$baseUrl/telemetry/batch" -Method "POST" -Body $batchBody

# 11. Export Telemetry
$testResults += Test-Endpoint -Name "Export Telemetry" -Url "$baseUrl/telemetry/export"

Write-Host ""
Write-Host "=== ML Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 12. ML Anomalies
$anomalyBody = '{"data":[1,2,3,4,5,100]}'
$testResults += Test-Endpoint -Name "ML Anomaly Detection" -Url "$baseUrl/ml/anomalies" -Method "POST" -Body $anomalyBody

# 13. ML Train
$trainBody = '{"model":"test","data":[]}'
$testResults += Test-Endpoint -Name "ML Training" -Url "$baseUrl/ml/train" -Method "POST" -Body $trainBody

Write-Host ""
Write-Host "=== Data Conversion Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 14. Data Convert
$convertBody = '{"data":"test","format":"json"}'
$testResults += Test-Endpoint -Name "Data Conversion" -Url "$baseUrl/data/convert" -Method "POST" -Body $convertBody

Write-Host ""
Write-Host "=== Storage Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 15. Storage Export
$exportBody = '{"format":"json"}'
$testResults += Test-Endpoint -Name "Storage Export" -Url "$baseUrl/storage/export" -Method "POST" -Body $exportBody

Write-Host ""
Write-Host "=== MCP (GitHub) Endpoints ===" -ForegroundColor Yellow
Write-Host ""

# 16. MCP Repos
$testResults += Test-Endpoint -Name "MCP Repositories" -Url "$baseUrl/mcp/repos"

# 17. MCP Issues (example repo)
$testResults += Test-Endpoint -Name "MCP Issues List" -Url "$baseUrl/mcp/issues/microsoft/vscode"

Write-Host ""
Write-Host "=== Dashboard Endpoint ===" -ForegroundColor Yellow
Write-Host ""

# 18. Dashboard
$testResults += Test-Endpoint -Name "Dashboard Page" -Url "$baseUrl/dashboard"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$passCount = ($testResults | Where-Object { $_.Status -eq "✅ PASS" }).Count
$failCount = ($testResults | Where-Object { $_.Status -eq "❌ FAIL" }).Count
$totalCount = $testResults.Count

Write-Host "Total Tests: $totalCount" -ForegroundColor White
Write-Host "Passed: $passCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor Red
Write-Host "Success Rate: $([math]::Round(($passCount/$totalCount)*100, 2))%" -ForegroundColor $(if ($passCount -eq $totalCount) { "Green" } else { "Yellow" })

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Detailed Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($result in $testResults) {
    Write-Host "$($result.Status) $($result.Name) - Status Code: $($result.StatusCode)"
    if ($result.Error) {
        Write-Host "   Error: $($result.Error)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Testing Complete!" -ForegroundColor Green
Write-Host ""

# Export results to JSON
$testResults | ConvertTo-Json -Depth 10 | Out-File "test_results.json"
Write-Host "Results exported to: test_results.json" -ForegroundColor Cyan
