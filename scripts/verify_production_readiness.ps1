# Production Readiness Verification Script
# JPMorgan Financial APIs
# Version: 1.0.0

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PRODUCTION READINESS VERIFICATION" -ForegroundColor Cyan
Write-Host "  JPMorgan Financial APIs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$projectRoot = "c:\Users\bizle\Desktop\jpmorgan_financial_apis"
Set-Location $projectRoot

$results = @{
    Passed = 0
    Failed = 0
    Warnings = 0
}

function Test-Check {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [string]$SuccessMessage,
        [string]$FailureMessage
    )
    
    Write-Host "Checking: $Name..." -ForegroundColor Yellow -NoNewline
    
    try {
        $result = & $Test
        if ($result) {
            Write-Host " ✓ PASS" -ForegroundColor Green
            Write-Host "  → $SuccessMessage" -ForegroundColor Gray
            $script:results.Passed++
            return $true
        } else {
            Write-Host " ✗ FAIL" -ForegroundColor Red
            Write-Host "  → $FailureMessage" -ForegroundColor Gray
            $script:results.Failed++
            return $false
        }
    } catch {
        Write-Host " ✗ ERROR" -ForegroundColor Red
        Write-Host "  → $($_.Exception.Message)" -ForegroundColor Gray
        $script:results.Failed++
        return $false
    }
}

Write-Host "PHASE 1: FILE STRUCTURE VERIFICATION" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check Phase 3 files
Test-Check -Name "Phase 3: validators_comprehensive.py" -Test {
    Test-Path "src/validators_comprehensive.py"
} -SuccessMessage "Comprehensive validators module found" -FailureMessage "Missing validators_comprehensive.py"

Test-Check -Name "Phase 3: structured_logger.py" -Test {
    Test-Path "src/structured_logger.py"
} -SuccessMessage "Structured logger module found" -FailureMessage "Missing structured_logger.py"

Test-Check -Name "Phase 3: database_optimizer.py" -Test {
    Test-Path "src/database_optimizer.py"
} -SuccessMessage "Database optimizer module found" -FailureMessage "Missing database_optimizer.py"

Test-Check -Name "Phase 3: test_comprehensive.py" -Test {
    Test-Path "tests/test_comprehensive.py"
} -SuccessMessage "Comprehensive test suite found" -FailureMessage "Missing test_comprehensive.py"

# Check Phase 4 files
Test-Check -Name "Phase 4: swagger_config.py" -Test {
    Test-Path "src/swagger_config.py"
} -SuccessMessage "Swagger configuration found" -FailureMessage "Missing swagger_config.py"

Test-Check -Name "Phase 4: Grafana dashboard" -Test {
    Test-Path "grafana/dashboards/jpmorgan_api_dashboard.json"
} -SuccessMessage "Grafana dashboard configuration found" -FailureMessage "Missing Grafana dashboard"

Test-Check -Name "Phase 4: security_audit.py" -Test {
    Test-Path "scripts/security_audit.py"
} -SuccessMessage "Security audit script found" -FailureMessage "Missing security_audit.py"

# Check main application
Test-Check -Name "Main Application: app_final.py" -Test {
    Test-Path "app_final.py"
} -SuccessMessage "Main application file found" -FailureMessage "Missing app_final.py"

Write-Host ""
Write-Host "PHASE 2: PYTHON ENVIRONMENT VERIFICATION" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Test-Check -Name "Python Installation" -Test {
    $pythonVersion = python --version 2>&1
    $pythonVersion -match "Python 3\."
} -SuccessMessage "Python 3.x installed" -FailureMessage "Python 3.x not found"

# Check pip
Test-Check -Name "Pip Package Manager" -Test {
    $pipVersion = pip --version 2>&1
    $pipVersion -match "pip"
} -SuccessMessage "Pip is available" -FailureMessage "Pip not found"

# Check critical dependencies
$criticalPackages = @("flask", "sqlalchemy", "redis", "pytest", "prometheus-client")
foreach ($package in $criticalPackages) {
    Test-Check -Name "Package: $package" -Test {
        $installed = pip list 2>&1 | Select-String $package
        $null -ne $installed
    } -SuccessMessage "$package is installed" -FailureMessage "$package not installed"
}

Write-Host ""
Write-Host "PHASE 3: DOCKER ENVIRONMENT VERIFICATION" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker
Test-Check -Name "Docker Installation" -Test {
    $dockerVersion = docker --version 2>&1
    $dockerVersion -match "Docker version"
} -SuccessMessage "Docker is installed" -FailureMessage "Docker not found"

# Check Docker Compose
Test-Check -Name "Docker Compose" -Test {
    $composeVersion = docker-compose --version 2>&1
    $composeVersion -match "docker-compose version"
} -SuccessMessage "Docker Compose is available" -FailureMessage "Docker Compose not found"

# Check if Docker is running
Test-Check -Name "Docker Service Status" -Test {
    $dockerInfo = docker info 2>&1
    $dockerInfo -notmatch "error"
} -SuccessMessage "Docker service is running" -FailureMessage "Docker service not running"

# Check production containers
Write-Host ""
Write-Host "Checking Docker containers..." -ForegroundColor Yellow
try {
    $containers = docker-compose -f docker-compose.production.yml ps 2>&1
    if ($containers -match "Up") {
        Write-Host "  ✓ Production containers are running" -ForegroundColor Green
        $script:results.Passed++
    } else {
        Write-Host "  ⚠ Production containers not running" -ForegroundColor Yellow
        Write-Host "  → Run: docker-compose -f docker-compose.production.yml up -d" -ForegroundColor Gray
        $script:results.Warnings++
    }
} catch {
    Write-Host "  ⚠ Could not check container status" -ForegroundColor Yellow
    $script:results.Warnings++
}

Write-Host ""
Write-Host "PHASE 4: PYTHON IMPORTS VERIFICATION" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Test Phase 3 imports
Test-Check -Name "Import: validators_comprehensive" -Test {
    $output = python -c "from src.validators_comprehensive import ComprehensiveValidators; print('OK')" 2>&1
    $output -match "OK"
} -SuccessMessage "Validators module imports successfully" -FailureMessage "Failed to import validators"

Test-Check -Name "Import: structured_logger" -Test {
    $output = python -c "from src.structured_logger import app_logger; print('OK')" 2>&1
    $output -match "OK"
} -SuccessMessage "Logger module imports successfully" -FailureMessage "Failed to import logger"

Test-Check -Name "Import: database_optimizer" -Test {
    $output = python -c "from src.database_optimizer import DatabaseOptimizer; print('OK')" 2>&1
    $output -match "OK"
} -SuccessMessage "Database optimizer imports successfully" -FailureMessage "Failed to import database optimizer"

# Test Phase 4 imports
Test-Check -Name "Import: swagger_config" -Test {
    $output = python -c "from src.swagger_config import configure_swagger; print('OK')" 2>&1
    $output -match "OK"
} -SuccessMessage "Swagger config imports successfully" -FailureMessage "Failed to import swagger config"

Write-Host ""
Write-Host "PHASE 5: SERVICE HEALTH CHECKS" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# Check API health
Test-Check -Name "API Health Endpoint" -Test {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
        $response.status -eq "healthy"
    } catch {
        $false
    }
} -SuccessMessage "API is healthy and responding" -FailureMessage "API not responding or unhealthy"

# Check Prometheus
Test-Check -Name "Prometheus Service" -Test {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -Method Get -TimeoutSec 5 -ErrorAction Stop
        $response.StatusCode -eq 200
    } catch {
        $false
    }
} -SuccessMessage "Prometheus is healthy" -FailureMessage "Prometheus not responding"

# Check Grafana
Test-Check -Name "Grafana Service" -Test {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:3000/api/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
        $response.database -eq "ok"
    } catch {
        $false
    }
} -SuccessMessage "Grafana is healthy" -FailureMessage "Grafana not responding"

Write-Host ""
Write-Host "PHASE 6: CONFIGURATION VERIFICATION" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check .env file
Test-Check -Name "Environment Configuration" -Test {
    Test-Path ".env"
} -SuccessMessage ".env file exists" -FailureMessage ".env file missing"

# Check docker-compose.production.yml
Test-Check -Name "Production Docker Compose" -Test {
    Test-Path "docker-compose.production.yml"
} -SuccessMessage "Production compose file exists" -FailureMessage "Production compose file missing"

# Check requirements.txt
Test-Check -Name "Python Requirements" -Test {
    Test-Path "requirements.txt"
} -SuccessMessage "Requirements file exists" -FailureMessage "Requirements file missing"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$total = $results.Passed + $results.Failed + $results.Warnings
$passRate = if ($total -gt 0) { [math]::Round(($results.Passed / $total) * 100, 1) } else { 0 }

Write-Host "Total Checks: $total" -ForegroundColor White
Write-Host "Passed: $($results.Passed)" -ForegroundColor Green
Write-Host "Failed: $($results.Failed)" -ForegroundColor Red
Write-Host "Warnings: $($results.Warnings)" -ForegroundColor Yellow
Write-Host "Pass Rate: $passRate%" -ForegroundColor $(if ($passRate -ge 90) { "Green" } elseif ($passRate -ge 70) { "Yellow" } else { "Red" })
Write-Host ""

if ($results.Failed -eq 0 -and $passRate -ge 90) {
    Write-Host "✓ PRODUCTION READINESS: VERIFIED" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Run comprehensive tests: pytest tests/test_comprehensive.py -v" -ForegroundColor White
    Write-Host "2. Run security audit: python scripts/security_audit.py" -ForegroundColor White
    Write-Host "3. Review PRODUCTION_READINESS_EXECUTION_PLAN.md" -ForegroundColor White
    Write-Host "4. Proceed with deployment when ready" -ForegroundColor White
} elseif ($results.Failed -gt 0) {
    Write-Host "✗ PRODUCTION READINESS: ISSUES FOUND" -ForegroundColor Red
    Write-Host ""
    Write-Host "Action Required:" -ForegroundColor Yellow
    Write-Host "1. Review failed checks above" -ForegroundColor White
    Write-Host "2. Fix critical issues" -ForegroundColor White
    Write-Host "3. Re-run this verification script" -ForegroundColor White
    Write-Host "4. Consult PRODUCTION_READINESS_EXECUTION_PLAN.md for guidance" -ForegroundColor White
} else {
    Write-Host "⚠ PRODUCTION READINESS: WARNINGS PRESENT" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Recommended Actions:" -ForegroundColor Yellow
    Write-Host "1. Review warnings above" -ForegroundColor White
    Write-Host "2. Address warnings if possible" -ForegroundColor White
    Write-Host "3. Proceed with caution" -ForegroundColor White
}

Write-Host ""
Write-Host "Verification completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

# Return exit code based on results
if ($results.Failed -gt 0) {
    exit 1
} else {
    exit 0
}
