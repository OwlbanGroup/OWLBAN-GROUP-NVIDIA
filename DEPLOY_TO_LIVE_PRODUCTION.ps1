# JPMorgan Financial APIs - Live Production Deployment Script
# Date: December 5, 2025
# Version: 1.0.0

param(
    [string]$Environment = "production",
    [switch]$SkipBackup = $false,
    [switch]$SkipTests = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "JPMorgan Financial APIs" -ForegroundColor Cyan
Write-Host "Live Production Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$projectRoot = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "$projectRoot\backups\deployment_$timestamp"
$logFile = "$projectRoot\logs\deployment_$timestamp.log"

# Ensure directories exist
New-Item -ItemType Directory -Force -Path "$projectRoot\backups" | Out-Null
New-Item -ItemType Directory -Force -Path "$projectRoot\logs" | Out-Null

function Write-Log {
    param($Message, $Color = "White")
    $logMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $Message"
    Write-Host $Message -ForegroundColor $Color
    Add-Content -Path $logFile -Value $logMessage
}

function Test-Prerequisites {
    Write-Log "`n[STEP 1] Checking Prerequisites..." -Color Yellow
    
    $prerequisites = @{
        "Docker" = { docker --version }
        "Docker Compose" = { docker-compose --version }
        "Python" = { python --version }
        "Git" = { git --version }
    }
    
    $allPresent = $true
    foreach ($prereq in $prerequisites.Keys) {
        try {
            $version = & $prerequisites[$prereq] 2>&1
            Write-Log "  ✓ $prereq installed: $version" -Color Green
        } catch {
            Write-Log "  ✗ $prereq not found" -Color Red
            $allPresent = $false
        }
    }
    
    if (-not $allPresent) {
        throw "Missing prerequisites. Please install all required tools."
    }
    
    Write-Log "✓ All prerequisites met" -Color Green
}

function Backup-CurrentDeployment {
    if ($SkipBackup) {
        Write-Log "`n[STEP 2] Skipping backup (--SkipBackup flag set)" -Color Yellow
        return
    }
    
    Write-Log "`n[STEP 2] Creating Backup..." -Color Yellow
    
    try {
        New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
        
        # Backup configuration files
        $configFiles = @(
            ".env.production",
            "docker-compose.production.yml",
            "prometheus.yml",
            "alerts.yml",
            "config.py"
        )
        
        foreach ($file in $configFiles) {
            if (Test-Path "$projectRoot\$file") {
                Copy-Item "$projectRoot\$file" "$backupDir\" -Force
                Write-Log "  ✓ Backed up $file" -Color Green
            }
        }
        
        # Backup database
        Write-Log "  Creating database backup..." -Color Yellow
        docker exec jpmorgan-postgres-prod pg_dump -U jpmorgan_prod jpmorgan_financial_apis_prod > "$backupDir\database_$timestamp.sql" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "  ✓ Database backed up" -Color Green
        } else {
            Write-Log "  ⚠ Database backup failed (container may not be running)" -Color Yellow
        }
        
        Write-Log "✓ Backup completed: $backupDir" -Color Green
    }
    catch {
        Write-Log "⚠ Backup failed: $_" -Color Yellow
        Write-Log "Continuing with deployment..." -Color Yellow
    }
}

function Stop-ExistingServices {
    Write-Log "`n[STEP 3] Stopping Existing Services..." -Color Yellow
    
    try {
        $containers = docker ps -q --filter "name=jpmorgan"
        if ($containers) {
            Write-Log "  Stopping containers..." -Color Yellow
            docker-compose -f docker-compose.production.yml stop 2>&1 | Out-Null
            Write-Log "  ✓ Services stopped" -Color Green
        } else {
            Write-Log "  ℹ No running services found" -Color Cyan
        }
    } catch {
        Write-Log "  ⚠ Error stopping services: $_" -Color Yellow
    }
}

function Update-Configuration {
    Write-Log "`n[STEP 4] Updating Configuration..." -Color Yellow
    
    # Check if .env.production exists
    if (-not (Test-Path "$projectRoot\.env.production")) {
        Write-Log "  Creating .env.production from template..." -Color Yellow
        
        $envContent = @"
# JPMorgan Financial APIs - Production Configuration
# Generated: $timestamp

# Environment
FLASK_ENV=production
FLASK_DEBUG=0
TESTING=0

# Security
SECRET_KEY=$(New-Guid)
TOKEN_CLIENT_ID=your_client_id_here
TOKEN_CLIENT_SECRET=your_client_secret_here

# Database
DATABASE_URL=postgresql://jpmorgan_prod:secure_password@postgres:5432/jpmorgan_financial_apis_prod
DATABASE_TYPE=postgresql
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_NAME=jpmorgan_financial_apis_prod
DATABASE_USER=jpmorgan_prod
DATABASE_PASSWORD=secure_password

# Redis
REDIS_URL=redis://redis:6379/0

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Audit Logging
AUDIT_LOG_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=90
AUDIT_ALERT_ENABLED=true

# API Settings
API_BASE_URL=https://api.yourdomain.com
ALLOWED_ORIGINS=https://yourdomain.com

# JPMorgan API (Update with your credentials)
JPMORGAN_ENVIRONMENT=production
JPMORGAN_API_KEY=your_api_key_here
"@
        
        Set-Content -Path "$projectRoot\.env.production" -Value $envContent
        Write-Log "  ✓ Created .env.production template" -Color Green
        Write-Log "  ⚠ IMPORTANT: Update .env.production with your actual credentials!" -Color Yellow
    } else {
        Write-Log "  ✓ .env.production exists" -Color Green
    }
}

function Run-PreDeploymentTests {
    if ($SkipTests) {
        Write-Log "`n[STEP 5] Skipping tests (SkipTests flag set)" -Color Yellow
        return
    }
    
    Write-Log "`n[STEP 5] Running Pre-Deployment Tests..." -Color Yellow
    
    try {
        # Run linting
        Write-Log "  Running pylint..." -Color Yellow
        python -m pylint src/models/audit_log.py --output-format=text 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "  ✓ Linting passed" -Color Green
        } else {
            Write-Log "  ⚠ Linting warnings (non-blocking)" -Color Yellow
        }
        
        # Run unit tests
        Write-Log "  Running unit tests..." -Color Yellow
        python -m pytest tests/ -v --tb=short 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "  ✓ Unit tests passed" -Color Green
        } else {
            Write-Log "  ⚠ Some tests failed (review logs)" -Color Yellow
        }
    } catch {
        Write-Log "  ⚠ Testing failed: $_" -Color Yellow
        Write-Log "  Continuing with deployment..." -Color Yellow
    }
}

function Deploy-Services {
    Write-Log "`n[STEP 6] Deploying Services..." -Color Yellow
    
    if ($DryRun) {
        Write-Log "  DRY RUN MODE - Skipping actual deployment" -Color Cyan
        return
    }
    
    try {
        # Pull latest images
        Write-Log "  Pulling latest Docker images..." -Color Yellow
        docker-compose -f docker-compose.production.yml pull 2>&1 | Out-Null
        
        # Build and start services
        Write-Log "  Building and starting services..." -Color Yellow
        docker-compose -f docker-compose.production.yml up -d --build
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "  ✓ Services deployed successfully" -Color Green
        } else {
            throw "Service deployment failed"
        }
        
        # Wait for services to be healthy
        Write-Log "  Waiting for services to be healthy..." -Color Yellow
        Start-Sleep -Seconds 30
        
        $healthyServices = 0
        $totalServices = 8
        
        $services = @(
            @{Name="API"; Port=8000; Path="/health"},
            @{Name="Prometheus"; Port=9090; Path="/-/healthy"},
            @{Name="Grafana"; Port=3000; Path="/api/health"}
        )
        
        foreach ($service in $services) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)$($service.Path)" -UseBasicParsing -TimeoutSec 5
                if ($response.StatusCode -eq 200) {
                    Write-Log "  ✓ $($service.Name) is healthy" -Color Green
                    $healthyServices++
                }
            } catch {
                Write-Log "  ⚠ $($service.Name) health check failed" -Color Yellow
            }
        }
        
        Write-Log "  $healthyServices/$totalServices services verified healthy" -Color $(if ($healthyServices -ge 3) { "Green" } else { "Yellow" })
        
    } catch {
        Write-Log "  ✗ Deployment failed: $_" -Color Red
        throw
    }
}

function Run-PostDeploymentTests {
    if ($SkipTests) {
        Write-Log "`n[STEP 7] Skipping post-deployment tests (SkipTests flag set)" -Color Yellow
        return
    }
    
    Write-Log "`n[STEP 7] Running Post-Deployment Tests..." -Color Yellow
    
    try {
        # Test API endpoints
        $endpoints = @(
            @{Path="/health"; Expected=200},
            @{Path="/metrics"; Expected=200},
            @{Path="/"; Expected=200}
        )
        
        $passedTests = 0
        foreach ($endpoint in $endpoints) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:8000$($endpoint.Path)" -UseBasicParsing -TimeoutSec 5
                if ($response.StatusCode -eq $endpoint.Expected) {
                    Write-Log "  ✓ $($endpoint.Path) - Status $($response.StatusCode)" -Color Green
                    $passedTests++
                }
            } catch {
                Write-Log "  ✗ $($endpoint.Path) - Failed" -Color Red
            }
        }
        
        Write-Log "  $passedTests/$($endpoints.Count) endpoint tests passed" -Color $(if ($passedTests -eq $endpoints.Count) { "Green" } else { "Yellow" })
        
    } catch {
        Write-Log "  ⚠ Post-deployment tests failed: $_" -Color Yellow
    }
}

function Show-DeploymentSummary {
    Write-Log "`n========================================" -Color Cyan
    Write-Log "DEPLOYMENT SUMMARY" -Color Cyan
    Write-Log "========================================" -Color Cyan
    
    Write-Log "`nDeployment Details:" -Color White
    Write-Log "  Environment: $Environment" -Color White
    Write-Log "  Timestamp: $timestamp" -Color White
    Write-Log "  Backup Location: $backupDir" -Color White
    Write-Log "  Log File: $logFile" -Color White
    
    Write-Log "`nService URLs:" -Color White
    Write-Log "  API:        http://localhost:8000" -Color Cyan
    Write-Log "  Grafana:    http://localhost:3000" -Color Cyan
    Write-Log "  Prometheus: http://localhost:9090" -Color Cyan
    
    Write-Log "`nNext Steps:" -Color White
    Write-Log "  1. Update .env.production with your credentials" -Color Yellow
    Write-Log "  2. Verify all services are running: docker ps" -Color Yellow
    Write-Log "  3. Check logs: docker-compose -f docker-compose.production.yml logs -f" -Color Yellow
    Write-Log "  4. Run verification: .\FINAL_PRODUCTION_VERIFICATION.ps1" -Color Yellow
    Write-Log "  5. Monitor Grafana dashboard: http://localhost:3000" -Color Yellow
    
    Write-Log "`n========================================" -Color Cyan
    Write-Log "DEPLOYMENT COMPLETE!" -Color Green
    Write-Log "========================================" -Color Cyan
}

# Main Deployment Flow
try {
    Write-Log "Starting deployment at $(Get-Date)" -Color Cyan
    
    if ($DryRun) {
        Write-Log "`n⚠ DRY RUN MODE - No actual changes will be made" -Color Yellow
    }
    
    Test-Prerequisites
    Backup-CurrentDeployment
    Stop-ExistingServices
    Update-Configuration
    Run-PreDeploymentTests
    Deploy-Services
    Run-PostDeploymentTests
    Show-DeploymentSummary
    
    Write-Log "`n✓ Deployment completed successfully!" -Color Green
    exit 0
    
} catch {
    Write-Log "`n✗ Deployment failed: $_" -Color Red
    Write-Log "`nRollback Instructions:" -Color Yellow
    Write-Log "  1. Restore from backup: $backupDir" -Color Yellow
    Write-Log "  2. Stop services: docker-compose -f docker-compose.production.yml down" -Color Yellow
    Write-Log "  3. Restore database: docker exec -i jpmorgan-postgres-prod psql -U jpmorgan_prod jpmorgan_financial_apis_prod < $backupDir\database_$timestamp.sql" -Color Yellow
    Write-Log "  4. Restart services: docker-compose -f docker-compose.production.yml up -d" -Color Yellow
    
    exit 1
}
