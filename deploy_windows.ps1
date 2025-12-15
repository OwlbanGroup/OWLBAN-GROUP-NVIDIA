# JPMorgan Financial APIs - Windows Production Deployment Script
# This script sets up the production environment on Windows using Docker

param(
    [string]$Domain = "api.equityshieldadvocates.com",
    [string]$MainDomain = "equityshieldadvocates.com",
    [string]$Email = "admin@equityshieldadvocates.com"
)

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Blue = "Cyan"
$NC = "White"

function Write-ColorOutput {
    param([string]$Color, [string]$Message)
    Write-Host $Message -ForegroundColor $Color
}

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-ColorOutput $Blue "[$timestamp] $Message"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput $Green "✓ $Message"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput $Red "✗ $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $Yellow "⚠ $Message"
}

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator"
    exit 1
}

Write-Log "Starting JPMorgan Financial APIs Windows production deployment..."

# Step 1: Check prerequisites
Write-Log "Step 1: Checking prerequisites..."

# Check if Docker is installed
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker is installed: $dockerVersion"
    } else {
        throw "Docker not found"
    }
} catch {
    Write-Error "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
}

# Check if Docker Compose is available
try {
    $composeVersion = docker-compose --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker Compose is available: $composeVersion"
    } else {
        # Try docker compose (newer syntax)
        $composeVersion = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker Compose V2 is available: $composeVersion"
        } else {
            throw "Docker Compose not found"
        }
    }
} catch {
    Write-Error "Docker Compose is not available. Please install Docker Desktop which includes Compose."
    exit 1
}

# Step 2: Create deployment directory
Write-Log "Step 2: Setting up deployment directory..."

$deployDir = "C:\jpmorgan-api"
if (-not (Test-Path $deployDir)) {
    New-Item -ItemType Directory -Path $deployDir -Force | Out-Null
    Write-Success "Created deployment directory: $deployDir"
} else {
    Write-Success "Deployment directory already exists: $deployDir"
}

Set-Location $deployDir

# Step 3: Copy project files
Write-Log "Step 3: Setting up project files..."

# Copy all necessary files from the project directory
$projectFiles = @(
    "app_final.py",
    "requirements.txt",
    "config.py",
    "Dockerfile",
    "docker-compose.yml",
    "nginx.conf",
    "demo_script.py",
    "API_INSTRUCTIONAL_DEMO.md",
    "DEPLOYMENT_GUIDE.md",
    "DNS_SETUP.md"
)

$sourceDir = Split-Path -Parent $PSScriptRoot

foreach ($file in $projectFiles) {
    $sourcePath = Join-Path $sourceDir $file
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $deployDir -Force
        Write-Success "Copied $file"
    } else {
        Write-Warning "File not found: $file"
    }
}

# Copy src directory
if (Test-Path (Join-Path $sourceDir "src")) {
    Copy-Item (Join-Path $sourceDir "src") $deployDir -Recurse -Force
    Write-Success "Copied src directory"
}

# Create necessary directories
$dirs = @("logs", "data", "ssl", "backups")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Success "Created necessary directories"

# Step 4: Environment configuration
Write-Log "Step 4: Configuring environment..."

$envContent = @"
FLASK_ENV=production
TESTING=0
DATABASE_URL=sqlite:///data/jpmorgan_api.db
SECRET_KEY=$((New-Guid).Guid.Replace('-',''))
JWT_SECRET_KEY=$((New-Guid).Guid.Replace('-',''))
DOMAIN=$Domain
MAIN_DOMAIN=$MainDomain
SSL_EMAIL=$Email
"@

$envContent | Out-File -FilePath ".env" -Encoding UTF8
Write-Success "Created environment configuration"

# Step 5: Generate self-signed SSL certificates (for development/testing)
Write-Log "Step 5: Setting up SSL certificates..."

# Create self-signed certificate for development
$cert = New-SelfSignedCertificate -DnsName $Domain -CertStoreLocation "cert:\LocalMachine\My" -NotAfter (Get-Date).AddYears(1)

# Export certificate and private key
$certPath = Join-Path $PSScriptRoot "ssl"
$certFile = Join-Path $certPath "fullchain.pem"
$keyFile = Join-Path $certPath "privkey.pem"

# Export certificate
$certBytes = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
[System.IO.File]::WriteAllBytes($certFile, $certBytes)

# Export private key
$rsa = [System.Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($cert)
$keyBytes = $rsa.Key.Export([System.Security.Cryptography.CngKeyBlobFormat]::Pkcs8PrivateBlob)
[System.IO.File]::WriteAllBytes($keyFile, $keyBytes)

Write-Success "Generated self-signed SSL certificates"

# Step 6: Deploy application
Write-Log "Step 6: Deploying application..."

# Stop any existing containers
docker-compose down 2>$null

# Build and start services
docker-compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Success "Application deployed successfully"
} else {
    Write-Error "Application deployment failed"
    docker-compose logs
    exit 1
}

# Wait for services to start
Start-Sleep -Seconds 30

# Step 7: Verify deployment
Write-Log "Step 7: Verifying deployment..."

# Check service status
docker-compose ps

# Test health endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost/health" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Success "Health check passed"
    } else {
        Write-Warning "Health check returned status: $($response.StatusCode)"
    }
} catch {
    Write-Warning "Health check failed - this may be normal if services are still starting"
}

# Step 8: Setup monitoring
Write-Log "Step 8: Setting up monitoring..."

# Create Windows service for monitoring (using NSSM or similar)
# For now, we'll create a scheduled task
$monitorScript = @"
# Monitor script for Windows
`$apiUrl = "http://localhost/health"
`$logFile = "C:\jpmorgan-api\logs\monitor.log"

try {
    `$response = Invoke-WebRequest -Uri `$apiUrl -TimeoutSec 10
    if (`$response.StatusCode -eq 200) {
        "`$(Get-Date) - Health check passed" | Out-File -FilePath `$logFile -Append
    } else {
        "`$(Get-Date) - Health check failed: `$(`$response.StatusCode)" | Out-File -FilePath `$logFile -Append
    }
} catch {
    "`$(Get-Date) - Health check error: `$_" | Out-File -FilePath `$logFile -Append
}
"@

$monitorScript | Out-File -FilePath "monitor.ps1" -Encoding UTF8

# Create scheduled task for monitoring
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\jpmorgan-api\monitor.ps1"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "JPMorganAPIMonitor" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

Write-Success "Monitoring scheduled task created"

# Step 9: Create backup script
Write-Log "Step 9: Setting up backup system..."

$backupScript = @"
# Backup script for Windows
`$backupDir = "C:\jpmorgan-api\backups"
`$date = Get-Date -Format "yyyyMMdd_HHmmss"

if (-not (Test-Path `$backupDir)) {
    New-Item -ItemType Directory -Path `$backupDir -Force | Out-Null
}

# Backup database (if using SQLite)
if (Test-Path "C:\jpmorgan-api\data\jpmorgan_api.db") {
    Copy-Item "C:\jpmorgan-api\data\jpmorgan_api.db" "`$backupDir\db_`$date.db"
}

# Backup configuration
Compress-Archive -Path ".env", "docker-compose.yml", "nginx.conf" -DestinationPath "`$backupDir\config_`$date.zip" -Force

# Backup logs
if (Test-Path "logs") {
    Compress-Archive -Path "logs" -DestinationPath "`$backupDir\logs_`$date.zip" -Force
}

# Clean old backups (keep last 7 days)
Get-ChildItem `$backupDir -File | Where-Object { `$_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force

Write-Host "Backup completed: `$date"
"@

$backupScript | Out-File -FilePath "backup.ps1" -Encoding UTF8

# Create daily backup scheduled task
$backupAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\jpmorgan-api\backup.ps1"
$backupTrigger = New-ScheduledTaskTrigger -Daily -At "02:00"
$backupPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken
$backupSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "JPMorganAPIBackup" -Action $backupAction -Trigger $backupTrigger -Principal $backupPrincipal -Settings $backupSettings -Force | Out-Null

Write-Success "Backup system configured"

# Step 10: Final configuration and display results
Write-Log "Step 10: Final configuration..."

# Create a startup script
$startupScript = @"
# JPMorgan API Startup Script
Write-Host "Starting JPMorgan Financial APIs..." -ForegroundColor Green
Set-Location C:\jpmorgan-api
docker-compose up -d
Start-Sleep -Seconds 10
Write-Host "API should be available at: http://localhost" -ForegroundColor Green
Write-Host "Health check: http://localhost/health" -ForegroundColor Green
Write-Host "Dashboard: http://localhost/dashboard" -ForegroundColor Green
"@

$startupScript | Out-File -FilePath "start_api.ps1" -Encoding UTF8

# Display final status
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT SUMMARY" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Domain: https://$Domain" -ForegroundColor White
Write-Host "Local Access: http://localhost" -ForegroundColor White
Write-Host "Health Check: http://localhost/health" -ForegroundColor White
Write-Host "API Docs: http://localhost/api/docs" -ForegroundColor White
Write-Host "Dashboard: http://localhost/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "Service Status:" -ForegroundColor White
docker-compose ps
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Configure DNS A record: api -> YOUR_SERVER_IP" -ForegroundColor White
Write-Host "2. Test all endpoints using demo_script.py" -ForegroundColor White
Write-Host "3. Run .\start_api.ps1 to start services" -ForegroundColor White
Write-Host "4. Monitor logs in C:\jpmorgan-api\logs\" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan

Write-Success "Production environment setup complete!"

# Offer to run the demo script
$runDemo = Read-Host "Would you like to run the demo script to test the APIs? (y/n)"
if ($runDemo -eq "y" -or $runDemo -eq "Y") {
    Write-Log "Running demo script..."
    python demo_script.py
}
