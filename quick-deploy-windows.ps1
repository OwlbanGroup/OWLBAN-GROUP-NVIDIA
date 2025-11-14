################################################################################
# JPMorgan Financial APIs - Windows Production Deployment
################################################################################
# PowerShell deployment script for Windows with Docker Desktop
################################################################################

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Info { Write-ColorOutput Cyan $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }

Write-Info @"

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        JPMorgan Financial APIs - Production Deployment       ║
║                     Windows Edition                           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

"@

# Function to check if Docker Desktop is running
function Test-DockerRunning {
    try {
        $null = docker version 2>&1
        return $?
    }
    catch {
        return $false
    }
}

# Function to start Docker Desktop
function Start-DockerDesktop {
    Write-Info "Checking Docker Desktop status..."
    
    if (Test-DockerRunning) {
        Write-Success "✓ Docker Desktop is already running"
        return $true
    }
    
    Write-Warning "Docker Desktop is not running. Attempting to start..."
    
    $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    if (-not (Test-Path $dockerPath)) {
        Write-Error "✗ Docker Desktop not found at: $dockerPath"
        Write-Error ""
        Write-Error "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        return $false
    }
    
    try {
        Start-Process $dockerPath
        Write-Info "Waiting for Docker Desktop to start (this may take 30-60 seconds)..."
        
        $maxWaitTime = 120
        $waitInterval = 5
        $elapsed = 0
        
        while ($elapsed -lt $maxWaitTime) {
            Start-Sleep -Seconds $waitInterval
            $elapsed += $waitInterval
            
            if (Test-DockerRunning) {
                Write-Success "✓ Docker Desktop is now running!"
                return $true
            }
            
            Write-Info "Still waiting... ($elapsed seconds elapsed)"
        }
        
        Write-Error "✗ Docker Desktop did not start within $maxWaitTime seconds"
        Write-Warning "Please start Docker Desktop manually and wait for it to be ready, then run this script again."
        return $false
    }
    catch {
        Write-Error "✗ Failed to start Docker Desktop: $_"
        return $false
    }
}

# Check prerequisites
Write-Info "[1/7] Checking prerequisites..."

if (-not (Start-DockerDesktop)) {
    Write-Error "Cannot proceed without Docker Desktop. Exiting."
    exit 1
}

# Check Docker Compose
try {
    $null = docker-compose version 2>&1
    if (-not $?) {
        throw "Docker Compose not available"
    }
    Write-Success "✓ Docker Compose is available"
}
catch {
    Write-Error "✗ Docker Compose is not available"
    Write-Error "Please ensure Docker Desktop is properly installed with Docker Compose"
    exit 1
}

# Create necessary directories
Write-Info "[2/7] Creating directories..."
$directories = @("logs", "backups", "nginx\ssl", "models")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Success "✓ Directories created"

# Check environment file
Write-Info "[3/7] Checking environment configuration..."
if (-not (Test-Path ".env.production")) {
    if (Test-Path ".env.production.example") {
        Write-Warning "Creating .env.production from example..."
        Copy-Item ".env.production.example" ".env.production"
        Write-Warning "⚠ Please edit .env.production with your actual values"
        Write-Warning "⚠ Press Enter when ready to continue..."
        Read-Host
    }
    else {
        Write-Error "✗ .env.production.example not found"
        exit 1
    }
}
else {
    Write-Success "✓ Environment file exists"
}

# Generate SSL certificates
Write-Info "[4/7] Checking SSL certificates..."
if (-not (Test-Path "nginx\ssl\server.crt") -or -not (Test-Path "nginx\ssl\server.key")) {
    Write-Info "Generating self-signed SSL certificates..."
    try {
        python scripts/setup_https.py --action generate-self-signed --domain localhost --cert-dir nginx/ssl --days 365
        Write-Success "✓ SSL certificates generated"
        Write-Warning "⚠ For production, replace with CA-signed certificates!"
    }
    catch {
        Write-Error "✗ Failed to generate SSL certificates: $_"
        Write-Warning "You may need to generate them manually"
    }
}
else {
    Write-Success "✓ SSL certificates exist"
}

# Stop any existing containers
Write-Info "[5/7] Stopping any existing containers..."
try {
    docker-compose -f docker-compose.production.yml down 2>&1 | Out-Null
}
catch {
    # Ignore errors if no containers are running
}

# Build and start services
Write-Info "[6/7] Building and starting services..."
Write-Info "This may take several minutes on first run..."

try {
    docker-compose -f docker-compose.production.yml up -d --build
    
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE"
    }
    
    Write-Success "✓ Services started successfully"
}
catch {
    Write-Error "✗ Failed to start services: $_"
    Write-Error ""
    Write-Error "Check logs with: docker-compose -f docker-compose.production.yml logs"
    exit 1
}

# Wait for services to be ready
Write-Info "[7/7] Waiting for services to be ready..."
Start-Sleep -Seconds 15

# Health check
Write-Info ""
Write-Info "Running health checks..."

$healthCheckPassed = $false
$maxRetries = 6
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "https://localhost/health" -SkipCertificateCheck -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $healthCheckPassed = $true
            break
        }
    }
    catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Info "Health check attempt $retryCount failed, retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
}

if ($healthCheckPassed) {
    Write-Success "✓ Application is healthy"
}
else {
    Write-Warning "⚠ Health check did not pass after $maxRetries attempts"
    Write-Warning "The application may still be starting up. Check logs with:"
    Write-Warning "  docker-compose -f docker-compose.production.yml logs app"
}

# Display summary
Write-Success @"

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                  🎉 DEPLOYMENT SUCCESSFUL! 🎉                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

"@

Write-Info "Access Points:"
Write-Output "  • API:        https://localhost"
Write-Output "  • Health:     https://localhost/health"
Write-Output "  • Docs:       https://localhost/docs"
Write-Output "  • Grafana:    http://localhost:3000 (admin/SecureGrafanaP@ss2024)"
Write-Output "  • Prometheus: http://localhost:9090"
Write-Output ""

Write-Info "Management Commands:"
Write-Output "  • View logs:    docker-compose -f docker-compose.production.yml logs -f"
Write-Output "  • Stop:         docker-compose -f docker-compose.production.yml stop"
Write-Output "  • Restart:      docker-compose -f docker-compose.production.yml restart"
Write-Output "  • Status:       docker-compose -f docker-compose.production.yml ps"
Write-Output ""

Write-Warning "Important Next Steps:"
Write-Output "  1. Update SECRET_KEY in .env.production"
Write-Output "  2. Replace SSL certificates with CA-signed certificates"
Write-Output "  3. Configure your domain name"
Write-Output "  4. Set up automated backups"
Write-Output "  5. Configure monitoring alerts"
Write-Output ""

Write-Success "For detailed instructions, see: PRODUCTION_DEPLOYMENT_GUIDE.md"
Write-Output ""

# Open browser to health endpoint
Write-Info "Opening health check in browser..."
Start-Sleep -Seconds 2
Start-Process "https://localhost/health"
