#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy OWLBAN GROUP authentication services.
.DESCRIPTION
    Builds and starts all auth services using Docker Compose.
    Includes: OWLBAN GROUP website, OSCAR BROOME, BLACKBOX AI, API Server.
.EXAMPLE
    .\scripts\deploy.ps1
    .\scripts\deploy.ps1 -initUsers
    .\scripts\deploy.ps1 -stop
    .\scripts\deploy.ps1 -logs
#>

param(
    [switch]$initUsers,
    [switch]$stop,
    [switch]$logs,
    [switch]$build,
    [switch]$help
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host @"

OWLBAN GROUP - Auth Services Deployment
=======================================

Usage: .\scripts\deploy.ps1 [Options]

Options:
  -initUsers    Seed default admin/demo users after deployment
  -stop         Stop all running services
  -logs         Show service logs
  -build        Force rebuild of all images
  -help         Show this help message

Services:
  - API Server (FastAPI)        : http://localhost:8000
  - OWLBAN GROUP Website        : http://localhost:3000
  - OSCAR BROOME Revenue        : http://localhost:3001
  - BLACKBOX AI Portal          : http://localhost:3002
  - Web Dashboard (Streamlit)   : http://localhost:8501
  - Grafana Monitoring          : http://localhost:3000
  - Prometheus                  : http://localhost:9090

Default Users (after -initUsers):
  - admin@owlban.com    / Admin2024!     (OWLBAN_GROUP admin)
  - demo@owlban.com     / Demo2024!      (OWLBAN_GROUP user)
  - oscar@owlban.com    / Oscar2024!     (OSCAR_BROOME executive)
  - ai@owlban.com       / Ai2024!!       (BLACKBOX_AI developer)

"@
}

function Stop-Services {
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "All services stopped." -ForegroundColor Green
}

function Start-Services {
    Write-Host "Starting OWLBAN GROUP auth services..." -ForegroundColor Cyan

    if ($build) {
        Write-Host "Building all images..." -ForegroundColor Yellow
        docker-compose build --parallel
    }

    Write-Host "Starting services..." -ForegroundColor Yellow
    docker-compose up -d

    Write-Host ""
    Write-Host "Services started! Waiting for health checks..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10

    # Check service health
    Write-Host ""
    Write-Host "Service Status:" -ForegroundColor Cyan
    Write-Host "---------------"

    $services = @(
        @{Name="API Server"; URL="http://localhost:8000/health"},
        @{Name="OWLBAN GROUP"; URL="http://localhost:3000/api/subscription/plans"},
        @{Name="OSCAR BROOME"; URL="http://localhost:3001/"},
        @{Name="BLACKBOX AI"; URL="http://localhost:3002/"}
    )

    foreach ($svc in $services) {
        try {
            $response = Invoke-WebRequest -Uri $svc.URL -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "  [UP] $($svc.Name)" -ForegroundColor Green
            } else {
                Write-Host "  [??] $($svc.Name) (Status: $($response.StatusCode))" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  [DOWN] $($svc.Name)" -ForegroundColor Red
        }
    }

    Write-Host ""
    Write-Host "Access Points:" -ForegroundColor Cyan
    Write-Host "  API Server:        http://localhost:8000"
    Write-Host "  OWLBAN GROUP:      http://localhost:3000"
    Write-Host "  OSCAR BROOME:      http://localhost:3001"
    Write-Host "  BLACKBOX AI:       http://localhost:3002"
    Write-Host "  Web Dashboard:     http://localhost:8501"
    Write-Host "  Grafana:           http://localhost:3003"
    Write-Host ""
    Write-Host "Run '.\scripts\deploy.ps1 -initUsers' to seed default users." -ForegroundColor Cyan
}

function Initialize-Users {
    Write-Host "Seeding default users..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe scripts/init_users.py
}

function Show-Logs {
    docker-compose logs -f --tail=100
}

# Main execution
if ($help) {
    Show-Help
    exit 0
}

if ($stop) {
    Stop-Services
    exit 0
}

if ($logs) {
    Show-Logs
    exit 0
}

# Default: deploy
Start-Services

if ($initUsers) {
    Start-Sleep -Seconds 5
    Initialize-Users
}

Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor Green
