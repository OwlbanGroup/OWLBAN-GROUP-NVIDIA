# JPMorgan Financial APIs Production Deployment Script (PowerShell)

param(
    [switch]$Build = $false,
    [switch]$Force = $false,
    [string]$ComposeFile = 'docker-compose.production.yml'
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $MyInvocation.MyCommand.Path -Parent
if ($projectDir -eq '') { $projectDir = (Get-Location).Path }
if ((Get-Location).Path -ne $projectDir) {
    Write-Host "Auto CD to project dir: $projectDir" -ForegroundColor Cyan
    Set-Location $projectDir
}

Write-Host '=== JPMorgan Financial APIs Production Deployment ===' -ForegroundColor Green
Write-Host "Project: $projectDir" -ForegroundColor Cyan
Write-Host "Compose: $ComposeFile" -ForegroundColor Cyan

if (!(Get-Command docker -ErrorAction SilentlyContinue)) { 
    Write-Error 'Docker not found. Install from https://docker.com'; exit 1 
}
if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) { 
    Write-Error 'docker-compose not found. Install Docker Compose v1 or enable v2.'; exit 1 
}
Write-Host 'Docker OK' -ForegroundColor Green

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

if (!(Test-Path '.env.production')) { 
    Write-Warning '.env.production not found. Create from .env.example or set env vars manually.'
}

Write-Host "Build images: $Build" -ForegroundColor Cyan
Write-Host "Force recreate: $Force" -ForegroundColor Cyan

Write-Host '[1/6] Stop existing services...' -ForegroundColor Yellow
docker-compose -f $ComposeFile down --remove-orphans
if ($Force) { docker-compose -f $ComposeFile down -v }

Write-Host '[2/6] Prune unused resources...' -ForegroundColor Yellow
docker system prune -f

Write-Host '[3/6] Build/pull images...' -ForegroundColor Yellow
if ($Build) {
    docker-compose -f $ComposeFile build --no-cache
} else {
    docker-compose -f $ComposeFile pull
}

Write-Host '[4/6] Start services...' -ForegroundColor Yellow
docker-compose -f $ComposeFile up -d

Write-Host '[5/6] Wait for health checks...' -ForegroundColor Yellow
Start-Sleep -Seconds 30
Write-Host '[Health retry loop...]'
1..10 | ForEach-Object {
  docker-compose -f $ComposeFile ps
  Start-Sleep -Seconds 5
}

Write-Host '[6/6] Health checks...' -ForegroundColor Yellow
$projectName = (Split-Path $projectDir -Leaf)
$services = @('nginx', 'jpmorgan-api', 'postgres', 'redis', 'prometheus', 'alertmanager', 'grafana')
foreach ($service in $services) {
    $containerName = "${projectName}_${service}_1"
    $statusRaw = docker inspect --format='{{.State.Health.Status}}' $containerName 2>$null | Select-Object -First 1
    $status = if ($statusRaw) { $statusRaw.Trim() } else { 'unknown' }
    if ($status -eq 'healthy') {
        Write-Host ('  ✓ ' + $service + ': healthy') -ForegroundColor Green
    } else {
        Write-Host ('  ! ' + $service + ': ' + $status) -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host 'Deployment complete!' -ForegroundColor Green
Write-Host 'Access:' -ForegroundColor Cyan
Write-Host '  API: http://localhost'
Write-Host '  Grafana: http://localhost:3000 (admin/admin)'
Write-Host '  Prometheus: http://localhost:9090'

docker-compose -f $ComposeFile logs --tail=50 -t

