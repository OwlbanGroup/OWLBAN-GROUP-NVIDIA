# Deployment script for JPMorgan Financial APIs
# This script builds and runs the application using Docker

Write-Host "Starting deployment of JPMorgan Financial APIs..."

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH. Please install Docker and try again."
    exit 1
}

# Navigate to the project directory
Set-Location -Path "jpmorgan_financial_apis"

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t jpmorgan-financial-apis .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build Docker image."
    exit 1
}

# Run docker-compose to start all services
Write-Host "Starting services with docker-compose..."
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start services."
    exit 1
}

Write-Host "Deployment completed successfully!"
Write-Host "The application should be running on http://localhost:8080"
Write-Host "Monitoring services (Prometheus, Grafana) should also be available."
