@echo off
REM JPMorgan Financial APIs - Production Deployment Fix Script
REM Handles Windows PowerShell limitations

echo Starting deployment fix...

REM Step 1: Navigate and cleanup
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis"
echo Cleaning up existing containers...
docker compose down --volumes --remove-orphans --timeout 30

REM Step 2: Check port 8000
echo Checking port 8000...
netstat -ano | findstr :8000
echo If process shown (not Docker), kill manually if needed.

REM Step 3: Pull images and build
echo Pulling latest images...
docker compose pull

echo Building API image...
docker compose build jpmorgan-api

REM Step 4: Start core services (skip SSL profile for dev)
echo Starting services ^(DEV mode, no certbot)...
docker compose --env-file .env up -d postgres redis jpmorgan-api nginx prometheus grafana alertmanager

REM Step 5: Wait for health
echo Waiting for services to be healthy...
timeout /t 30 /nobreak > nul

REM Step 6: Status check
echo Container status:
docker compose ps

echo API logs ^(last 20 lines^):
docker compose logs --tail=20 jpmorgan-api

echo Nginx logs:
docker compose logs --tail=10 nginx

REM Step 7: Health tests
echo Testing health endpoints...
curl -k http://localhost/health || echo "Nginx health OK? Check logs"
curl -k http://127.0.0.1:8000/health || echo "API direct health OK?"
curl -k http://localhost:3000 || echo "Grafana at http://localhost:3000 ^(admin/admin^)"
echo "Prometheus: http://localhost:9090"

echo.
echo ========================================
echo Deployment complete! Check:
echo - API: http://localhost/
echo - Direct API: http://127.0.0.1:8000/
echo - Grafana: http://localhost:3000 ^(admin/admin^)
echo - Prometheus: http://localhost:9090
echo ========================================

