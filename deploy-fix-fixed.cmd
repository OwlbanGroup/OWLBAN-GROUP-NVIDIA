@echo off
REM JPMorgan Financial APIs - Production Deployment Fix Script - CMD Compatible
REM Hardened for Windows cmd.exe

echo Starting deployment fix...

REM Step 1: Navigate and cleanup
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis"
echo Cleaning up existing containers...
docker compose -f docker-compose.production.yml down --volumes --remove-orphans --timeout 30

REM Step 2: Check port 8000
echo Checking port 8000...
netstat -ano | findstr :8000
echo If process shown ^(not Docker^), kill manually: taskkill /PID ^<pid^> /F

REM Step 3: Pull images and build
echo Pulling latest images...
docker compose -f docker-compose.production.yml pull

echo Building API image...
docker compose -f docker-compose.production.yml build jpmorgan-api

REM Step 4: Start core services ^(skip certbot for local^)
echo Starting services ^(production mode^)...
docker compose -f docker-compose.production.yml --env-file .env.production up -d postgres redis jpmorgan-api nginx prometheus grafana alertmanager

REM Step 5: Wait 30s for startup
echo Waiting 30s for services to start...
ping -n 31 127.0.0.1 ^>nul

REM Step 6: Status check
echo Container status:
docker compose -f docker-compose.production.yml ps

echo API logs ^(last 20 lines^):
docker compose -f docker-compose.production.yml logs --tail=20 jpmorgan-api

echo Nginx logs ^(last 10 lines^):
docker compose -f docker-compose.production.yml logs --tail=10 nginx

REM Step 7: Health tests
echo Testing health endpoints...
curl -k http://localhost/health
if errorlevel 1 echo Nginx health check failed - check logs above
curl -k http://127.0.0.1:8000/health
if errorlevel 1 echo API direct health check failed - check logs above
curl -k http://localhost:3000
if errorlevel 1 echo Grafana access test ^(port 3000^)
echo Prometheus available at: http://localhost:9090

echo.
echo ========================================
echo Deployment complete! Access points:
echo - Main API ^(nginx^): http://localhost/
echo - Direct API: http://127.0.0.1:8000/
echo - Grafana: http://localhost:3000 ^(admin/admin^)
echo - Prometheus: http://localhost:9090
echo ========================================

REM Keep window open
pause
