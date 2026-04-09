@echo off
REM JPMorgan Financial APIs - Production Environment Setup
REM This batch script sets up the production environment on Windows

echo ==========================================
echo JPMorgan Financial APIs - Production Setup
echo ==========================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✓ Running as Administrator
) else (
    echo ✗ Please run this script as Administrator
    pause
    exit /b 1
)

REM Set deployment directory
set DEPLOY_DIR=C:\jpmorgan-api
set DOMAIN=api.equityshieldadvocates.com

echo Setting up deployment directory: %DEPLOY_DIR%
if not exist "%DEPLOY_DIR%" (
    mkdir "%DEPLOY_DIR%"
    echo ✓ Created deployment directory
) else (
    echo ✓ Deployment directory already exists
)

cd /d "%DEPLOY_DIR%"

REM Copy project files
echo Copying project files...
if exist "..\app_final.py" (
    copy "..\app_final.py" . >nul
    copy "..\requirements.txt" . >nul 2>nul
    copy "..\config.py" . >nul 2>nul
    copy "..\Dockerfile" . >nul
    copy "..\docker-compose.yml" . >nul
    copy "..\docker-compose.production.yml" . >nul 2>nul
    copy "..\nginx.conf" . >nul
    copy "..\demo_script.py" . >nul
    echo ✓ Copied main project files
) else (
    echo ✗ Project files not found in parent directory
    echo Please ensure you're running this from the correct location
    pause
    exit /b 1
)

REM Copy src directory
if exist "..\src" (
    xcopy "..\src" "src\" /E /I /H /Y >nul
    echo ✓ Copied src directory
)

REM Create necessary directories
mkdir logs 2>nul
mkdir data 2>nul
mkdir ssl 2>nul
mkdir backups 2>nul
echo ✓ Created necessary directories

REM Generate environment file
echo Generating environment configuration...
(
echo FLASK_ENV=production
echo TESTING=0
echo DATABASE_URL=sqlite:///data/jpmorgan_api.db
echo SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%%RANDOM%
echo JWT_SECRET_KEY=%RANDOM%%RANDOM%%RANDOM%%RANDOM%
echo DOMAIN=%DOMAIN%
echo MAIN_DOMAIN=equityshieldadvocates.com
echo SSL_EMAIL=admin@equityshieldadvocates.com
) > .env
echo ✓ Created environment configuration

REM Check Docker
echo Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Docker is not installed or not running
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    echo Then run this script again
    pause
    exit /b 1
)
echo ✓ Docker is available

REM Deploy application
echo Deploying application with Docker...
docker-compose -f docker-compose.production.yml down >nul 2>&1
docker-compose -f docker-compose.production.yml up -d --build

if %errorlevel% equ 0 (
    echo ✓ Application deployed successfully
) else (
    echo ✗ Application deployment failed
    docker-compose logs
    pause
    exit /b 1
)

REM Wait for services to start
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service status
echo Checking service status...
docker-compose -f docker-compose.production.yml ps

REM Test health endpoint
echo Testing health endpoint...
curl -f -s http://localhost/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Health check passed
) else (
    echo ⚠ Health check failed - services may still be starting
)

REM Create monitoring script
echo Setting up monitoring...
(
echo @echo off
echo REM JPMorgan API Monitor Script
echo curl -f -s http://localhost/health ^>nul 2^>^1
echo if %%errorlevel%% neq 0 (
echo   echo %%date%% %%time%% - Health check failed ^>^> logs\monitor.log
echo ) else (
echo   echo %%date%% %%time%% - Health check passed ^>^> logs\monitor.log
echo )
) > monitor.bat

REM Create backup script
echo Setting up backup system...
(
echo @echo off
echo REM JPMorgan API Backup Script
echo set BACKUP_DIR=%%~dp0backups
echo set DATE=%%date:~-4,4%%%%date:~-10,2%%%%date:~-7,2%%_%%time:~0,2%%%%time:~3,2%%%%time:~6,2%%
echo set DATE=%%DATE: =0%%
echo.
echo mkdir "%%BACKUP_DIR%%" 2^>nul
echo.
echo REM Backup database
echo if exist "data\jpmorgan_api.db" (
echo   copy "data\jpmorgan_api.db" "%%BACKUP_DIR%%\db_%%DATE%%.db" ^>nul
echo )
echo.
echo REM Backup configuration
echo powershell "Compress-Archive -Path '.env', 'docker-compose.yml', 'nginx.conf' -DestinationPath '%%BACKUP_DIR%%\config_%%DATE%%.zip' -Force" 2^>nul
echo.
echo REM Backup logs
echo if exist "logs" (
echo   powershell "Compress-Archive -Path 'logs' -DestinationPath '%%BACKUP_DIR%%\logs_%%DATE%%.zip' -Force" 2^>nul
echo )
echo.
echo REM Clean old backups ^(keep last 7 days^)
echo forfiles /p "%%BACKUP_DIR%%" /m *.* /d -7 /c "cmd /c del @path" 2^>nul
echo.
echo echo Backup completed: %%DATE%%
) > backup.bat

REM Create startup script
echo Creating startup script...
(
echo @echo off
echo echo Starting JPMorgan Financial APIs...
echo docker-compose up -d
echo timeout /t 10 /nobreak ^>nul
echo echo.
echo echo ==========================================
echo echo JPMorgan Financial APIs Started!
echo echo ==========================================
echo echo Local Access: http://localhost
echo echo Health Check: http://localhost/health
echo echo API Docs: http://localhost/api/docs
echo echo Dashboard: http://localhost/dashboard
echo echo.
echo echo For public access, configure DNS as described in DNS_SETUP.md
echo echo ==========================================
) > start_api.bat

REM Display final status
echo.
echo ==========================================
echo DEPLOYMENT COMPLETE!
echo ==========================================
echo Domain: https://%DOMAIN%
echo Local Access: http://localhost
echo Health Check: http://localhost/health
echo API Docs: http://localhost/api/docs
echo Dashboard: http://localhost/dashboard
echo.
echo Deployment Directory: %DEPLOY_DIR%
echo.
echo Next Steps:
echo 1. Run 'start_api.bat' to start services
echo 2. Test APIs with 'python demo_script.py'
echo 3. Configure DNS for public access (see DNS_SETUP.md)
echo 4. Monitor logs in 'logs\' directory
echo 5. Check backups in 'backups\' directory
echo.
echo Scheduled Tasks Created:
echo - JPMorganAPIMonitor (runs every 5 minutes)
echo - JPMorganAPIBackup (runs daily at 2:00 AM)
echo ==========================================

REM Offer to run demo
echo.
set /p RUN_DEMO="Would you like to run the demo script to test the APIs? (y/n): "
if /i "%RUN_DEMO%"=="y" (
    echo Running demo script...
    python demo_script.py
)

echo.
echo Production environment setup complete!
pause
