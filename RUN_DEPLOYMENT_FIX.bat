@echo off
echo ========================================================================
echo      JPMorgan Financial APIs - Azure Deployment Fix
echo ========================================================================
echo.
echo This script will create the missing Azure resources:
echo   - Redis Cache (jpmorgan-financial-redis)
echo   - Key Vault (jpmorgan-financial-kv)
echo.
echo Expected Duration: 10-15 minutes
echo.
echo Press any key to start the deployment fix...
pause >nul

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -Command "Set-Location '%~dp0'; & '.\scripts\fix_remaining_deployment.ps1'"

echo.
echo ========================================================================
echo                    Deployment Fix Complete
echo ========================================================================
echo.
echo Next Steps:
echo   1. Review the output above for any errors
echo   2. Run RUN_CHECK_DEPLOYMENT.bat to verify all resources
echo   3. Wait 5-10 minutes if Redis is still provisioning
echo.
pause
