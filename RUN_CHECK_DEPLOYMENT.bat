@echo off
echo ========================================================================
echo      JPMorgan Financial APIs - Check Deployment Status
echo ========================================================================
echo.

cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -Command "Set-Location '%~dp0'; & '.\scripts\check_deployment_status.ps1'"

echo.
echo ========================================================================
echo                    Status Check Complete
echo ========================================================================
echo.
echo If any resources show as "Not found" or "Creating":
echo   - Wait 5-10 minutes for provisioning to complete
echo   - Run this script again to check updated status
echo   - Redis typically takes 10-15 minutes to provision
echo.
pause
