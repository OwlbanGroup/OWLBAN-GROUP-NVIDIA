@echo off
echo ========================================
echo AZURE ACCOUNT SETUP
echo ========================================
echo.
echo This will run the Azure account setup script.
echo.
echo What it does:
echo 1. Check Azure CLI installation
echo 2. Guide you through account creation
echo 3. Help with Azure login
echo 4. Register resource providers
echo 5. Create service principal
echo 6. Run verification tests
echo.
echo Time required: 30-40 minutes
echo.
pause

cd /d "%~dp0scripts"
powershell.exe -ExecutionPolicy Bypass -File "setup_azure_account.ps1"

echo.
echo ========================================
echo Setup script completed!
echo ========================================
echo.
pause
