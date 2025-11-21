@echo off
cd /d "c:\Users\bizle\Desktop\jpmorgan_financial_apis"
powershell.exe -ExecutionPolicy Bypass -NoProfile -File ".\scripts\verify_production_readiness.ps1"
pause
