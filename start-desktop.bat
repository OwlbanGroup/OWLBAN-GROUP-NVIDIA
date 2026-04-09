@echo off
echo Starting JPMorgan Financial APIs in Docker Desktop GUI...
cd /d "C:\Users\bizle\Desktop\jpmorgan_financial_apis"
call make up
echo Opening Docker Desktop and endpoints...
start "" "docker desktop://"
timeout /t 5 /nobreak >nul
start "" "http://localhost:8000/docs"
start "" "http://localhost:8080/mcp/tools"
start "" "http://localhost:11434/api/tags"
echo ✅ Ready! Manage stack in Docker Desktop ^> Compose ^> jpm-finance-gui
pause
