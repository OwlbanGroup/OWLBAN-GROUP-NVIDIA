# PowerShell launcher for Docker Desktop GUI
Write-Host "🚀 Starting JPMorgan Financial APIs Docker Desktop GUI..." -ForegroundColor Green
Set-Location "C:\Users\bizle\Desktop\jpmorgan_financial_apis"
& make up
Write-Host "Opening Docker Desktop and endpoints..." -ForegroundColor Yellow
Start-Process "docker desktop://"
Start-Sleep -Seconds 5
Start-Process "http://localhost:8000/docs"
Start-Process "http://localhost:8080/mcp/tools"
Start-Process "http://localhost:11434/api/tags"
Write-Host "✅ Ready! Docker Desktop > Compose > jpm-finance-gui" -ForegroundColor Green
Read-Host "Press Enter to exit"
