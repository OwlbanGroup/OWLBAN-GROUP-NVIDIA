# Start Flask application with audit logging enabled
$env:TESTING = "1"
$env:AUDIT_LOG_ENABLED = "true"
$env:FLASK_RUN_PORT = "8000"

Write-Host "🚀 Starting Flask Application with Audit Logging..." -ForegroundColor Green
Write-Host "Environment Variables:" -ForegroundColor Cyan
Write-Host "  TESTING = $env:TESTING" -ForegroundColor Yellow
Write-Host "  AUDIT_LOG_ENABLED = $env:AUDIT_LOG_ENABLED" -ForegroundColor Yellow
Write-Host "  FLASK_RUN_PORT = $env:FLASK_RUN_PORT" -ForegroundColor Yellow
Write-Host ""

cd C:\Users\bizle\Desktop\jpmorgan_financial_apis
python app_final.py
