# PowerShell script to test JPMorgan Financial APIs
# Navigate to the project directory
cd jpmorgan_financial_apis

# Generate token using TokenManager
python -c "from src.token_manager import TokenManager; from config import config; tm = TokenManager(config.TOKEN_CLIENT_ID, config.TOKEN_CLIENT_SECRET, config.TOKEN_URL, config.TOKEN_SCOPE); print(tm.generate_token())" > token.txt

# Start the Flask app in the background
Start-Process python -ArgumentList "app.py" -NoNewWindow

# Wait for 5 seconds
Start-Sleep -Seconds 5

# Read the token
$TOKEN = Get-Content token.txt

# Test endpoints
Write-Host "Testing /health endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/health" -Method GET

Write-Host "Testing /telemetry/metrics endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/telemetry/metrics" -Method GET

Write-Host "Testing /telemetry endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/telemetry" -Method POST -Headers @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
} -Body '{"test": "data"}'

Write-Host "Testing /telemetry/batch endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/telemetry/batch" -Method POST -Headers @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
} -Body '{"telemetry_data": [{"test": "data1"}, {"test": "data2"}]}'

Write-Host "Testing /ml/anomalies endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/ml/anomalies" -Method POST -Headers @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
} -Body '{"telemetry_data": [{"test": "data"}]}'

Write-Host "Testing /ml/train endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/ml/train" -Method POST -Headers @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
} -Body '{"telemetry_data": [{"test": "data"}]}'

Write-Host "Testing /telemetry/export endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/telemetry/export" -Method GET -Headers @{
    "Authorization" = "Bearer $TOKEN"
}

Write-Host "Testing /telemetry with invalid JSON..."
Invoke-WebRequest -Uri "http://localhost:5000/telemetry" -Method POST -Headers @{
    "Content-Type" = "application/json"
} -Body "invalid json"

Write-Host "Testing non-existent endpoint..."
Invoke-WebRequest -Uri "http://localhost:5000/nonexistent" -Method GET

# Kill Python processes
Stop-Process -Name python -Force

# Delete token file
Remove-Item token.txt

Write-Host "Testing completed."
