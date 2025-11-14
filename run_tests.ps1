# Script to run unit tests for JPMorgan Financial APIs

Write-Host "Running unit tests..."

# Navigate to the project directory
Set-Location -Path "jpmorgan_financial_apis"

# Run the unit tests
python -m unittest test_security.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "Unit tests failed."
    exit 1
}

Write-Host "Unit tests completed successfully!"
