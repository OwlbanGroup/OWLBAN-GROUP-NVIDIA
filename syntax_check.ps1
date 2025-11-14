# PowerShell script to perform syntax check on app.py
cd jpmorgan_financial_apis
python -m py_compile app.py
if ($LASTEXITCODE -eq 0) {
    Write-Output "Syntax check passed"
} else {
    Write-Output "Syntax check failed"
}
