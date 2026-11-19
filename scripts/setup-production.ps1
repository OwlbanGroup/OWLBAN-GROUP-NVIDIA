[CmdletBinding()]
param(
    [switch]$Force,
    [string]$CorsOrigin = "https://app.example.com",
    [string]$Domain = "localhost"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env.production"

if ((Test-Path $envFile) -and -not $Force) {
    throw "Existing .env.production found. Use -Force to replace it."
}

$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$apiPasswordBytes = New-Object byte[] 36
$jwtSecretBytes = New-Object byte[] 48
$rng.GetBytes($apiPasswordBytes)
$rng.GetBytes($jwtSecretBytes)
$rng.Dispose()
$apiPassword = [Convert]::ToBase64String($apiPasswordBytes)
$jwtSecret = [Convert]::ToBase64String($jwtSecretBytes)

$content = @"
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
API_USERNAME=owlban_service
API_PASSWORD=$apiPassword
JWT_SECRET=$jwtSecret
CORS_ORIGINS=$CorsOrigin
APP_API_KEYS={}
APP_DB_PATH=/data/owlban_app.db
RATE_LIMIT_PER_MINUTE=60
"@

Set-Content -Path $envFile -Value $content -Encoding ascii

$tlsDir = Join-Path $root "deploy\tls"
$fullchain = Join-Path $tlsDir "fullchain.pem"
$privkey = Join-Path $tlsDir "privkey.pem"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

if ($Force -or -not (Test-Path $fullchain) -or -not (Test-Path $privkey)) {
    & $python -m pip install cryptography --disable-pip-version-check | Out-Null
    & $python (Join-Path $root "scripts\generate-tls-certificate.py") `
        --domain $Domain `
        --output-dir $tlsDir
    Write-Host "Self-signed development certificate generated. Replace it with a trusted certificate before public release."
} else {
    Write-Host "Existing TLS certificate files retained."
}

Write-Host "Created $envFile"
