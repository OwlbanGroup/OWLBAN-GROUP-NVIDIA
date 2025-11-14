# PowerShell script to set required environment variables for JPMorgan Financial APIs
# Run this script before running tests or the application

# JPMorgan API Credentials (provided by user)
$env:TOKEN_URL = "https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token"
$env:TOKEN_CLIENT_ID = "0369026e-0d67-4454-8a13-a0129a5cd3f6"
$env:TOKEN_CLIENT_SECRET = "piAKagzhmiQFFnGbdwvDkCz0mvdC1IBGIzdYl6bLch-vegBy4HmhXNATJwLNFfmGYlWeIDH3eHTF6q0KNcJoqg"

# Application Security
$env:SECRET_KEY = "your-secret-key-for-session-security-change-this-in-production"

# Allow missing tokens for testing
$env:ALLOW_MISSING_TOKENS = "true"

# Optional: NGC and GPU settings (leave as defaults if not needed)
$env:NGC_API_KEY = ""
$env:NVIDIA_VISIBLE_DEVICES = "all"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:GPU_MEMORY_FRACTION = "0.8"

# Logging and other settings
$env:LOG_LEVEL = "INFO"
$env:TELEMETRY_ENABLED = "true"

Write-Host "Environment variables set successfully!"
Write-Host "TOKEN_CLIENT_ID: $env:TOKEN_CLIENT_ID"
Write-Host "SECRET_KEY: $env:SECRET_KEY"
Write-Host "ALLOW_MISSING_TOKENS: $env:ALLOW_MISSING_TOKENS"
