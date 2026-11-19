[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$PublicIp,
    [string]$Domain = "owlbangroup.com",
    [string]$ApiKey = $env:GODADDY_API_KEY,
    [string]$ApiSecret = $env:GODADDY_API_SECRET
)

$ErrorActionPreference = "Stop"

if (-not [System.Net.IPAddress]::TryParse($PublicIp, [ref]$null)) {
    throw "PublicIp must be a valid IPv4 or IPv6 address."
}

if ([string]::IsNullOrWhiteSpace($ApiKey) -or [string]::IsNullOrWhiteSpace($ApiSecret)) {
    throw "Set GODADDY_API_KEY and GODADDY_API_SECRET environment variables."
}

$headers = @{
    Authorization = "sso-key $ApiKey`:$ApiSecret"
    Accept = "application/json"
    "Content-Type" = "application/json"
}

foreach ($name in @("@", "www")) {
    $uri = "https://api.godaddy.com/v1/domains/$Domain/records/A/$name"
    $body = ConvertTo-Json @(
        @{
            data = $PublicIp
            ttl = 600
        }
    )
    Invoke-RestMethod -Method Put -Uri $uri -Headers $headers -Body $body | Out-Null
    Write-Host "Configured $name.$Domain -> $PublicIp"
}
