<#
.SYNOPSIS
    Check Azure Deployment Status

.DESCRIPTION
    Monitors the status of all deployed Azure resources
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "jpmorgan-financial-apis-rg"
)

function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-ErrorMsg { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "     Azure Deployment Status Check" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

Write-Info "Checking all resources in resource group: $ResourceGroup"
Write-Host ""

# Check all resources
Write-Host "All Resources:" -ForegroundColor Yellow
az resource list --resource-group $ResourceGroup --output table

Write-Host "`n"

# Check AKS
Write-Host "AKS Cluster Status:" -ForegroundColor Yellow
$aksStatus = az aks show --resource-group $ResourceGroup --name jpmorgan-financial-aks --query "provisioningState" -o tsv 2>$null
if ($aksStatus -eq "Succeeded") {
    Write-Success "AKS Cluster: Running"
} else {
    Write-Warning "AKS Cluster: $aksStatus"
}

# Check PostgreSQL
Write-Host "`nPostgreSQL Status:" -ForegroundColor Yellow
$dbStatus = az postgres flexible-server show --resource-group $ResourceGroup --name jpmorgan-financial-db --query "state" -o tsv 2>$null
if ($dbStatus -eq "Ready") {
    Write-Success "PostgreSQL: Ready"
} elseif ($dbStatus) {
    Write-Warning "PostgreSQL: $dbStatus"
} else {
    Write-ErrorMsg "PostgreSQL: Not found or still creating"
}

# Check Redis
Write-Host "`nRedis Cache Status:" -ForegroundColor Yellow
$redisStatus = az redis show --resource-group $ResourceGroup --name jpmorgan-financial-redis --query "provisioningState" -o tsv 2>$null
if ($redisStatus -eq "Succeeded") {
    Write-Success "Redis Cache: Running"
} elseif ($redisStatus) {
    Write-Warning "Redis Cache: $redisStatus"
} else {
    Write-ErrorMsg "Redis Cache: Not found or still creating"
}

# Check Key Vault
Write-Host "`nKey Vault Status:" -ForegroundColor Yellow
$kvStatus = az keyvault show --resource-group $ResourceGroup --name jpmorgan-financial-kv --query "properties.provisioningState" -o tsv 2>$null
if ($kvStatus -eq "Succeeded") {
    Write-Success "Key Vault: Active"
} elseif ($kvStatus) {
    Write-Warning "Key Vault: $kvStatus"
} else {
    Write-ErrorMsg "Key Vault: Not found or still creating"
}

# Check ACR
Write-Host "`nContainer Registry Status:" -ForegroundColor Yellow
$acrStatus = az acr show --resource-group $ResourceGroup --name jpmorganfinancialacr --query "provisioningState" -o tsv 2>$null
if ($acrStatus -eq "Succeeded") {
    Write-Success "ACR: Running"
} else {
    Write-Warning "ACR: $acrStatus"
}

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "                    Status Check Complete" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan
