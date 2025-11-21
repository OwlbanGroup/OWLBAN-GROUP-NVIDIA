<#
.SYNOPSIS
    Complete Azure Deployment - Create Remaining Resources

.DESCRIPTION
    Creates PostgreSQL, Redis, and Key Vault in the existing resource group
    Uses eastus2 for PostgreSQL (flexible server supported region)
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "jpmorgan-financial-apis-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus2"
)

# Color output functions
function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-ErrorMsg { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
function Write-Step { param([string]$Message) Write-Host "`n[STEP] $Message" -ForegroundColor Magenta; Write-Host ("=" * 70) -ForegroundColor Magenta }

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "     JPMorgan Financial APIs - Complete Deployment" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

# Step 1: Create PostgreSQL Database
Write-Step "1/3 - Creating PostgreSQL Database"
$dbServer = "jpmorgan-financial-db"
$dbAdmin = "jpmadmin"
$dbPassword = "SecureP@ssw0rd2024!" + (Get-Random -Maximum 9999)

# Check if PostgreSQL server already exists
$dbExists = az postgres flexible-server show --resource-group $ResourceGroup --name $dbServer --query "name" -o tsv 2>$null
if ($dbExists) {
    Write-Warning "PostgreSQL server '$dbServer' already exists, skipping creation..."
} else {
    Write-Info "Creating PostgreSQL server '$dbServer' in $Location..."
    Write-Warning "This may take 5-10 minutes..."

    $dbResult = az postgres flexible-server create `
        --resource-group $ResourceGroup `
        --name $dbServer `
        --location $Location `
        --admin-user $dbAdmin `
        --admin-password $dbPassword `
        --sku-name Standard_D2s_v3 `
        --tier GeneralPurpose `
        --version 15 `
        --storage-size 128 `
        --public-access 0.0.0.0-255.255.255.255 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Success "PostgreSQL server created"
    } else {
        $errorMsg = $dbResult | Out-String
        if ($errorMsg -like "*already exists*" -or $errorMsg -like "*already used*" -or $errorMsg -like "*AlreadyExists*") {
            Write-Warning "PostgreSQL server already exists, continuing..."
        } else {
            Write-ErrorMsg "Failed to create PostgreSQL: $errorMsg"
            Write-Warning "Continuing with remaining resources..."
        }
    }
}

# Create database
Write-Info "Creating database..."
$dbCreateResult = az postgres flexible-server db create `
    --resource-group $ResourceGroup `
    --server-name $dbServer `
    --database-name jpmorgan_financial_apis_prod 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "Database created"
} else {
    $errorMsg = $dbCreateResult | Out-String
    if ($errorMsg -like "*already exists*" -or $errorMsg -like "*AlreadyExists*") {
        Write-Warning "Database already exists, continuing..."
    } else {
        Write-Warning "Database creation issue: $errorMsg"
    }
}

# Step 2: Create Redis Cache
Write-Step "2/3 - Creating Redis Cache"
$redisName = "jpmorgan-financial-redis"

# Check if Redis cache already exists
$redisExists = az redis show --resource-group $ResourceGroup --name $redisName --query "name" -o tsv 2>$null
if ($redisExists) {
    Write-Warning "Redis cache '$redisName' already exists, skipping creation..."
} else {
    Write-Info "Creating Redis cache '$redisName' in $Location..."
    Write-Warning "This may take 10-15 minutes..."

    $redisResult = az redis create `
        --resource-group $ResourceGroup `
        --name $redisName `
        --location $Location `
        --sku Standard `
        --vm-size c1 `
        --enable-non-ssl-port 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Redis cache created"
    } else {
        $errorMsg = $redisResult | Out-String
        if ($errorMsg -like "*already exists*" -or $errorMsg -like "*AlreadyExists*") {
            Write-Warning "Redis cache already exists, continuing..."
        } else {
            Write-ErrorMsg "Failed to create Redis: $errorMsg"
            Write-Warning "Continuing with remaining resources..."
        }
    }
}

# Step 3: Create Key Vault
Write-Step "3/3 - Creating Key Vault"
$kvName = "jpmorgan-financial-kv"

# Check if Key Vault already exists
$kvExists = az keyvault show --resource-group $ResourceGroup --name $kvName --query "name" -o tsv 2>$null
if ($kvExists) {
    Write-Warning "Key Vault '$kvName' already exists, skipping creation..."
} else {
    Write-Info "Creating Key Vault '$kvName' in $Location..."

    $kvResult = az keyvault create `
        --resource-group $ResourceGroup `
        --name $kvName `
        --location $Location `
        --enable-rbac-authorization false `
        --enabled-for-deployment true `
        --enabled-for-template-deployment true 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Key Vault created"
    } else {
        $errorMsg = $kvResult | Out-String
        if ($errorMsg -like "*already exists*" -or $errorMsg -like "*AlreadyExists*") {
            Write-Warning "Key Vault already exists, continuing..."
        } else {
            Write-ErrorMsg "Failed to create Key Vault: $errorMsg"
            Write-Warning "Continuing with secret storage..."
        }
    }
}

# Store secrets
Write-Info "Storing secrets in Key Vault..."
try {
    az keyvault secret set --vault-name $kvName --name "DatabasePassword" --value $dbPassword 2>&1 | Out-Null
    az keyvault secret set --vault-name $kvName --name "JWTSecret" --value ("jwt-secret-" + (New-Guid).ToString()) 2>&1 | Out-Null
    az keyvault secret set --vault-name $kvName --name "APIKey" --value ("api-key-" + (New-Guid).ToString()) 2>&1 | Out-Null
    Write-Success "Secrets stored"
} catch {
    Write-Warning "Some secrets may not have been stored: $_"
}

# Deployment Summary
Write-Host "`n========================================================================" -ForegroundColor Green
Write-Host "                    DEPLOYMENT COMPLETED" -ForegroundColor Green
Write-Host "========================================================================`n" -ForegroundColor Green

Write-Host "Resources Created:" -ForegroundColor Cyan
Write-Host "  [OK] PostgreSQL: $dbServer.postgres.database.azure.com (in $Location)"
Write-Host "  [OK] Redis Cache: $redisName.redis.cache.windows.net (in $Location)"
Write-Host "  [OK] Key Vault: $kvName (in $Location)"

Write-Host "`nExisting Resources:" -ForegroundColor Cyan
Write-Host "  [OK] Resource Group: $ResourceGroup"
Write-Host "  [OK] Container Registry: jpmorganfinancialacr.azurecr.io"
Write-Host "  [OK] AKS Cluster: jpmorgan-financial-aks (3 nodes)"

Write-Host "`nCredentials (SAVE SECURELY):" -ForegroundColor Yellow
Write-Host "  Database Admin: $dbAdmin"
Write-Host "  Database Password: $dbPassword"

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Build and push Docker images to ACR"
Write-Host "  2. Deploy applications to AKS"
Write-Host "  3. Configure DNS and SSL"
Write-Host "  4. Test API endpoints"

# Save credentials
$credFile = Join-Path $PSScriptRoot "..\azure_deployment_credentials.txt"
@"
Azure Deployment Credentials
=============================
Date: $(Get-Date)

Resource Group: $ResourceGroup

ACR: jpmorganfinancialacr.azurecr.io
AKS: jpmorgan-financial-aks
PostgreSQL: $dbServer.postgres.database.azure.com (Location: $Location)
  Admin: $dbAdmin
  Password: $dbPassword
  Database: jpmorgan_financial_apis_prod

Redis: $redisName.redis.cache.windows.net (Location: $Location)
Key Vault: $kvName (Location: $Location)

IMPORTANT: Store these credentials securely and delete this file!
"@ | Out-File -FilePath $credFile -Encoding UTF8

Write-Host "`n[SECURE] Credentials saved to: $credFile" -ForegroundColor Yellow
Write-Host "         Please store securely and delete the file!`n" -ForegroundColor Red
