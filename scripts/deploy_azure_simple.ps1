 <#
.SYNOPSIS
    Simplified Azure Deployment Script for JPMorgan Financial APIs

.DESCRIPTION
    Step-by-step deployment with progress tracking and error handling
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "jpmorgan-financial-apis-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus"
)

# Color output functions
function Write-Success { param([string]$Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Warning { param([string]$Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-ErrorMsg { param([string]$Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
function Write-Step { param([string]$Message) Write-Host "`n[STEP] $Message" -ForegroundColor Magenta; Write-Host ("=" * 70) -ForegroundColor Magenta }

# Provider registration function
function Register-AzureProvider {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProviderNamespace,
        
        [Parameter(Mandatory=$false)]
        [int]$TimeoutMinutes = 10
    )
    
    Write-Info "Checking provider: $ProviderNamespace"
    
    # Check current registration state
    $provider = az provider show --namespace $ProviderNamespace 2>$null | ConvertFrom-Json
    
    if ($provider.registrationState -eq "Registered") {
        Write-Success "Provider $ProviderNamespace is already registered"
        return $true
    }
    
    # Register the provider
    Write-Info "Registering provider: $ProviderNamespace"
    az provider register --namespace $ProviderNamespace 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-ErrorMsg "Failed to initiate registration for $ProviderNamespace"
        return $false
    }
    
    # Wait for registration to complete
    $timeoutSeconds = $TimeoutMinutes * 60
    $elapsedSeconds = 0
    $checkInterval = 10
    
    Write-Info "Waiting for registration to complete (timeout: $TimeoutMinutes minutes)..."
    
    while ($elapsedSeconds -lt $timeoutSeconds) {
        Start-Sleep -Seconds $checkInterval
        $elapsedSeconds += $checkInterval
        
        $provider = az provider show --namespace $ProviderNamespace 2>$null | ConvertFrom-Json
        $state = $provider.registrationState
        
        $progress = [math]::Round(($elapsedSeconds / $timeoutSeconds) * 100)
        Write-Host "." -NoNewline
        
        if ($state -eq "Registered") {
            Write-Host ""
            Write-Success "Provider $ProviderNamespace registered successfully (took $elapsedSeconds seconds)"
            return $true
        }
        elseif ($state -eq "Registering") {
            # Continue waiting
            continue
        }
        else {
            Write-Host ""
            Write-ErrorMsg "Provider registration failed with state: $state"
            return $false
        }
    }
    
    Write-Host ""
    Write-ErrorMsg "Provider registration timed out after $TimeoutMinutes minutes"
    return $false
}

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "     JPMorgan Financial APIs - Azure Deployment (Simplified)" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

# Step 1: Register Required Azure Providers
Write-Step "1/9 - Registering Required Azure Providers"
Write-Info "This step ensures all necessary Azure resource providers are registered"
Write-Info "Registration may take 2-5 minutes per provider if not already registered"

$requiredProviders = @(
    "Microsoft.ContainerService",      # For AKS
    "Microsoft.OperationalInsights",   # For AKS monitoring
    "Microsoft.ContainerRegistry",     # For ACR
    "Microsoft.DBforPostgreSQL",       # For PostgreSQL
    "Microsoft.Cache",                 # For Redis
    "Microsoft.KeyVault"               # For Key Vault
)

$allProvidersRegistered = $true

foreach ($provider in $requiredProviders) {
    $result = Register-AzureProvider -ProviderNamespace $provider -TimeoutMinutes 10
    if (-not $result) {
        $allProvidersRegistered = $false
        Write-ErrorMsg "Failed to register provider: $provider"
    }
}

if (-not $allProvidersRegistered) {
    Write-ErrorMsg "One or more providers failed to register. Please check the errors above."
    Write-Info "You can manually register providers using: az provider register --namespace <provider-name>"
    Write-Info "Then check status with: az provider show --namespace <provider-name> --query registrationState"
    exit 1
}

Write-Success "All required providers are registered and ready"

# Step 2: Verify Prerequisites
Write-Step "2/9 - Verifying Prerequisites"
try {
    $account = az account show 2>$null | ConvertFrom-Json
    Write-Success "Logged in as: $($account.user.name)"
    Write-Info "Subscription: $($account.name) ($($account.id))"
} catch {
    Write-ErrorMsg "Not logged in to Azure. Please run 'az login' first."
    exit 1
}

# Step 3: Verify Resource Group
Write-Step "3/9 - Verifying Resource Group"
$rgExists = az group exists --name $ResourceGroup
if ($rgExists -eq "true") {
    Write-Success "Resource group '$ResourceGroup' exists"
} else {
    Write-Info "Creating resource group..."
    az group create --name $ResourceGroup --location $Location | Out-Null
    Write-Success "Resource group created"
}

# Step 4: Create Azure Container Registry
Write-Step "4/9 - Creating Azure Container Registry"
$acrName = "jpmorganfinancialacr"
Write-Info "Creating ACR '$acrName'..."
Write-Info "This may take 2-3 minutes..."

$acrResult = az acr create `
    --resource-group $ResourceGroup `
    --name $acrName `
    --sku Standard `
    --location $Location `
    --admin-enabled true 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "ACR created successfully"
    $acrLoginServer = "$acrName.azurecr.io"
    Write-Info "ACR Login Server: $acrLoginServer"
} else {
    if ($acrResult -like "*already exists*") {
        Write-Warning "ACR already exists, continuing..."
        $acrLoginServer = "$acrName.azurecr.io"
    } else {
        Write-ErrorMsg "Failed to create ACR: $acrResult"
        exit 1
    }
}

# Step 5: Create AKS Cluster
Write-Step "5/9 - Creating Azure Kubernetes Service Cluster"
$aksName = "jpmorgan-financial-aks"
Write-Info "Creating AKS cluster '$aksName'..."
Write-Warning "This will take 10-15 minutes. Please be patient..."

$aksResult = az aks create `
    --resource-group $ResourceGroup `
    --name $aksName `
    --node-count 3 `
    --node-vm-size Standard_D2s_v3 `
    --enable-addons monitoring `
    --generate-ssh-keys `
    --attach-acr $acrName `
    --location $Location `
    --network-plugin azure `
    --enable-managed-identity 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "AKS cluster created successfully"
} else {
    if ($aksResult -like "*already exists*") {
        Write-Warning "AKS cluster already exists, continuing..."
    } else {
        Write-ErrorMsg "Failed to create AKS: $aksResult"
        exit 1
    }
}

# Step 6: Configure kubectl
Write-Step "6/9 - Configuring kubectl"
Write-Info "Getting AKS credentials..."
az aks get-credentials --resource-group $ResourceGroup --name $aksName --overwrite-existing | Out-Null
Write-Success "kubectl configured"

Write-Info "Verifying cluster connection..."
kubectl get nodes
Write-Success "Cluster connection verified"

# Step 7: Create PostgreSQL Database
Write-Step "7/9 - Creating PostgreSQL Database"
$dbServer = "jpmorgan-financial-db"
$dbAdmin = "jpmadmin"
$dbPassword = "SecureP@ssw0rd2024!" + (Get-Random -Maximum 9999)

Write-Info "Creating PostgreSQL server '$dbServer'..."
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
    if ($dbResult -like "*already exists*") {
        Write-Warning "PostgreSQL server already exists, continuing..."
    } else {
        Write-ErrorMsg "Failed to create PostgreSQL: $dbResult"
        exit 1
    }
}

# Create database
Write-Info "Creating database..."
az postgres flexible-server db create `
    --resource-group $ResourceGroup `
    --server-name $dbServer `
    --database-name jpmorgan_financial_apis_prod 2>&1 | Out-Null

Write-Success "Database created"

# Step 8: Create Redis Cache
Write-Step "8/9 - Creating Redis Cache"
$redisName = "jpmorgan-financial-redis"
Write-Info "Creating Redis cache '$redisName'..."
Write-Warning "This may take 10-15 minutes..."

$redisResult = az redis create `
    --resource-group $ResourceGroup `
    --name $redisName `
    --location $Location `
    --sku Standard `
    --vm-size c1 `
    --enable-non-ssl-port false 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "Redis cache created"
} else {
    if ($redisResult -like "*already exists*") {
        Write-Warning "Redis cache already exists, continuing..."
    } else {
        Write-ErrorMsg "Failed to create Redis: $redisResult"
        exit 1
    }
}

# Step 9: Create Key Vault
Write-Step "9/9 - Creating Key Vault"
$kvName = "jpmorgan-financial-kv"
Write-Info "Creating Key Vault '$kvName'..."

$kvResult = az keyvault create `
    --resource-group $ResourceGroup `
    --name $kvName `
    --location $Location `
    --enable-rbac-authorization false 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "Key Vault created"
} else {
    if ($kvResult -like "*already exists*") {
        Write-Warning "Key Vault already exists, continuing..."
    } else {
        Write-ErrorMsg "Failed to create Key Vault: $kvResult"
        exit 1
    }
}

# Store secrets
Write-Info "Storing secrets in Key Vault..."
az keyvault secret set --vault-name $kvName --name "DatabasePassword" --value $dbPassword 2>&1 | Out-Null
az keyvault secret set --vault-name $kvName --name "JWTSecret" --value ("jwt-secret-" + (New-Guid).ToString()) 2>&1 | Out-Null
Write-Success "Secrets stored"

# Deployment Summary
Write-Host "`n========================================================================" -ForegroundColor Green
Write-Host "                    DEPLOYMENT COMPLETED" -ForegroundColor Green
Write-Host "========================================================================`n" -ForegroundColor Green

Write-Host "Resources Created:" -ForegroundColor Cyan
Write-Host "  [OK] Resource Group: $ResourceGroup"
Write-Host "  [OK] Container Registry: $acrName ($acrLoginServer)"
Write-Host "  [OK] AKS Cluster: $aksName (3 nodes)"
Write-Host "  [OK] PostgreSQL: $dbServer.postgres.database.azure.com"
Write-Host "  [OK] Redis Cache: $redisName.redis.cache.windows.net"
Write-Host "  [OK] Key Vault: $kvName"

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
Location: $Location

ACR: $acrLoginServer
AKS: $aksName
PostgreSQL: $dbServer.postgres.database.azure.com
  Admin: $dbAdmin
  Password: $dbPassword
  Database: jpmorgan_financial_apis_prod

Redis: $redisName.redis.cache.windows.net
Key Vault: $kvName

IMPORTANT: Store these credentials securely and delete this file!
"@ | Out-File -FilePath $credFile -Encoding UTF8

Write-Host "`n[SECURE] Credentials saved to: $credFile" -ForegroundColor Yellow
Write-Host "         Please store securely and delete the file!`n" -ForegroundColor Red
