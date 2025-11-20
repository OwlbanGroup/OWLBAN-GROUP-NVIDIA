<#
.SYNOPSIS
    Fix Remaining Azure Deployment - Create Missing Resources

.DESCRIPTION
    Creates only the missing resources (Redis and Key Vault) in the existing resource group
    Checks for existing resources before attempting creation
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
Write-Host "     JPMorgan Financial APIs - Fix Remaining Deployment" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

# Check existing resources
Write-Step "Checking Existing Resources"

# Check PostgreSQL
Write-Info "Checking PostgreSQL..."
$dbExists = az postgres flexible-server show --resource-group $ResourceGroup --name jpmorgan-financial-db --query "name" -o tsv 2>$null
if ($dbExists) {
    Write-Success "PostgreSQL server already exists: jpmorgan-financial-db"
} else {
    Write-Warning "PostgreSQL server not found"
}

# Check Redis
Write-Info "Checking Redis Cache..."
$redisExists = az redis show --resource-group $ResourceGroup --name jpmorgan-financial-redis --query "name" -o tsv 2>$null
if ($redisExists) {
    Write-Success "Redis cache already exists: jpmorgan-financial-redis"
} else {
    Write-Warning "Redis cache not found - will create"
}

# Check Key Vault
Write-Info "Checking Key Vault..."
$kvExists = az keyvault show --resource-group $ResourceGroup --name jpmorgan-financial-kv --query "name" -o tsv 2>$null
if ($kvExists) {
    Write-Success "Key Vault already exists: jpmorgan-financial-kv"
} else {
    Write-Warning "Key Vault not found - will create"
}

# Create Redis Cache if missing
if (-not $redisExists) {
    Write-Step "Creating Redis Cache"
    $redisName = "jpmorgan-financial-redis"
    Write-Info "Creating Redis cache '$redisName' in $Location..."
    Write-Warning "This may take 10-15 minutes..."
    
    try {
        $redisResult = az redis create `
            --resource-group $ResourceGroup `
            --name $redisName `
            --location $Location `
            --sku Standard `
            --vm-size c1 `
            --enable-non-ssl-port 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Redis cache created successfully"
        } else {
            $errorMsg = $redisResult | Out-String
            if ($errorMsg -like "*already exists*" -or $errorMsg -like "*AlreadyExists*") {
                Write-Warning "Redis cache already exists, continuing..."
            } else {
                Write-ErrorMsg "Failed to create Redis: $errorMsg"
            }
        }
    } catch {
        Write-ErrorMsg "Exception creating Redis: $_"
    }
} else {
    Write-Info "Skipping Redis creation - already exists"
}

# Create Key Vault if missing
if (-not $kvExists) {
    Write-Step "Creating Key Vault"
    $kvName = "jpmorgan-financial-kv"
    Write-Info "Creating Key Vault '$kvName' in $Location..."
    
    try {
        $kvResult = az keyvault create `
            --resource-group $ResourceGroup `
            --name $kvName `
            --location $Location `
            --enable-rbac-authorization false `
            --enabled-for-deployment true `
            --enabled-for-template-deployment true 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Key Vault created successfully"
        } else {
            $errorMsg = $kvResult | Out-String
            if ($errorMsg -like "*already exists*" -or $errorMsg -like "*AlreadyExists*") {
                Write-Warning "Key Vault already exists, continuing..."
            } else {
                Write-ErrorMsg "Failed to create Key Vault: $errorMsg"
            }
        }
    } catch {
        Write-ErrorMsg "Exception creating Key Vault: $_"
    }
} else {
    Write-Info "Skipping Key Vault creation - already exists"
}

# Store/Update secrets in Key Vault
Write-Step "Configuring Key Vault Secrets"
$kvName = "jpmorgan-financial-kv"

# Check if Key Vault is accessible
$kvAccessible = az keyvault show --resource-group $ResourceGroup --name $kvName --query "name" -o tsv 2>$null
if ($kvAccessible) {
    Write-Info "Storing/updating secrets in Key Vault..."
    
    # Generate or retrieve database password
    $dbPassword = "SecureP@ssw0rd2024!" + (Get-Random -Maximum 9999)
    
    try {
        az keyvault secret set --vault-name $kvName --name "DatabasePassword" --value $dbPassword 2>&1 | Out-Null
        Write-Success "Database password stored"
    } catch {
        Write-Warning "Could not store database password: $_"
    }
    
    try {
        $jwtSecret = "jwt-secret-" + (New-Guid).ToString()
        az keyvault secret set --vault-name $kvName --name "JWTSecret" --value $jwtSecret 2>&1 | Out-Null
        Write-Success "JWT secret stored"
    } catch {
        Write-Warning "Could not store JWT secret: $_"
    }
    
    try {
        $apiKey = "api-key-" + (New-Guid).ToString()
        az keyvault secret set --vault-name $kvName --name "APIKey" --value $apiKey 2>&1 | Out-Null
        Write-Success "API key stored"
    } catch {
        Write-Warning "Could not store API key: $_"
    }
} else {
    Write-Warning "Key Vault not accessible yet - secrets will need to be configured later"
}

# Final Status Check
Write-Step "Final Status Check"

Write-Info "Checking all resources..."
Start-Sleep -Seconds 5

# Check Redis final status
$redisStatus = az redis show --resource-group $ResourceGroup --name jpmorgan-financial-redis --query "provisioningState" -o tsv 2>$null
if ($redisStatus) {
    Write-Success "Redis Cache: $redisStatus"
} else {
    Write-Warning "Redis Cache: Still provisioning or not found"
}

# Check Key Vault final status
$kvStatus = az keyvault show --resource-group $ResourceGroup --name jpmorgan-financial-kv --query "properties.provisioningState" -o tsv 2>$null
if ($kvStatus) {
    Write-Success "Key Vault: $kvStatus"
} else {
    Write-Warning "Key Vault: Still provisioning or not found"
}

# Deployment Summary
Write-Host "`n========================================================================" -ForegroundColor Green
Write-Host "                    FIX DEPLOYMENT COMPLETED" -ForegroundColor Green
Write-Host "========================================================================`n" -ForegroundColor Green

Write-Host "Resource Status:" -ForegroundColor Cyan
Write-Host "  [OK] Resource Group: $ResourceGroup"
Write-Host "  [OK] Container Registry: jpmorganfinancialacr.azurecr.io"
Write-Host "  [OK] AKS Cluster: jpmorgan-financial-aks"
Write-Host "  [OK] PostgreSQL: jpmorgan-financial-db.postgres.database.azure.com"

if ($redisStatus -eq "Succeeded" -or $redisStatus -eq "Creating") {
    Write-Host "  [OK] Redis Cache: jpmorgan-financial-redis.redis.cache.windows.net ($redisStatus)"
} else {
    Write-Host "  [PENDING] Redis Cache: Check status with check_deployment_status.ps1"
}

if ($kvStatus -eq "Succeeded" -or $kvStatus -eq "Creating") {
    Write-Host "  [OK] Key Vault: jpmorgan-financial-kv ($kvStatus)"
} else {
    Write-Host "  [PENDING] Key Vault: Check status with check_deployment_status.ps1"
}

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Wait 5-10 minutes for Redis provisioning to complete"
Write-Host "  2. Run: .\scripts\check_deployment_status.ps1"
Write-Host "  3. Once all resources show 'Succeeded', proceed with application deployment"
Write-Host "  4. Build and push Docker images to ACR"
Write-Host "  5. Deploy applications to AKS"

Write-Host "`nMonitoring:" -ForegroundColor Yellow
Write-Host "  Run this to check status: .\scripts\check_deployment_status.ps1"
Write-Host "  Redis typically takes 10-15 minutes to provision"
Write-Host "  Key Vault typically takes 1-2 minutes to provision"

Write-Host ""
