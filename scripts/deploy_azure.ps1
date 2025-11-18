<#
.SYNOPSIS
    Automated Azure Deployment Script for JPMorgan Financial APIs

.DESCRIPTION
    This script automates the complete deployment of JPMorgan Financial APIs to Azure,
    including resource creation, container registry setup, AKS deployment, and monitoring.

.EXAMPLE
    .\deploy_azure.ps1 -ResourceGroup "jpmorgan-rg" -Location "eastus"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "jpmorgan-financial-apis-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$ACRName = "jpmorganfinancialacr",
    
    [Parameter(Mandatory=$false)]
    [string]$AKSCluster = "jpmorgan-financial-aks",
    
    [Parameter(Mandatory=$false)]
    [string]$DBServer = "jpmorgan-financial-db",
    
    [Parameter(Mandatory=$false)]
    [string]$RedisName = "jpmorgan-financial-redis",
    
    [Parameter(Mandatory=$false)]
    [string]$KeyVaultName = "jpmorgan-financial-kv",
    
    [Parameter(Mandatory=$false)]
    [string]$StorageAccount = "jpmorganfinancialstorage"
)

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Cyan
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n🚀 $Message" -ForegroundColor Magenta
    Write-Host ("=" * 70) -ForegroundColor Magenta
}

# Check prerequisites
function Test-Prerequisites {
    Write-Step "Checking Prerequisites"
    
    # Check Azure CLI
    try {
        $azVersion = az --version
        Write-Success "Azure CLI is installed"
    }
    catch {
        Write-Error "Azure CLI is not installed. Please install from: https://aka.ms/installazurecliwindows"
        exit 1
    }
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Success "Docker is installed"
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop"
        exit 1
    }
    
    # Check kubectl
    try {
        $kubectlVersion = kubectl version --client
        Write-Success "kubectl is installed"
    }
    catch {
        Write-Warning "kubectl is not installed. Will install via Azure CLI"
    }
    
    # Check Azure login
    try {
        $account = az account show 2>$null
        if ($account) {
            Write-Success "Logged in to Azure"
            $accountInfo = $account | ConvertFrom-Json
            Write-Info "Subscription: $($accountInfo.name)"
        }
        else {
            Write-Warning "Not logged in to Azure. Initiating login..."
            az login
        }
    }
    catch {
        Write-Warning "Not logged in to Azure. Initiating login..."
        az login
    }
}

# Create Resource Group
function New-AzureResourceGroup {
    Write-Step "Creating Resource Group"
    
    $exists = az group exists --name $ResourceGroup
    if ($exists -eq "true") {
        Write-Info "Resource group '$ResourceGroup' already exists"
    }
    else {
        Write-Info "Creating resource group '$ResourceGroup' in '$Location'..."
        az group create --name $ResourceGroup --location $Location
        Write-Success "Resource group created"
    }
}

# Create Azure Container Registry
function New-AzureContainerRegistry {
    Write-Step "Creating Azure Container Registry"
    
    Write-Info "Creating ACR '$ACRName'..."
    az acr create `
        --resource-group $ResourceGroup `
        --name $ACRName `
        --sku Standard `
        --location $Location `
        --admin-enabled true
    
    Write-Success "ACR created successfully"
    
    # Get ACR credentials
    Write-Info "Retrieving ACR credentials..."
    $acrCreds = az acr credential show --name $ACRName | ConvertFrom-Json
    Write-Info "ACR Username: $($acrCreds.username)"
    
    return $acrCreds
}

# Create AKS Cluster
function New-AzureKubernetesCluster {
    Write-Step "Creating Azure Kubernetes Service Cluster"
    
    Write-Info "Creating AKS cluster '$AKSCluster' (this may take 10-15 minutes)..."
    az aks create `
        --resource-group $ResourceGroup `
        --name $AKSCluster `
        --node-count 3 `
        --node-vm-size Standard_D2s_v3 `
        --enable-addons monitoring `
        --generate-ssh-keys `
        --attach-acr $ACRName `
        --location $Location `
        --network-plugin azure `
        --enable-managed-identity
    
    Write-Success "AKS cluster created successfully"
    
    # Get AKS credentials
    Write-Info "Configuring kubectl..."
    az aks get-credentials --resource-group $ResourceGroup --name $AKSCluster --overwrite-existing
    
    # Verify connection
    Write-Info "Verifying cluster connection..."
    kubectl get nodes
    Write-Success "kubectl configured successfully"
}

# Create PostgreSQL Database
function New-AzurePostgreSQL {
    Write-Step "Creating Azure Database for PostgreSQL"
    
    $dbAdmin = "jpmadmin"
    $dbPassword = "SecureP@ssw0rd2024!" + (Get-Random -Maximum 9999)
    
    Write-Info "Creating PostgreSQL server '$DBServer'..."
    az postgres flexible-server create `
        --resource-group $ResourceGroup `
        --name $DBServer `
        --location $Location `
        --admin-user $dbAdmin `
        --admin-password $dbPassword `
        --sku-name Standard_D2s_v3 `
        --tier GeneralPurpose `
        --version 15 `
        --storage-size 128 `
        --public-access 0.0.0.0-255.255.255.255
    
    Write-Success "PostgreSQL server created"
    
    # Create database
    Write-Info "Creating database..."
    az postgres flexible-server db create `
        --resource-group $ResourceGroup `
        --server-name $DBServer `
        --database-name jpmorgan_financial_apis_prod
    
    Write-Success "Database created successfully"
    
    # Return connection info
    return @{
        Server = "$DBServer.postgres.database.azure.com"
        Admin = $dbAdmin
        Password = $dbPassword
        Database = "jpmorgan_financial_apis_prod"
    }
}

# Create Redis Cache
function New-AzureRedisCache {
    Write-Step "Creating Azure Cache for Redis"
    
    Write-Info "Creating Redis cache '$RedisName' (this may take 10-15 minutes)..."
    az redis create `
        --resource-group $ResourceGroup `
        --name $RedisName `
        --location $Location `
        --sku Standard `
        --vm-size c1 `
        --enable-non-ssl-port false
    
    Write-Success "Redis cache created successfully"
    
    # Get Redis keys
    Write-Info "Retrieving Redis keys..."
    $redisKeys = az redis list-keys --resource-group $ResourceGroup --name $RedisName | ConvertFrom-Json
    
    return @{
        Host = "$RedisName.redis.cache.windows.net"
        Port = 6380
        PrimaryKey = $redisKeys.primaryKey
    }
}

# Create Key Vault
function New-AzureKeyVault {
    param(
        [hashtable]$DBInfo,
        [hashtable]$RedisInfo
    )
    
    Write-Step "Creating Azure Key Vault"
    
    Write-Info "Creating Key Vault '$KeyVaultName'..."
    az keyvault create `
        --resource-group $ResourceGroup `
        --name $KeyVaultName `
        --location $Location `
        --enable-rbac-authorization false
    
    Write-Success "Key Vault created successfully"
    
    # Store secrets
    Write-Info "Storing secrets in Key Vault..."
    
    az keyvault secret set --vault-name $KeyVaultName --name "DatabasePassword" --value $DBInfo.Password
    az keyvault secret set --vault-name $KeyVaultName --name "RedisPassword" --value $RedisInfo.PrimaryKey
    az keyvault secret set --vault-name $KeyVaultName --name "JWTSecret" --value ("jwt-secret-" + (New-Guid).ToString())
    
    Write-Success "Secrets stored in Key Vault"
}

# Create Storage Account
function New-AzureStorageAccount {
    Write-Step "Creating Azure Storage Account"
    
    Write-Info "Creating storage account '$StorageAccount'..."
    az storage account create `
        --resource-group $ResourceGroup `
        --name $StorageAccount `
        --location $Location `
        --sku Standard_LRS `
        --kind StorageV2
    
    Write-Success "Storage account created"
    
    # Create blob container
    Write-Info "Creating blob container..."
    az storage container create `
        --account-name $StorageAccount `
        --name telemetry-data `
        --public-access off
    
    # Get connection string
    $connString = az storage account show-connection-string `
        --resource-group $ResourceGroup `
        --name $StorageAccount `
        --query connectionString `
        --output tsv
    
    Write-Success "Storage account configured"
    
    return $connString
}

# Build and Push Docker Images
function Build-AndPushImages {
    param([string]$ACRLoginServer)
    
    Write-Step "Building and Pushing Docker Images"
    
    # Login to ACR
    Write-Info "Logging in to ACR..."
    az acr login --name $ACRName
    
    $services = @(
        "auth",
        "benefits",
        "payroll",
        "patterns",
        "traction",
        "purchasing",
        "bill-pay",
        "ml",
        "telemetry",
        "storage",
        "dashboard",
        "api-gateway"
    )
    
    $projectRoot = Split-Path -Parent $PSScriptRoot
    
    foreach ($service in $services) {
        Write-Info "Building $service..."
        
        $dockerfilePath = Join-Path $projectRoot "microservices\$service\Dockerfile"
        $contextPath = Join-Path $projectRoot "microservices\$service"
        
        if (Test-Path $dockerfilePath) {
            docker build `
                -t "${ACRLoginServer}/jpmorgan-${service}:latest" `
                -f $dockerfilePath `
                $contextPath
            
            Write-Info "Pushing $service to ACR..."
            docker push "${ACRLoginServer}/jpmorgan-${service}:latest"
            
            Write-Success "$service image pushed successfully"
        }
        else {
            Write-Warning "Dockerfile not found for $service at $dockerfilePath"
        }
    }
}

# Deploy to Kubernetes
function Deploy-ToKubernetes {
    param(
        [hashtable]$DBInfo,
        [hashtable]$RedisInfo,
        [string]$StorageConnString
    )
    
    Write-Step "Deploying to Kubernetes"
    
    # Create namespace
    Write-Info "Creating namespace..."
    kubectl create namespace jpmorgan-financial --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    Write-Info "Creating Kubernetes secrets..."
    
    $dbUrl = "postgresql://$($DBInfo.Admin):$($DBInfo.Password)@$($DBInfo.Server):5432/$($DBInfo.Database)"
    $redisUrl = "rediss://:$($RedisInfo.PrimaryKey)@$($RedisInfo.Host):$($RedisInfo.Port)/0"
    
    kubectl create secret generic app-secrets `
        --from-literal=DATABASE_URL=$dbUrl `
        --from-literal=REDIS_URL=$redisUrl `
        --from-literal=JWT_SECRET="jwt-secret-key" `
        --from-literal=AZURE_STORAGE_CONNECTION_STRING=$StorageConnString `
        --namespace jpmorgan-financial `
        --dry-run=client -o yaml | kubectl apply -f -
    
    Write-Success "Secrets created"
    
    # Apply Kubernetes manifests
    Write-Info "Applying Kubernetes manifests..."
    $k8sPath = Join-Path (Split-Path -Parent $PSScriptRoot) "microservices\deployment\kubernetes"
    
    if (Test-Path $k8sPath) {
        kubectl apply -f $k8sPath --namespace jpmorgan-financial
        Write-Success "Kubernetes manifests applied"
    }
    else {
        Write-Warning "Kubernetes manifests not found at $k8sPath"
    }
    
    # Wait for deployments
    Write-Info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment --all --namespace jpmorgan-financial
    
    Write-Success "All deployments are ready"
}

# Setup Monitoring
function Setup-Monitoring {
    Write-Step "Setting Up Monitoring"
    
    Write-Info "Enabling Container Insights..."
    az aks enable-addons `
        --resource-group $ResourceGroup `
        --name $AKSCluster `
        --addons monitoring
    
    Write-Info "Creating Application Insights..."
    az monitor app-insights component create `
        --app "jpmorgan-financial-insights" `
        --location $Location `
        --resource-group $ResourceGroup `
        --application-type web
    
    Write-Success "Monitoring configured"
}

# Generate deployment summary
function Write-DeploymentSummary {
    param(
        [hashtable]$DBInfo,
        [hashtable]$RedisInfo,
        [string]$ACRLoginServer
    )
    
    Write-Step "Deployment Summary"
    
    Write-Host "`n📊 Azure Resources Created:" -ForegroundColor Cyan
    Write-Host "  Resource Group: $ResourceGroup"
    Write-Host "  Location: $Location"
    Write-Host "`n🐳 Container Registry:"
    Write-Host "  ACR Name: $ACRName"
    Write-Host "  Login Server: $ACRLoginServer"
    Write-Host "`n☸️  Kubernetes Cluster:"
    Write-Host "  AKS Cluster: $AKSCluster"
    Write-Host "  Nodes: 3 x Standard_D2s_v3"
    Write-Host "`n🗄️  Database:"
    Write-Host "  Server: $($DBInfo.Server)"
    Write-Host "  Database: $($DBInfo.Database)"
    Write-Host "  Admin: $($DBInfo.Admin)"
    Write-Host "`n🔴 Redis Cache:"
    Write-Host "  Host: $($RedisInfo.Host)"
    Write-Host "  Port: $($RedisInfo.Port)"
    Write-Host "`n🔐 Key Vault:"
    Write-Host "  Name: $KeyVaultName"
    Write-Host "`n💾 Storage Account:"
    Write-Host "  Name: $StorageAccount"
    
    Write-Host "`n✅ Deployment completed successfully!" -ForegroundColor Green
    Write-Host "`n📝 Next Steps:" -ForegroundColor Yellow
    Write-Host "  1. Get external IP: kubectl get services --namespace jpmorgan-financial"
    Write-Host "  2. Configure DNS records"
    Write-Host "  3. Set up SSL certificates"
    Write-Host "  4. Test API endpoints"
    Write-Host "  5. Configure monitoring alerts"
    
    # Save credentials to file
    $credFile = Join-Path (Split-Path -Parent $PSScriptRoot) "azure_credentials.txt"
    @"
Azure Deployment Credentials
=============================
Date: $(Get-Date)

Resource Group: $ResourceGroup
Location: $Location

ACR:
  Name: $ACRName
  Login Server: $ACRLoginServer

AKS:
  Cluster: $AKSCluster

Database:
  Server: $($DBInfo.Server)
  Database: $($DBInfo.Database)
  Admin: $($DBInfo.Admin)
  Password: $($DBInfo.Password)

Redis:
  Host: $($RedisInfo.Host)
  Port: $($RedisInfo.Port)
  Primary Key: $($RedisInfo.PrimaryKey)

Key Vault: $KeyVaultName
Storage Account: $StorageAccount

IMPORTANT: Store these credentials securely and delete this file after saving them to a secure location.
"@ | Out-File -FilePath $credFile -Encoding UTF8
    
    Write-Host "`n🔒 Credentials saved to: $credFile" -ForegroundColor Yellow
    Write-Host "   Please store these securely and delete the file!" -ForegroundColor Red
}

# Main deployment function
function Start-AzureDeployment {
    Write-Host "`n" -NoNewline
    Write-Host "╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                                                                    ║" -ForegroundColor Cyan
    Write-Host "║        JPMorgan Financial APIs - Azure Deployment Script          ║" -ForegroundColor Cyan
    Write-Host "║                                                                    ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host "`n"
    
    try {
        # Check prerequisites
        Test-Prerequisites
        
        # Create resources
        New-AzureResourceGroup
        $acrCreds = New-AzureContainerRegistry
        $acrLoginServer = "$ACRName.azurecr.io"
        
        New-AzureKubernetesCluster
        $dbInfo = New-AzurePostgreSQL
        $redisInfo = New-AzureRedisCache
        New-AzureKeyVault -DBInfo $dbInfo -RedisInfo $redisInfo
        $storageConnString = New-AzureStorageAccount
        
        # Build and deploy
        Build-AndPushImages -ACRLoginServer $acrLoginServer
        Deploy-ToKubernetes -DBInfo $dbInfo -RedisInfo $redisInfo -StorageConnString $storageConnString
        Setup-Monitoring
        
        # Summary
        Write-DeploymentSummary -DBInfo $dbInfo -RedisInfo $redisInfo -ACRLoginServer $acrLoginServer
        
    }
    catch {
        Write-Error "Deployment failed: $_"
        Write-Host $_.ScriptStackTrace -ForegroundColor Red
        exit 1
    }
}

# Run deployment
Start-AzureDeployment
