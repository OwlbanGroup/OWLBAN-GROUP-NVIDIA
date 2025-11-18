# Azure Deployment Guide - JPMorgan Financial APIs

## 🌐 Azure Cloud Deployment Strategy

This guide provides step-by-step instructions for deploying the JPMorgan Financial APIs microservices architecture to Microsoft Azure.

---

## 📋 Table of Contents

1. [Azure Services Overview](#azure-services-overview)
2. [Prerequisites](#prerequisites)
3. [Azure Resource Setup](#azure-resource-setup)
4. [Deployment Options](#deployment-options)
5. [Configuration](#configuration)
6. [Deployment Steps](#deployment-steps)
7. [Monitoring & Management](#monitoring--management)
8. [Cost Optimization](#cost-optimization)

---

## 🏗️ Azure Services Overview

### Recommended Azure Services

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Cloud Platform                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Azure Kubernetes │  │  Azure Container │                │
│  │   Service (AKS)  │  │  Registry (ACR)  │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Azure Database   │  │  Azure Cache     │                │
│  │  for PostgreSQL  │  │  for Redis       │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Azure Monitor    │  │  Application     │                │
│  │  & Log Analytics │  │  Insights        │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Azure Key Vault  │  │  Azure Storage   │                │
│  │  (Secrets)       │  │  (Blob/Files)    │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Service Mapping

| Component | Azure Service | Purpose |
|-----------|--------------|---------|
| **Microservices** | Azure Kubernetes Service (AKS) | Container orchestration |
| **Container Images** | Azure Container Registry (ACR) | Private Docker registry |
| **Database** | Azure Database for PostgreSQL | Managed PostgreSQL |
| **Cache** | Azure Cache for Redis | Managed Redis |
| **Monitoring** | Azure Monitor + Application Insights | Metrics & logs |
| **Secrets** | Azure Key Vault | Secure credential storage |
| **Storage** | Azure Blob Storage | File storage |
| **Load Balancer** | Azure Load Balancer | Traffic distribution |
| **DNS** | Azure DNS | Domain management |
| **CDN** | Azure CDN | Content delivery |

---

## 📦 Prerequisites

### 1. Azure Account Setup

```bash
# Create Azure account (if not exists)
# Visit: https://azure.microsoft.com/free/

# Install Azure CLI
# Windows (PowerShell)
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'

# Verify installation
az --version
```

### 2. Login to Azure

```bash
# Login to Azure
az login

# Set subscription (if multiple)
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verify current subscription
az account show
```

### 3. Install Required Tools

```bash
# Install kubectl
az aks install-cli

# Install Docker Desktop (if not installed)
# Download from: https://www.docker.com/products/docker-desktop

# Install Helm
choco install kubernetes-helm  # Windows
```

---

## 🚀 Azure Resource Setup

### Step 1: Create Resource Group

```bash
# Set variables
$RESOURCE_GROUP="jpmorgan-financial-apis-rg"
$LOCATION="eastus"  # or your preferred region

# Create resource group
az group create `
  --name $RESOURCE_GROUP `
  --location $LOCATION
```

### Step 2: Create Azure Container Registry (ACR)

```bash
# Set ACR name (must be globally unique)
$ACR_NAME="jpmorganfinancialacr"

# Create ACR
az acr create `
  --resource-group $RESOURCE_GROUP `
  --name $ACR_NAME `
  --sku Standard `
  --location $LOCATION

# Enable admin access
az acr update `
  --name $ACR_NAME `
  --admin-enabled true

# Get ACR credentials
az acr credential show --name $ACR_NAME
```

### Step 3: Create Azure Kubernetes Service (AKS)

```bash
# Set AKS cluster name
$AKS_CLUSTER="jpmorgan-financial-aks"

# Create AKS cluster
az aks create `
  --resource-group $RESOURCE_GROUP `
  --name $AKS_CLUSTER `
  --node-count 3 `
  --node-vm-size Standard_D2s_v3 `
  --enable-addons monitoring `
  --generate-ssh-keys `
  --attach-acr $ACR_NAME `
  --location $LOCATION

# Get AKS credentials
az aks get-credentials `
  --resource-group $RESOURCE_GROUP `
  --name $AKS_CLUSTER

# Verify connection
kubectl get nodes
```

### Step 4: Create Azure Database for PostgreSQL

```bash
# Set database variables
$DB_SERVER="jpmorgan-financial-db"
$DB_ADMIN="jpmadmin"
$DB_PASSWORD="SecureP@ssw0rd2024!"  # Change this!

# Create PostgreSQL server
az postgres flexible-server create `
  --resource-group $RESOURCE_GROUP `
  --name $DB_SERVER `
  --location $LOCATION `
  --admin-user $DB_ADMIN `
  --admin-password $DB_PASSWORD `
  --sku-name Standard_D2s_v3 `
  --tier GeneralPurpose `
  --version 15 `
  --storage-size 128 `
  --public-access 0.0.0.0

# Create database
az postgres flexible-server db create `
  --resource-group $RESOURCE_GROUP `
  --server-name $DB_SERVER `
  --database-name jpmorgan_financial_apis_prod

# Get connection string
az postgres flexible-server show-connection-string `
  --server-name $DB_SERVER `
  --database-name jpmorgan_financial_apis_prod `
  --admin-user $DB_ADMIN `
  --admin-password $DB_PASSWORD
```

### Step 5: Create Azure Cache for Redis

```bash
# Set Redis cache name
$REDIS_NAME="jpmorgan-financial-redis"

# Create Redis cache
az redis create `
  --resource-group $RESOURCE_GROUP `
  --name $REDIS_NAME `
  --location $LOCATION `
  --sku Standard `
  --vm-size c1 `
  --enable-non-ssl-port

# Get Redis connection info
az redis show `
  --resource-group $RESOURCE_GROUP `
  --name $REDIS_NAME

# Get Redis keys
az redis list-keys `
  --resource-group $RESOURCE_GROUP `
  --name $REDIS_NAME
```

### Step 6: Create Azure Key Vault

```bash
# Set Key Vault name
$KEYVAULT_NAME="jpmorgan-financial-kv"

# Create Key Vault
az keyvault create `
  --resource-group $RESOURCE_GROUP `
  --name $KEYVAULT_NAME `
  --location $LOCATION `
  --enable-rbac-authorization false

# Store secrets
az keyvault secret set `
  --vault-name $KEYVAULT_NAME `
  --name "DatabasePassword" `
  --value $DB_PASSWORD

az keyvault secret set `
  --vault-name $KEYVAULT_NAME `
  --name "RedisPassword" `
  --value "YOUR_REDIS_PRIMARY_KEY"

az keyvault secret set `
  --vault-name $KEYVAULT_NAME `
  --name "JWTSecret" `
  --value "your-jwt-secret-key-here"
```

### Step 7: Create Azure Storage Account

```bash
# Set storage account name
$STORAGE_ACCOUNT="jpmorganfinancialstorage"

# Create storage account
az storage account create `
  --resource-group $RESOURCE_GROUP `
  --name $STORAGE_ACCOUNT `
  --location $LOCATION `
  --sku Standard_LRS `
  --kind StorageV2

# Create blob container
az storage container create `
  --account-name $STORAGE_ACCOUNT `
  --name telemetry-data `
  --public-access off

# Get connection string
az storage account show-connection-string `
  --resource-group $RESOURCE_GROUP `
  --name $STORAGE_ACCOUNT
```

---

## 🐳 Build and Push Docker Images

### Step 1: Login to ACR

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Get ACR login server
$ACR_LOGIN_SERVER = az acr show `
  --name $ACR_NAME `
  --query loginServer `
  --output tsv
```

### Step 2: Build and Push Images

```bash
# Navigate to project root
cd c:/Users/bizle/Desktop/jpmorgan_financial_apis

# Build and push each microservice
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

foreach ($service in $services) {
    Write-Host "Building $service..."
    
    # Build image
    docker build `
        -t "${ACR_LOGIN_SERVER}/jpmorgan-${service}:latest" `
        -f "microservices/${service}/Dockerfile" `
        "microservices/${service}"
    
    # Push image
    docker push "${ACR_LOGIN_SERVER}/jpmorgan-${service}:latest"
}
```

---

## ⚙️ Kubernetes Deployment

### Step 1: Create Kubernetes Secrets

```bash
# Create namespace
kubectl create namespace jpmorgan-financial

# Create secrets from Azure Key Vault
kubectl create secret generic app-secrets `
  --from-literal=DATABASE_URL="postgresql://${DB_ADMIN}:${DB_PASSWORD}@${DB_SERVER}.postgres.database.azure.com:5432/jpmorgan_financial_apis_prod" `
  --from-literal=REDIS_URL="redis://:YOUR_REDIS_KEY@${REDIS_NAME}.redis.cache.windows.net:6380/0?ssl=true" `
  --from-literal=JWT_SECRET="your-jwt-secret" `
  --from-literal=AZURE_STORAGE_CONNECTION_STRING="YOUR_STORAGE_CONNECTION_STRING" `
  --namespace jpmorgan-financial
```

### Step 2: Deploy to AKS

```bash
# Apply Kubernetes manifests
kubectl apply -f microservices/deployment/kubernetes/ --namespace jpmorgan-financial

# Verify deployments
kubectl get deployments --namespace jpmorgan-financial
kubectl get pods --namespace jpmorgan-financial
kubectl get services --namespace jpmorgan-financial
```

### Step 3: Configure Ingress

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Wait for external IP
kubectl get service ingress-nginx-controller --namespace ingress-nginx --watch

# Apply ingress configuration
kubectl apply -f microservices/deployment/kubernetes/ingress.yaml --namespace jpmorgan-financial
```

---

## 📊 Monitoring Setup

### Azure Monitor Integration

```bash
# Enable Container Insights
az aks enable-addons `
  --resource-group $RESOURCE_GROUP `
  --name $AKS_CLUSTER `
  --addons monitoring

# Create Log Analytics Workspace
$WORKSPACE_NAME="jpmorgan-financial-logs"

az monitor log-analytics workspace create `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $WORKSPACE_NAME `
  --location $LOCATION
```

### Application Insights

```bash
# Create Application Insights
$APPINSIGHTS_NAME="jpmorgan-financial-insights"

az monitor app-insights component create `
  --app $APPINSIGHTS_NAME `
  --location $LOCATION `
  --resource-group $RESOURCE_GROUP `
  --application-type web

# Get instrumentation key
az monitor app-insights component show `
  --app $APPINSIGHTS_NAME `
  --resource-group $RESOURCE_GROUP `
  --query instrumentationKey
```

---

## 🔒 Security Configuration

### 1. Network Security

```bash
# Create Network Security Group
az network nsg create `
  --resource-group $RESOURCE_GROUP `
  --name jpmorgan-financial-nsg

# Add security rules
az network nsg rule create `
  --resource-group $RESOURCE_GROUP `
  --nsg-name jpmorgan-financial-nsg `
  --name AllowHTTPS `
  --priority 100 `
  --destination-port-ranges 443 `
  --protocol Tcp `
  --access Allow
```

### 2. Enable Azure AD Authentication

```bash
# Enable Azure AD integration for AKS
az aks update `
  --resource-group $RESOURCE_GROUP `
  --name $AKS_CLUSTER `
  --enable-azure-rbac `
  --enable-aad
```

### 3. Configure SSL/TLS

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind:ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

---

## 💰 Cost Optimization

### Recommended Tier Selection

| Service | Tier | Monthly Cost (Est.) |
|---------|------|---------------------|
| AKS (3 nodes) | Standard_D2s_v3 | ~$200 |
| PostgreSQL | GeneralPurpose D2s_v3 | ~$150 |
| Redis | Standard C1 | ~$75 |
| Storage | Standard LRS | ~$20 |
| Monitoring | Pay-as-you-go | ~$50 |
| **Total** | | **~$495/month** |

### Cost Saving Tips

1. **Use Azure Reserved Instances** - Save up to 72%
2. **Auto-scaling** - Scale down during off-hours
3. **Spot Instances** - Use for non-critical workloads
4. **Storage Lifecycle** - Move old data to cool/archive tiers
5. **Monitor Usage** - Use Azure Cost Management

---

## 🔄 CI/CD Pipeline

### Azure DevOps Pipeline

Create `.azure-pipelines.yml`:

```yaml
trigger:
  branches:
    include:
    - main
    - develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  azureSubscription: 'YOUR_SERVICE_CONNECTION'
  acrName: 'jpmorganfinancialacr'
  aksCluster: 'jpmorgan-financial-aks'
  resourceGroup: 'jpmorgan-financial-apis-rg'

stages:
- stage: Build
  jobs:
  - job: BuildAndPush
    steps:
    - task: Docker@2
      inputs:
        containerRegistry: '$(azureSubscription)'
        repository: 'jpmorgan-financial-apis'
        command: 'buildAndPush'
        Dockerfile: '**/Dockerfile'
        tags: |
          $(Build.BuildId)
          latest

- stage: Deploy
  jobs:
  - job: DeployToAKS
    steps:
    - task: Kubernetes@1
      inputs:
        connectionType: 'Azure Resource Manager'
        azureSubscriptionEndpoint: '$(azureSubscription)'
        azureResourceGroup: '$(resourceGroup)'
        kubernetesCluster: '$(aksCluster)'
        command: 'apply'
        arguments: '-f microservices/deployment/kubernetes/'
```

---

## 📝 Post-Deployment Checklist

- [ ] Verify all pods are running
- [ ] Test database connectivity
- [ ] Verify Redis cache connection
- [ ] Test API endpoints
- [ ] Configure DNS records
- [ ] Set up SSL certificates
- [ ] Configure monitoring alerts
- [ ] Test backup and restore
- [ ] Document access credentials
- [ ] Set up log aggregation
- [ ] Configure auto-scaling
- [ ] Test disaster recovery

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: Pods not starting
```bash
kubectl describe pod POD_NAME --namespace jpmorgan-financial
kubectl logs POD_NAME --namespace jpmorgan-financial
```

**Issue**: Database connection failed
```bash
# Test connection
az postgres flexible-server connect `
  --name $DB_SERVER `
  --admin-user $DB_ADMIN `
  --admin-password $DB_PASSWORD
```

**Issue**: Image pull errors
```bash
# Verify ACR integration
az aks check-acr `
  --resource-group $RESOURCE_GROUP `
  --name $AKS_CLUSTER `
  --acr $ACR_NAME
```

---

## 📚 Additional Resources

- [Azure Kubernetes Service Documentation](https://docs.microsoft.com/azure/aks/)
- [Azure Database for PostgreSQL](https://docs.microsoft.com/azure/postgresql/)
- [Azure Cache for Redis](https://docs.microsoft.com/azure/azure-cache-for-redis/)
- [Azure Monitor](https://docs.microsoft.com/azure/azure-monitor/)
- [Azure DevOps](https://docs.microsoft.com/azure/devops/)

---

**Deployment Guide Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Ready for Production Deployment
