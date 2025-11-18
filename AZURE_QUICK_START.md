# Azure Deployment - Quick Start Guide

## 🚀 Deploy to Azure in 3 Steps

This guide will help you deploy the JPMorgan Financial APIs to Microsoft Azure Cloud in minutes.

---

## Prerequisites

Before you begin, ensure you have:

1. **Azure Account** - [Create free account](https://azure.microsoft.com/free/) ($200 credit for 30 days)
2. **Azure CLI** - [Install Azure CLI](https://aka.ms/installazurecliwindows)
3. **Docker Desktop** - [Install Docker](https://www.docker.com/products/docker-desktop)
4. **PowerShell 7+** - Already installed on Windows 10/11

---

## Step 1: Prepare Your Environment (5 minutes)

### Install Azure CLI

```powershell
# Download and install Azure CLI
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'

# Verify installation
az --version
```

### Login to Azure

```powershell
# Login to your Azure account
az login

# This will open a browser window for authentication
# After login, you'll see your subscription list

# Set your subscription (if you have multiple)
az account set --subscription "YOUR_SUBSCRIPTION_NAME"
```

---

## Step 2: Run Automated Deployment (30-45 minutes)

### Option A: Automated Deployment (Recommended)

```powershell
# Navigate to the scripts directory
cd c:\Users\bizle\Desktop\jpmorgan_financial_apis\scripts

# Run the automated deployment script
.\deploy_azure.ps1

# The script will:
# ✅ Create all Azure resources
# ✅ Build and push Docker images
# ✅ Deploy to Kubernetes
# ✅ Configure monitoring
# ✅ Generate credentials file
```

### Option B: Custom Deployment

```powershell
# Deploy with custom parameters
.\deploy_azure.ps1 `
    -ResourceGroup "my-custom-rg" `
    -Location "westus2" `
    -ACRName "mycustomacr" `
    -AKSCluster "my-aks-cluster"
```

---

## Step 3: Verify Deployment (5 minutes)

### Check Deployment Status

```powershell
# Get all pods
kubectl get pods --namespace jpmorgan-financial

# Get all services
kubectl get services --namespace jpmorgan-financial

# Get external IP (may take a few minutes)
kubectl get service api-gateway --namespace jpmorgan-financial --watch
```

### Test API Endpoints

```powershell
# Get the external IP
$EXTERNAL_IP = kubectl get service api-gateway --namespace jpmorgan-financial -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Test health endpoint
curl "http://${EXTERNAL_IP}/health"

# Test dashboard
Start-Process "http://${EXTERNAL_IP}"
```

---

## 🎉 You're Done!

Your JPMorgan Financial APIs are now running on Azure!

### What Was Created:

| Resource | Purpose | Cost/Month |
|----------|---------|------------|
| **AKS Cluster** | Kubernetes orchestration | ~$200 |
| **Container Registry** | Docker images | ~$5 |
| **PostgreSQL** | Database | ~$150 |
| **Redis Cache** | Caching | ~$75 |
| **Storage Account** | File storage | ~$20 |
| **Key Vault** | Secrets management | ~$0.03 |
| **Monitoring** | Logs & metrics | ~$50 |
| **Total** | | **~$500/month** |

---

## 📊 Access Your Services

### Dashboard
```
http://YOUR_EXTERNAL_IP/
```

### API Gateway
```
http://YOUR_EXTERNAL_IP/api/
```

### Prometheus
```
http://YOUR_EXTERNAL_IP/prometheus/
```

### Grafana
```
http://YOUR_EXTERNAL_IP/grafana/
```

---

## 🔐 Important Security Steps

### 1. Retrieve Credentials

```powershell
# Credentials are saved in:
cat ..\azure_credentials.txt

# IMPORTANT: Store these securely and delete the file!
```

### 2. Configure SSL/TLS

```powershell
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Configure Let's Encrypt (free SSL)
# See AZURE_DEPLOYMENT_GUIDE.md for details
```

### 3. Set Up Custom Domain

```powershell
# Create DNS A record pointing to your external IP
# Example: api.yourcompany.com -> YOUR_EXTERNAL_IP

# Update ingress with your domain
kubectl edit ingress --namespace jpmorgan-financial
```

---

## 📈 Monitoring & Management

### Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your resource group: `jpmorgan-financial-apis-rg`
3. View all resources and metrics

### Kubernetes Dashboard

```powershell
# Install Kubernetes dashboard
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Create admin user
kubectl create serviceaccount dashboard-admin-sa --namespace kube-system
kubectl create clusterrolebinding dashboard-admin-sa --clusterrole=cluster-admin --serviceaccount=kube-system:dashboard-admin-sa

# Get access token
kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | grep dashboard-admin-sa | awk '{print $1}')

# Start proxy
kubectl proxy

# Access dashboard at:
# http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

### Application Insights

```powershell
# View in Azure Portal
az monitor app-insights component show `
    --app jpmorgan-financial-insights `
    --resource-group jpmorgan-financial-apis-rg
```

---

## 🔧 Common Tasks

### Scale Your Application

```powershell
# Scale a deployment
kubectl scale deployment dashboard --replicas=5 --namespace jpmorgan-financial

# Enable auto-scaling
kubectl autoscale deployment dashboard `
    --cpu-percent=70 `
    --min=2 `
    --max=10 `
    --namespace jpmorgan-financial
```

### Update Application

```powershell
# Build new image
docker build -t jpmorganfinancialacr.azurecr.io/jpmorgan-dashboard:v2 .

# Push to ACR
docker push jpmorganfinancialacr.azurecr.io/jpmorgan-dashboard:v2

# Update deployment
kubectl set image deployment/dashboard `
    dashboard=jpmorganfinancialacr.azurecr.io/jpmorgan-dashboard:v2 `
    --namespace jpmorgan-financial

# Check rollout status
kubectl rollout status deployment/dashboard --namespace jpmorgan-financial
```

### View Logs

```powershell
# View pod logs
kubectl logs -f POD_NAME --namespace jpmorgan-financial

# View logs for all pods of a deployment
kubectl logs -f deployment/dashboard --namespace jpmorgan-financial

# View logs in Azure Portal
# Navigate to: Container Insights > Logs
```

### Backup Database

```powershell
# Create database backup
az postgres flexible-server backup create `
    --resource-group jpmorgan-financial-apis-rg `
    --name jpmorgan-financial-db `
    --backup-name manual-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')
```

---

## 💰 Cost Management

### Monitor Costs

```powershell
# View current costs
az consumption usage list `
    --start-date $(Get-Date).AddDays(-30).ToString('yyyy-MM-dd') `
    --end-date $(Get-Date).ToString('yyyy-MM-dd')

# Set up budget alerts in Azure Portal:
# Cost Management + Billing > Budgets > Add
```

### Optimize Costs

1. **Use Reserved Instances** - Save up to 72%
   ```powershell
   # Purchase 1-year or 3-year reservations
   # Azure Portal > Reservations > Purchase
   ```

2. **Auto-shutdown for Dev/Test**
   ```powershell
   # Stop AKS cluster during off-hours
   az aks stop --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-aks
   
   # Start when needed
   az aks start --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-aks
   ```

3. **Use Spot Instances** - Save up to 90%
   ```powershell
   # Add spot node pool for non-critical workloads
   az aks nodepool add `
       --resource-group jpmorgan-financial-apis-rg `
       --cluster-name jpmorgan-financial-aks `
       --name spotpool `
       --priority Spot `
       --eviction-policy Delete `
       --spot-max-price -1 `
       --node-count 2
   ```

---

## 🆘 Troubleshooting

### Pods Not Starting

```powershell
# Check pod status
kubectl get pods --namespace jpmorgan-financial

# Describe pod for details
kubectl describe pod POD_NAME --namespace jpmorgan-financial

# Check logs
kubectl logs POD_NAME --namespace jpmorgan-financial
```

### Database Connection Issues

```powershell
# Test database connection
az postgres flexible-server connect `
    --name jpmorgan-financial-db `
    --admin-user jpmadmin `
    --admin-password YOUR_PASSWORD

# Check firewall rules
az postgres flexible-server firewall-rule list `
    --resource-group jpmorgan-financial-apis-rg `
    --name jpmorgan-financial-db
```

### Image Pull Errors

```powershell
# Verify ACR integration
az aks check-acr `
    --resource-group jpmorgan-financial-apis-rg `
    --name jpmorgan-financial-aks `
    --acr jpmorganfinancialacr

# Re-attach ACR if needed
az aks update `
    --resource-group jpmorgan-financial-apis-rg `
    --name jpmorgan-financial-aks `
    --attach-acr jpmorganfinancialacr
```

### External IP Pending

```powershell
# Check service status
kubectl describe service api-gateway --namespace jpmorgan-financial

# If stuck, delete and recreate
kubectl delete service api-gateway --namespace jpmorgan-financial
kubectl apply -f microservices/deployment/kubernetes/api-gateway.yaml --namespace jpmorgan-financial
```

---

## 🔄 Cleanup (Delete Everything)

### Delete All Resources

```powershell
# WARNING: This will delete everything!
az group delete --name jpmorgan-financial-apis-rg --yes --no-wait

# Verify deletion
az group list --output table
```

### Delete Specific Resources

```powershell
# Delete AKS cluster only
az aks delete --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-aks --yes

# Delete database only
az postgres flexible-server delete --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-db --yes
```

---

## 📚 Next Steps

1. **Configure Custom Domain** - Set up your own domain name
2. **Enable SSL/TLS** - Secure your APIs with HTTPS
3. **Set Up CI/CD** - Automate deployments with Azure DevOps
4. **Configure Backups** - Set up automated backup schedules
5. **Enable Monitoring Alerts** - Get notified of issues
6. **Implement Auto-Scaling** - Handle traffic spikes automatically
7. **Set Up Disaster Recovery** - Multi-region deployment

---

## 📞 Support & Resources

- **Azure Documentation**: https://docs.microsoft.com/azure/
- **Azure Support**: https://azure.microsoft.com/support/
- **Community Forums**: https://docs.microsoft.com/answers/
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/azure

---

## ✅ Deployment Checklist

- [ ] Azure account created
- [ ] Azure CLI installed
- [ ] Logged in to Azure
- [ ] Deployment script executed
- [ ] All pods running
- [ ] External IP obtained
- [ ] API endpoints tested
- [ ] Dashboard accessible
- [ ] Credentials saved securely
- [ ] SSL/TLS configured
- [ ] Custom domain set up
- [ ] Monitoring configured
- [ ] Backup schedule set
- [ ] Cost alerts configured

---

**Quick Start Guide Version**: 1.0.0  
**Last Updated**: 2024  
**Estimated Setup Time**: 45-60 minutes  
**Difficulty**: Beginner-Friendly
