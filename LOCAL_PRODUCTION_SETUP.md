# Local Production Environment Setup

## 🚀 Quick Start - Create Live Production Environment

Since Azure CLI is not yet installed, let's first set up a **local production environment** that mimics the cloud setup. This will allow you to test everything before deploying to Azure.

---

## Option 1: Local Production Environment (Immediate)

### Step 1: Install Prerequisites (5 minutes)

```powershell
# Check if Docker is installed
docker --version

# If not installed, download Docker Desktop:
# https://www.docker.com/products/docker-desktop

# Check if Docker Compose is available
docker-compose --version
```

### Step 2: Start Local Production Environment (2 minutes)

```powershell
# Navigate to project root
cd c:\Users\bizle\Desktop\jpmorgan_financial_apis

# Start all services in production mode
docker-compose -f docker-compose.production.yml up -d

# This will start:
# - PostgreSQL database
# - Redis cache
# - Prometheus monitoring
# - Grafana dashboards
# - AlertManager
# - All 12 microservices
# - Dashboard with live production data
```

### Step 3: Verify Services (2 minutes)

```powershell
# Check all containers are running
docker-compose -f docker-compose.production.yml ps

# Check logs
docker-compose -f docker-compose.production.yml logs -f

# Access services:
# Dashboard: http://localhost:8010
# API Gateway: http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

### Step 4: Test Live Production Data

```powershell
# Test health endpoint
curl http://localhost:8010/health

# Test production metrics
curl http://localhost:8010/api/production/metrics

# Open dashboard in browser
Start-Process "http://localhost:8010"
```

---

## Option 2: Azure Cloud Deployment (Requires Setup)

### Prerequisites Installation

#### 1. Install Azure CLI (10 minutes)

```powershell
# Download and install Azure CLI
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'

# Restart PowerShell after installation

# Verify installation
az --version
```

#### 2. Install Docker Desktop (if not installed)

```powershell
# Download from: https://www.docker.com/products/docker-desktop
# Or use winget:
winget install Docker.DockerDesktop
```

#### 3. Login to Azure

```powershell
# Login to Azure account
az login

# This will open a browser for authentication
# After login, select your subscription

# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### Azure Deployment Steps

Once prerequisites are installed:

```powershell
# Navigate to scripts directory
cd c:\Users\bizle\Desktop\jpmorgan_financial_apis\scripts

# Run automated deployment
.\deploy_azure.ps1

# Deployment will take 45-60 minutes
# It will create all Azure resources automatically
```

---

## 🎯 Recommended Approach

### For Immediate Testing:
**Use Option 1 (Local Production Environment)**
- No Azure account needed
- No costs involved
- Immediate deployment
- Full feature testing
- Perfect for development and testing

### For Production Deployment:
**Use Option 2 (Azure Cloud)**
- Requires Azure account
- ~$600/month cost (The Owlban Group pays)
- Scalable and reliable
- Enterprise-grade infrastructure
- 99.9% uptime SLA

---

## 📊 What You Get with Local Production Environment

### Services Running:
✅ PostgreSQL Database (Port 5432)  
✅ Redis Cache (Port 6379)  
✅ Prometheus Monitoring (Port 9090)  
✅ Grafana Dashboards (Port 3000)  
✅ AlertManager (Port 9093)  
✅ API Gateway (Port 8000)  
✅ Dashboard with Live Data (Port 8010)  
✅ Auth Service (Port 8001)  
✅ Payroll Service (Port 8002)  
✅ Benefits Service (Port 8003)  
✅ Bill-Pay Service (Port 8004)  
✅ Purchasing Service (Port 8005)  
✅ ML Service (Port 8006)  
✅ Patterns Service (Port 8007)  
✅ Traction Service (Port 8008)  
✅ Telemetry Service (Port 8009)  
✅ Storage Service (Port 8011)  

### Features Available:
✅ Real-time metrics from Prometheus  
✅ Live telemetry event streaming  
✅ WebSocket auto-updates  
✅ System health monitoring  
✅ Production alerts  
✅ Interactive charts  
✅ All microservices operational  

---

## 🔧 Management Commands

### Start Environment
```powershell
docker-compose -f docker-compose.production.yml up -d
```

### Stop Environment
```powershell
docker-compose -f docker-compose.production.yml down
```

### View Logs
```powershell
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker-compose -f docker-compose.production.yml logs -f dashboard
```

### Restart Service
```powershell
docker-compose -f docker-compose.production.yml restart dashboard
```

### Check Status
```powershell
docker-compose -f docker-compose.production.yml ps
```

### Clean Up (Remove all data)
```powershell
docker-compose -f docker-compose.production.yml down -v
```

---

## 🎉 Quick Start Summary

**To create live production environment NOW:**

```powershell
# 1. Navigate to project
cd c:\Users\bizle\Desktop\jpmorgan_financial_apis

# 2. Start production environment
docker-compose -f docker-compose.production.yml up -d

# 3. Wait 2-3 minutes for services to start

# 4. Open dashboard
Start-Process "http://localhost:8010"

# 5. Access Prometheus
Start-Process "http://localhost:9090"

# 6. Access Grafana
Start-Process "http://localhost:3000"
```

**That's it! Your live production environment is ready!**

---

## 📝 Next Steps

1. **Test locally first** - Use docker-compose
2. **Verify all features** - Check dashboard, metrics, alerts
3. **Install Azure CLI** - When ready for cloud deployment
4. **Deploy to Azure** - Run deployment script
5. **Monitor costs** - Track Azure spending

---

**Environment**: Local Production (Docker Compose)  
**Cost**: $0 (runs on your machine)  
**Setup Time**: 5 minutes  
**Perfect For**: Development, Testing, Demo
