# 🚀 GO-LIVE EXECUTION RUNBOOK
## JP Morgan Live Transaction System

**Created:** January 2, 2026  
**Version:** 1.0  
**Status:** READY FOR EXECUTION  
**Estimated Duration:** 4-6 hours

---

## ⚠️ CRITICAL WARNINGS

### **BEFORE YOU BEGIN:**
1. ✅ **Pre-Flight Checklist MUST be 100% complete** (see PRE_FLIGHT_READINESS_CHECKLIST.md)
2. ✅ **All stakeholders notified** (Treasury, Compliance, IT, Management)
3. ✅ **Rollback plan reviewed** and team trained
4. ✅ **Maintenance window scheduled** (recommended: off-hours)
5. ✅ **On-call team available** for 24 hours post-deployment
6. ✅ **JP Morgan support contact** information ready

### **STOP CONDITIONS:**
If ANY of these occur, STOP and execute rollback:
- ❌ OAuth2 token acquisition fails
- ❌ mTLS handshake fails
- ❌ Database connection fails
- ❌ Health checks fail
- ❌ Test transaction fails
- ❌ Any security validation fails

---

## 📋 EXECUTION TEAM

### **Required Roles:**
- **Deployment Lead** - Coordinates execution
- **DevOps Engineer** - Executes commands
- **Database Administrator** - Manages database
- **Security Engineer** - Validates security
- **Application Developer** - Troubleshoots issues
- **Treasury Representative** - Approves go-live
- **Compliance Officer** - Validates compliance

### **Communication Channels:**
- **Primary:** Slack #jpmorgan-golive
- **Escalation:** Phone bridge (number: _________)
- **JP Morgan Support:** (number: _________)

---

## 🕐 TIMELINE

### **T-24 Hours: Final Preparation**
- Review this runbook with entire team
- Verify all prerequisites
- Schedule maintenance window
- Notify all stakeholders

### **T-2 Hours: Pre-Deployment**
- Execute pre-flight checklist
- Backup current system
- Verify rollback procedures

### **T-0: Go-Live Execution**
- Follow steps 1-10 below
- Duration: 4-6 hours

### **T+24 Hours: Post-Deployment**
- Monitor system continuously
- Process test transactions
- Validate all metrics

---

## 📝 EXECUTION STEPS

## **STEP 1: BACKUP CURRENT SYSTEM** (15 minutes)

### **1.1 Backup Database**
```bash
# Connect to database server
ssh dbadmin@prod-db-server

# Create backup
pg_dump -h jpmorgan-prod-db.postgres.database.azure.com \
  -U jpmorgan_admin \
  -d jpmorgan_payments_prod \
  -F c \
  -f /backups/jpmorgan_payments_$(date +%Y%m%d_%H%M%S).backup

# Verify backup
ls -lh /backups/jpmorgan_payments_*.backup

# Store backup location
BACKUP_FILE="/backups/jpmorgan_payments_$(date +%Y%m%d_%H%M%S).backup"
echo "Backup stored at: $BACKUP_FILE"
```

**✅ Verification:**
- [ ] Backup file created successfully
- [ ] Backup file size > 0
- [ ] Backup location documented

**❌ If Failed:** Cannot proceed without backup

---

### **1.2 Backup Application Code**
```bash
# Tag current version
cd /app/jpmorgan-payments
git tag -a v1.0-pre-production -m "Pre-production backup"
git push origin v1.0-pre-production

# Create code archive
tar -czf /backups/jpmorgan-app-$(date +%Y%m%d_%H%M%S).tar.gz /app/jpmorgan-payments

# Verify
ls -lh /backups/jpmorgan-app-*.tar.gz
```

**✅ Verification:**
- [ ] Git tag created
- [ ] Archive created successfully
- [ ] Archive location documented

---

### **1.3 Backup Configuration**
```bash
# Backup current environment
cp /app/jpmorgan-payments/.env /backups/.env.backup.$(date +%Y%m%d_%H%M%S)

# Backup certificates
cp -r /app/certs /backups/certs.backup.$(date +%Y%m%d_%H%M%S)

# Verify
ls -lh /backups/
```

**✅ Verification:**
- [ ] Environment file backed up
- [ ] Certificates backed up
- [ ] All backups documented

---

## **STEP 2: PROVISION AZURE RESOURCES** (30 minutes)

### **2.1 Create Resource Group**
```bash
# Set variables
RESOURCE_GROUP="jpmorgan-prod-rg"
LOCATION="eastus"
SUBSCRIPTION_ID="your-subscription-id"

# Set subscription
az account set --subscription $SUBSCRIPTION_ID

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Verify
az group show --name $RESOURCE_GROUP
```

**✅ Verification:**
- [ ] Resource group created
- [ ] Location correct
- [ ] Subscription correct

---

### **2.2 Create Azure Key Vault**
```bash
# Create Key Vault
az keyvault create \
  --name jpmorgan-prod-kv \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --enable-rbac-authorization true \
  --sku premium

# Verify
az keyvault show --name jpmorgan-prod-kv
```

**✅ Verification:**
- [ ] Key Vault created
- [ ] RBAC enabled
- [ ] Premium SKU confirmed

---

### **2.3 Store Secrets in Key Vault**
```bash
# JP Morgan credentials (replace with actual values)
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name JPM-CLIENT-ID \
  --value "YOUR_PRODUCTION_CLIENT_ID"

az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name JPM-CLIENT-SECRET \
  --value "YOUR_PRODUCTION_CLIENT_SECRET"

# Database password
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name DB-PASSWORD \
  --value "YOUR_SECURE_DB_PASSWORD"

# Generate and store HMAC secret
HMAC_SECRET=$(openssl rand -base64 32)
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name HMAC-SECRET \
  --value "$HMAC_SECRET"

# Generate and store API keys
ADMIN_KEY=$(openssl rand -hex 32)
MAKER_KEY=$(openssl rand -hex 32)
CHECKER_KEY=$(openssl rand -hex 32)

az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-ADMIN --value "$ADMIN_KEY"
az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-MAKER --value "$MAKER_KEY"
az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-CHECKER --value "$CHECKER_KEY"

# Webhook secret
WEBHOOK_SECRET=$(openssl rand -base64 32)
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name JPM-WEBHOOK-SECRET \
  --value "$WEBHOOK_SECRET"

# Verify all secrets
az keyvault secret list --vault-name jpmorgan-prod-kv --query "[].name"
```

**✅ Verification:**
- [ ] All secrets stored
- [ ] Secret names correct
- [ ] No secrets exposed in logs

**🔐 SECURITY:** Document API keys securely and distribute to authorized users only

---

### **2.4 Create PostgreSQL Database**
```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --name jpmorgan-prod-db \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --admin-user jpmorgan_admin \
  --admin-password "$(az keyvault secret show --vault-name jpmorgan-prod-kv --name DB-PASSWORD --query value -o tsv)" \
  --sku-name Standard_D4s_v3 \
  --tier GeneralPurpose \
  --storage-size 256 \
  --version 14 \
  --high-availability Enabled \
  --backup-retention 30

# Create database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name jpmorgan-prod-db \
  --database-name jpmorgan_payments_prod

# Configure firewall (add your IPs)
az postgres flexible-server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --name jpmorgan-prod-db \
  --rule-name AllowAppServers \
  --start-ip-address 10.0.1.0 \
  --end-ip-address 10.0.1.255

# Enable SSL
az postgres flexible-server parameter set \
  --resource-group $RESOURCE_GROUP \
  --server-name jpmorgan-prod-db \
  --name require_secure_transport \
  --value ON

# Verify
az postgres flexible-server show \
  --resource-group $RESOURCE_GROUP \
  --name jpmorgan-prod-db
```

**✅ Verification:**
- [ ] Database server created
- [ ] High availability enabled
- [ ] SSL enforced
- [ ] Firewall rules configured
- [ ] Backup retention set to 30 days

---

### **2.5 Create Redis Cache**
```bash
# Create Redis cache
az redis create \
  --name jpmorgan-prod-cache \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Premium \
  --vm-size P1 \
  --enable-non-ssl-port false

# Get Redis connection info
REDIS_HOST=$(az redis show --name jpmorgan-prod-cache --resource-group $RESOURCE_GROUP --query hostName -o tsv)
REDIS_KEY=$(az redis list-keys --name jpmorgan-prod-cache --resource-group $RESOURCE_GROUP --query primaryKey -o tsv)

# Store in Key Vault
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name REDIS-PASSWORD \
  --value "$REDIS_KEY"

# Verify
az redis show --name jpmorgan-prod-cache --resource-group $RESOURCE_GROUP
```

**✅ Verification:**
- [ ] Redis cache created
- [ ] Premium SKU confirmed
- [ ] SSL-only enabled
- [ ] Connection info stored

---

## **STEP 3: DEPLOY APPLICATION** (45 minutes)

### **3.1 Build Production Image**
```bash
# Navigate to project
cd /app/jpmorgan-payments/nestjs-backend

# Install dependencies
npm ci --production

# Run tests
npm run test:prod

# Build application
npm run build

# Verify build
ls -lh dist/
```

**✅ Verification:**
- [ ] All tests passed
- [ ] Build successful
- [ ] dist/ directory created

---

### **3.2 Create Docker Image**
```bash
# Build Docker image
docker build -t jpmorgan-payments:prod -f Dockerfile.prod .

# Tag for Azure Container Registry
docker tag jpmorgan-payments:prod \
  yourregistry.azurecr.io/jpmorgan-payments:v1.0.0

docker tag jpmorgan-payments:prod \
  yourregistry.azurecr.io/jpmorgan-payments:latest

# Login to ACR
az acr login --name yourregistry

# Push images
docker push yourregistry.azurecr.io/jpmorgan-payments:v1.0.0
docker push yourregistry.azurecr.io/jpmorgan-payments:latest

# Verify
az acr repository show \
  --name yourregistry \
  --repository jpmorgan-payments
```

**✅ Verification:**
- [ ] Docker image built
- [ ] Images tagged correctly
- [ ] Images pushed to ACR
- [ ] Both v1.0.0 and latest tags present

---

### **3.3 Create App Service**
```bash
# Create App Service Plan
az appservice plan create \
  --name jpmorgan-prod-plan \
  --resource-group $RESOURCE_GROUP \
  --sku P2V2 \
  --is-linux

# Create Web App
az webapp create \
  --name jpmorgan-payments-app \
  --resource-group $RESOURCE_GROUP \
  --plan jpmorgan-prod-plan \
