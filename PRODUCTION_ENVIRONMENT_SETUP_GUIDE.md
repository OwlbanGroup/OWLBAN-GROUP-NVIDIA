# 🚀 PRODUCTION ENVIRONMENT SETUP GUIDE
## JP Morgan Live Transaction System

**Created:** January 2, 2026  
**Environment:** Production  
**Purpose:** Complete setup guide for live transactions

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites](#prerequisites)
2. [Environment Files](#environment-files)
3. [Azure Key Vault Setup](#azure-key-vault-setup)
4. [JP Morgan Production Credentials](#jp-morgan-production-credentials)
5. [mTLS Certificate Setup](#mtls-certificate-setup)
6. [Database Configuration](#database-configuration)
7. [Security Configuration](#security-configuration)
8. [Deployment Steps](#deployment-steps)
9. [Verification & Testing](#verification--testing)
10. [Monitoring Setup](#monitoring-setup)
11. [Troubleshooting](#troubleshooting)

---

## 🔐 PREREQUISITES

### **1. JP Morgan Production Access**
- ✅ Completed KYC/KYB process
- ✅ Production API access approved
- ✅ OAuth2 credentials received
- ✅ IP addresses allowlisted
- ✅ mTLS certificates exchanged (if required)

### **2. Azure Resources**
- ✅ Azure subscription active
- ✅ Resource group created
- ✅ Key Vault provisioned
- ✅ PostgreSQL database created
- ✅ Redis cache provisioned
- ✅ Application Insights configured

### **3. Required Credentials**
- ✅ JP Morgan production client ID
- ✅ JP Morgan production client secret
- ✅ Database credentials
- ✅ API keys generated
- ✅ HMAC secret generated
- ✅ Webhook secret generated

---

## 📁 ENVIRONMENT FILES

### **File Structure:**
```
nestjs-backend/
├── .env.production          # Production environment variables
├── .env.production.example  # Template (safe to commit)
├── .env.staging             # Staging environment
├── .env.development         # Development environment
└── .env.local               # Local overrides (gitignored)
```

### **Created Files:**
✅ `.env.production` - Complete production configuration (400+ variables)

### **Key Sections:**
1. **Application Settings** - Node environment, port, app info
2. **Database Configuration** - PostgreSQL connection with SSL
3. **JP Morgan API** - Production endpoints and credentials
4. **mTLS Configuration** - Certificate paths and settings
5. **HMAC Signing** - Request signing configuration
6. **Security** - API keys, IP allowlisting, CORS
7. **Monitoring** - Prometheus, Grafana, Application Insights
8. **Logging** - Log levels, formats, destinations
9. **Alerting** - Email, Slack, PagerDuty
10. **Payment Limits** - Transaction and daily limits
11. **Approval Workflow** - Thresholds and settings
12. **Fraud Detection** - Velocity checks, anomaly detection
13. **Performance** - Clustering, caching, Redis
14. **Feature Flags** - Enable/disable features

---

## 🔑 AZURE KEY VAULT SETUP

### **Step 1: Create Key Vault**
```bash
# Create resource group
az group create \
  --name jpmorgan-prod-rg \
  --location eastus

# Create Key Vault
az keyvault create \
  --name jpmorgan-prod-kv \
  --resource-group jpmorgan-prod-rg \
  --location eastus \
  --enable-rbac-authorization true
```

### **Step 2: Store Secrets**
```bash
# JP Morgan credentials
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name JPM-CLIENT-ID \
  --value "your-production-client-id"

az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name JPM-CLIENT-SECRET \
  --value "your-production-client-secret"

# Database password
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name DB-PASSWORD \
  --value "your-secure-db-password"

# HMAC secret
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name HMAC-SECRET \
  --value "$(openssl rand -base64 32)"

# API keys
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name API-KEY-ADMIN \
  --value "$(openssl rand -hex 32)"

az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name API-KEY-MAKER \
  --value "$(openssl rand -hex 32)"

az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name API-KEY-CHECKER \
  --value "$(openssl rand -hex 32)"

# Webhook secret
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name JPM-WEBHOOK-SECRET \
  --value "$(openssl rand -base64 32)"
```

### **Step 3: Grant Access**
```bash
# Get your app's managed identity
APP_IDENTITY=$(az webapp identity show \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg \
  --query principalId -o tsv)

# Grant Key Vault access
az role assignment create \
  --role "Key Vault Secrets User" \
  --assignee $APP_IDENTITY \
  --scope /subscriptions/{subscription-id}/resourceGroups/jpmorgan-prod-rg/providers/Microsoft.KeyVault/vaults/jpmorgan-prod-kv
```

---

## 🏦 JP MORGAN PRODUCTION CREDENTIALS

### **Credentials Checklist:**

#### **1. OAuth2 Credentials**
```bash
# Received from JP Morgan after onboarding
JPM_PROD_CLIENT_ID=prod_xxxxxxxxxxxxxxxx
JPM_PROD_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### **2. API Endpoints**
```bash
# Production URLs (provided by JP Morgan)
JPM_PROD_TOKEN_URL=https://api.jpmorgan.com/oauth2/access_token
JPM_PROD_BASE_URL=https://api.jpmorgan.com/v1
JPM_PROD_ACH_URL=https://api.jpmorgan.com/v1/ach
JPM_PROD_WIRE_URL=https://api.jpmorgan.com/v1/wire
JPM_PROD_RTP_URL=https://api.jpmorgan.com/v1/rtp
```

#### **3. Scopes**
```bash
# Required scopes for live transactions
JPM_PROD_SCOPES=payments:read payments:write ach:originate wire:send rtp:send
```

#### **4. Webhook Configuration**
```bash
# Your webhook endpoint (must be HTTPS)
JPM_PROD_WEBHOOK_URL=https://your-domain.com/api/webhooks/jpmorgan

# Webhook secret (for signature verification)
JPM_PROD_WEBHOOK_SECRET=your-webhook-secret
```

---

## 🔐 MTLS CERTIFICATE SETUP

### **Step 1: Generate Certificate Signing Request (CSR)**
```bash
# Create private key
openssl genrsa -out client-key.pem 2048

# Create CSR
openssl req -new -key client-key.pem -out client-csr.pem \
  -subj "/C=US/ST=New York/L=New York/O=Your Company/CN=jpmorgan-api-client"
```

### **Step 2: Submit CSR to JP Morgan**
1. Log into JP Morgan API portal
2. Navigate to Certificates section
3. Upload `client-csr.pem`
4. Wait for approval (1-3 business days)
5. Download signed certificate `client-cert.pem`
6. Download CA certificate `ca-cert.pem`

### **Step 3: Store Certificates Securely**
```bash
# Create certificates directory
mkdir -p /app/certs/production

# Copy certificates (with restricted permissions)
cp client-cert.pem /app/certs/production/
cp client-key.pem /app/certs/production/
cp ca-cert.pem /app/certs/production/

# Set permissions (read-only for app user)
chmod 400 /app/certs/production/*
chown app-user:app-user /app/certs/production/*
```

### **Step 4: Verify Certificate**
```bash
# Verify certificate
openssl x509 -in /app/certs/production/client-cert.pem -text -noout

# Test mTLS connection
curl --cert /app/certs/production/client-cert.pem \
     --key /app/certs/production/client-key.pem \
     --cacert /app/certs/production/ca-cert.pem \
     https://api.jpmorgan.com/v1/health
```

---

## 💾 DATABASE CONFIGURATION

### **Step 1: Create Production Database**
```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --name jpmorgan-prod-db \
  --resource-group jpmorgan-prod-rg \
  --location eastus \
  --admin-user jpmorgan_admin \
  --admin-password "SecurePassword123!" \
  --sku-name Standard_D4s_v3 \
  --tier GeneralPurpose \
  --storage-size 256 \
  --version 14

# Create database
az postgres flexible-server db create \
  --resource-group jpmorgan-prod-rg \
  --server-name jpmorgan-prod-db \
  --database-name jpmorgan_payments_prod
```

### **Step 2: Configure Firewall Rules**
```bash
# Add your application's IP addresses
az postgres flexible-server firewall-rule create \
  --resource-group jpmorgan-prod-rg \
  --name jpmorgan-prod-db \
  --rule-name AllowAppServers \
  --start-ip-address 10.0.1.0 \
  --end-ip-address 10.0.1.255
```

### **Step 3: Enable SSL**
```bash
# Enforce SSL connections
az postgres flexible-server parameter set \
  --resource-group jpmorgan-prod-rg \
  --server-name jpmorgan-prod-db \
  --name require_secure_transport \
  --value ON
```

### **Step 4: Run Migrations**
```bash
# Set environment
export NODE_ENV=production

# Run migrations
npm run migration:run

# Verify
npm run migration:show
```

---

## 🛡️ SECURITY CONFIGURATION

### **1. IP Allowlisting**

**Update .env.production:**
```bash
# Add your production IP addresses
ALLOWED_IPS=10.0.1.0/24,10.0.2.0/24,52.168.0.0/16
IP_ALLOWLIST_ENABLED=true
```

**Register IPs with JP Morgan:**
1. Log into JP Morgan API portal
2. Navigate to Security > IP Allowlist
3. Add each production IP address
4. Wait for approval (1-2 business days)

### **2. API Key Generation**

```bash
# Generate secure API keys
ADMIN_KEY=$(openssl rand -hex 32)
MAKER_KEY=$(openssl rand -hex 32)
CHECKER_KEY=$(openssl rand -hex 32)
VIEWER_KEY=$(openssl rand -hex 32)

# Store in Key Vault
az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-ADMIN --value "$ADMIN_KEY"
az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-MAKER --value "$MAKER_KEY"
az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-CHECKER --value "$CHECKER_KEY"
az keyvault secret set --vault-name jpmorgan-prod-kv --name API-KEY-VIEWER --value "$VIEWER_KEY"

# Distribute to authorized users securely
echo "Admin API Key: $ADMIN_KEY" | gpg --encrypt --recipient admin@company.com
```

### **3. HMAC Secret Generation**

```bash
# Generate HMAC secret
HMAC_SECRET=$(openssl rand -base64 32)

# Store in Key Vault
az keyvault secret set \
  --vault-name jpmorgan-prod-kv \
  --name HMAC-SECRET \
  --value "$HMAC_SECRET"
```

### **4. CORS Configuration**

**Update .env.production:**
```bash
CORS_ENABLED=true
CORS_ORIGIN=https://your-frontend-domain.com,https://admin.your-domain.com
CORS_CREDENTIALS=true
```

---

## 🚀 DEPLOYMENT STEPS

### **Step 1: Build Application**
```bash
# Install dependencies
npm ci --production

# Build TypeScript
npm run build

# Run tests
npm run test:prod
```

### **Step 2: Create Docker Image**
```bash
# Build production image
docker build -t jpmorgan-payments:prod -f Dockerfile.prod .

# Tag for registry
docker tag jpmorgan-payments:prod \
  yourregistry.azurecr.io/jpmorgan-payments:latest

# Push to registry
docker push yourregistry.azurecr.io/jpmorgan-payments:latest
```

### **Step 3: Deploy to Azure App Service**
```bash
# Create App Service Plan
az appservice plan create \
  --name jpmorgan-prod-plan \
  --resource-group jpmorgan-prod-rg \
  --sku P2V2 \
  --is-linux

# Create Web App
az webapp create \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg \
  --plan jpmorgan-prod-plan \
  --deployment-container-image-name yourregistry.azurecr.io/jpmorgan-payments:latest

# Enable managed identity
az webapp identity assign \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg

# Configure app settings
az webapp config appsettings set \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg \
  --settings @appsettings.json
```

### **Step 4: Configure Environment Variables**
```bash
# Load from Key Vault
az webapp config appsettings set \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg \
  --settings \
    JPM_PROD_CLIENT_ID="@Microsoft.KeyVault(SecretUri=https://jpmorgan-prod-kv.vault.azure.net/secrets/JPM-CLIENT-ID/)" \
    JPM_PROD_CLIENT_SECRET="@Microsoft.KeyVault(SecretUri=https://jpmorgan-prod-kv.vault.azure.net/secrets/JPM-CLIENT-SECRET/)" \
    DATABASE_PASSWORD="@Microsoft.KeyVault(SecretUri=https://jpmorgan-prod-kv.vault.azure.net/secrets/DB-PASSWORD/)"
```

### **Step 5: Start Application**
```bash
# Restart app
az webapp restart \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg

# Check logs
az webapp log tail \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg
```

---

## ✅ VERIFICATION & TESTING

### **1. Health Check**
```bash
# Check application health
curl https://jpmorgan-payments-app.azurewebsites.net/health

# Expected response:
{
  "status": "ok",
  "info": {
    "database": { "status": "up" },
    "jpmorgan": { "status": "up" },
    "redis": { "status": "up" }
  }
}
```

### **2. JP Morgan Connectivity**
```bash
# Test OAuth2 token acquisition
curl -X POST https://jpmorgan-payments-app.azurewebsites.net/api/jpmorgan/test-connection \
  -H "X-API-Key: your-admin-api-key"

# Expected response:
{
  "success": true,
  "tokenAcquired": true,
  "apiConnectivity": "ok"
}
```

### **3. Database Connectivity**
```bash
# Test database connection
curl https://jpmorgan-payments-app.azurewebsites.net/api/health/database \
  -H "X-API-Key: your-admin-api-key"

# Expected response:
{
  "status": "connected",
  "latency": "15ms"
}
```

### **4. mTLS Verification**
```bash
# Test mTLS handshake
curl https://jpmorgan-payments-app.azurewebsites.net/api/jpmorgan/test-mtls \
  -H "X-API-Key: your-admin-api-key"

# Expected response:
{
  "mtlsEnabled": true,
  "certificateValid": true,
  "expiresAt": "2027-01-01T00:00:00Z"
}
```

### **5. End-to-End Test**
```bash
# Create test ACH payment (sandbox mode first)
curl -X POST https://jpmorgan-payments-app.azurewebsites.net/api/ach/payments \
  -H "X-API-Key: your-maker-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "secCode": "PPD",
    "transactionType": "CREDIT",
    "amountCents": 10000,
    "originatorName": "Test Company",
    "originatorId": "1234567890",
    "receiverName": "John Doe",
    "receiverAccountNumber": "123456789",
    "receiverRoutingNumber": "021000021",
    "idempotencyKey": "test-payment-001"
  }'
```

---

## 📊 MONITORING SETUP

### **1. Application Insights**
```bash
# Create Application Insights
az monitor app-insights component create \
  --app jpmorgan-payments-insights \
  --location eastus \
  --resource-group jpmorgan-prod-rg \
  --application-type web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app jpmorgan-payments-insights \
  --resource-group jpmorgan-prod-rg \
  --query instrumentationKey -o tsv)

# Configure app
az webapp config appsettings set \
  --name jpmorgan-payments-app \
  --resource-group jpmorgan-prod-rg \
  --settings APPINSIGHTS_INSTRUMENTATION_KEY="$INSTRUMENTATION_KEY"
```

### **2. Prometheus & Grafana**
```bash
# Deploy Prometheus
kubectl apply -f kubernetes/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f kubernetes/grafana-deployment.yaml

# Import dashboards
curl -X POST http://grafana-url/api/dashboards/import \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @grafana-live-transaction-dashboard.json
```

### **3. Alert Rules**
```bash
# Create alert for failed payments
az monitor metrics alert create \
  --name payment-failures \
  --resource-group jpmorgan-prod-rg \
  --scopes /subscriptions/{sub-id}/resourceGroups/jpmorgan-prod-rg/providers/Microsoft.Web/sites/jpmorgan-payments-app \
  --condition "count customMetrics/payment_failures > 10" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action email ops@company.com
```

---

## 🔧 TROUBLESHOOTING

### **Common Issues:**

#### **1. OAuth2 Token Acquisition Fails**
```bash
# Check credentials
az keyvault secret show --vault-name jpmorgan-prod-kv --name JPM-CLIENT-ID
az keyvault secret show --vault-name jpmorgan-prod-kv --name JPM-CLIENT-SECRET

# Verify token URL
curl -X POST https://api.jpmorgan.com/oauth2/access_token \
  -d "grant_type=client_credentials" \
  -d "client_id=your-client-id" \
  -d "client_secret=your-client-secret"
```

#### **2. mTLS Handshake Fails**
```bash
# Verify certificate
openssl x509 -in /app/certs/production/client-cert.pem -text -noout

# Check expiration
openssl x509 -in /app/certs/production/client-cert.pem -noout -dates

# Test connection
openssl s_client -connect api.jpmorgan.com:443 \
  -cert /app/certs/production/client-cert.pem \
  -key /app/certs/production/client-key.pem
```

#### **3. Database Connection Issues**
```bash
# Test connection
psql "host=jpmorgan-prod-db.postgres.database.azure.com port=5432 dbname=jpmorgan_payments_prod user=jpmorgan_app_user password=xxx sslmode=require"

# Check firewall rules
az postgres flexible-server firewall-rule list \
  --resource-group jpmorgan-prod-rg \
  --name jpmorgan-prod-db
```

#### **4. IP Not Allowlisted**
```bash
# Check your public IP
curl https://api.ipify.org

# Verify with JP Morgan
# Log into JP Morgan portal > Security > IP Allowlist
# Ensure your IP is listed and approved
```

---

## 📋 POST-DEPLOYMENT CHECKLIST

### **Security:**
- [ ] All secrets stored in Key Vault
- [ ] mTLS certificates installed and verified
- [ ] IP allowlisting configured
- [ ] API keys distributed securely
- [ ] CORS configured correctly
- [ ] Rate limiting enabled
- [ ] Audit logging active

### **JP Morgan:**
- [ ] Production credentials configured
- [ ] OAuth2 token acquisition working
- [ ] API connectivity verified
- [ ] Webhook endpoint registered
- [ ] IP addresses allowlisted
- [ ] mTLS handshake successful

### **Infrastructure:**
- [ ] Database migrations complete
- [ ] Redis cache connected
- [ ] Application Insights configured
- [ ] Prometheus metrics exporting
- [ ] Grafana dashboards imported
- [ ] Alert rules configured

### **Testing:**
- [ ] Health checks passing
- [ ] End-to-end payment flow tested
- [ ] Approval workflows tested
- [ ] Webhook handling tested
- [ ] Error scenarios tested
- [ ] Load testing completed

### **Documentation:**
- [ ] Runbook created
- [ ] API documentation updated
- [ ] Team trained
- [ ] On-call rotation established
- [ ] Incident response plan ready

---

## 🎯 NEXT STEPS

1. **Monitor for 24 hours** - Watch logs and metrics
2. **Process test transactions** - Verify end-to-end flows
3. **Enable production features** - Turn on live transaction processing
4. **Schedule JP Morgan certification** - Complete final testing
5. **Go live** - Begin processing real payments

---

**Document Status:** ✅ COMPLETE  
**Last Updated:** January 2, 2026  
**Maintained By:** DevOps Team  
**Review Schedule:** Monthly
