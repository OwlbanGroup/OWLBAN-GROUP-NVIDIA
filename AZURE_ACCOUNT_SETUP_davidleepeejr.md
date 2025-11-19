# 🔐 AZURE ACCOUNT SETUP GUIDE
## For: davidleepeejr@owlbangroup.com

**The Owlban Group - JPMorgan Financial APIs**  
**Date**: 2024-11-19  
**Status**: New Account Setup Required  

---

## 📋 EXECUTIVE SUMMARY

This guide provides step-by-step instructions to set up the Azure account **davidleepeejr@owlbangroup.com** for deploying the JPMorgan Financial APIs to Microsoft Azure Cloud.

**Current Situation**:
- ✅ Azure CLI installed (v2.80.0)
- ✅ Previous account (bizleeper@gmail.com) logged in but no subscription
- 🆕 New corporate account: **davidleepeejr@owlbangroup.com**
- ❌ New account needs subscription setup

**Objective**: Configure the new corporate Azure account with proper subscription for production deployment.

---

## 🎯 ACCOUNT SETUP STRATEGY

### Recommended Approach: Corporate Account with Pay-As-You-Go

**Why This Account**:
- ✅ Corporate email (@owlbangroup.com) - Professional and traceable
- ✅ Proper governance and billing for The Owlban Group
- ✅ Easier to manage enterprise resources
- ✅ Better for compliance and auditing

**Subscription Type**: Pay-As-You-Go (Production-Ready)
- Estimated Cost: $550-600/month
- No upfront commitment
- Scalable for production workloads
- Can upgrade to Enterprise Agreement later

---

## 🚀 STEP-BY-STEP SETUP PROCESS

### Step 1: Logout from Current Account (2 minutes)

```powershell
# Logout from current Azure account
az logout

# Clear cached credentials
az account clear

# Verify logout
az account show
# Should show: "Please run 'az login' to setup account."
```

---

### Step 2: Create Azure Account (15 minutes)

#### Option A: Free Trial First (Recommended for Testing)

**Benefits**:
- $200 credit for 30 days
- Test deployment before production
- No charges during trial
- Easy upgrade to Pay-As-You-Go

**Steps**:

1. **Open Browser and Navigate**:
   ```
   https://azure.microsoft.com/free/
   ```

2. **Click "Start free"**

3. **Sign in with Microsoft Account**:
   - Email: **davidleepeejr@owlbangroup.com**
   - If no Microsoft account exists, create one with this email

4. **Complete Registration Form**:
   - First Name: David
   - Last Name: Lee Pee Jr
   - Country/Region: United States
   - Phone Number: (Your phone number)
   - Company: The Owlban Group
   - Job Title: (Your title)

5. **Identity Verification**:
   - Phone verification (SMS code)
   - Credit card verification (no charges during trial)

6. **Accept Terms**:
   - Review Microsoft Customer Agreement
   - Accept terms and conditions
   - Click "Sign up"

7. **Wait for Confirmation**:
   - Account creation takes 2-3 minutes
   - You'll receive confirmation email

---

#### Option B: Pay-As-You-Go (Direct to Production)

**If you want to skip trial and go directly to production**:

1. **Navigate to**:
   ```
   https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go/
   ```

2. **Click "Buy now"**

3. **Sign in**: davidleepeejr@owlbangroup.com

4. **Enter Billing Information**:
   - Company: The Owlban Group
   - Address: (Company address)
   - Tax ID: (If applicable)

5. **Add Payment Method**:
   - Credit card or bank account
   - Billing contact information

6. **Complete Setup**:
   - Review and confirm
   - Accept terms
   - Create subscription

---

### Step 3: Login with New Account (5 minutes)

```powershell
# Login to Azure with new account
az login

# This will open browser for authentication
# Sign in with: davidleepeejr@owlbangroup.com
# Enter password when prompted

# Wait for browser confirmation
# Browser will show: "You have signed in to the Microsoft Azure Cross-platform Command Line Interface application on your device."

# Close browser and return to PowerShell
```

**Expected Output**:
```json
[
  {
    "cloudName": "AzureCloud",
    "homeTenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "isDefault": true,
    "managedByTenants": [],
    "name": "Azure subscription 1" or "Free Trial",
    "state": "Enabled",
    "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "user": {
      "name": "davidleepeejr@owlbangroup.com",
      "type": "user"
    }
  }
]
```

---

### Step 4: Verify Subscription (3 minutes)

```powershell
# List all subscriptions
az account list --output table

# Expected output:
# Name                    CloudName    SubscriptionId                        State    IsDefault
# ----------------------  -----------  ------------------------------------  -------  -----------
# Azure subscription 1    AzureCloud   xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  Enabled  True

# Show current subscription details
az account show

# Show subscription in JSON format
az account show --output json

# Show subscription in table format
az account show --output table
```

**Verify These Details**:
- ✅ Subscription State: "Enabled"
- ✅ User Name: "davidleepeejr@owlbangroup.com"
- ✅ IsDefault: true
- ✅ Subscription ID exists

---

### Step 5: Set Default Subscription (if multiple exist)

```powershell
# If you have multiple subscriptions, set the one you want to use
az account set --subscription "Azure subscription 1"

# Or use subscription ID
az account set --subscription "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

# Verify it's set correctly
az account show --output table
```

---

### Step 6: Configure Subscription Settings (10 minutes)

#### A. Set Spending Limit (Optional for Free Trial)

```powershell
# For Free Trial, spending limit is automatic
# For Pay-As-You-Go, set up cost alerts in Azure Portal

# Open Azure Portal
Start-Process "https://portal.azure.com"

# Navigate to: Cost Management + Billing > Budgets
# Create budget alert for $1000/month
```

#### B. Enable Required Resource Providers

```powershell
# Register required resource providers
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.DBforPostgreSQL
az provider register --namespace Microsoft.Cache
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.Insights

# Verify registration (may take 2-3 minutes)
az provider show --namespace Microsoft.ContainerService --query "registrationState"
az provider show --namespace Microsoft.ContainerRegistry --query "registrationState"
az provider show --namespace Microsoft.DBforPostgreSQL --query "registrationState"

# All should show: "Registered"
```

#### C. Set Default Location

```powershell
# Set default location for resources
az configure --defaults location=eastus

# Verify configuration
az configure --list-defaults
```

---

### Step 7: Create Service Principal for Automation (10 minutes)

```powershell
# Get your subscription ID
$subscriptionId = az account show --query id --output tsv

# Create service principal for CI/CD and automation
az ad sp create-for-rbac `
    --name "jpmorgan-financial-apis-sp" `
    --role contributor `
    --scopes /subscriptions/$subscriptionId

# IMPORTANT: Save the output securely!
# Output will look like:
# {
#   "appId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "displayName": "jpmorgan-financial-apis-sp",
#   "password": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "tenant": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
# }

# Store in environment variables (for current session)
$env:AZURE_CLIENT_ID = "appId from above"
$env:AZURE_CLIENT_SECRET = "password from above"
$env:AZURE_TENANT_ID = "tenant from above"
$env:AZURE_SUBSCRIPTION_ID = $subscriptionId

# Verify service principal
az ad sp show --id $env:AZURE_CLIENT_ID
```

**Save These Credentials Securely**:
- Store in password manager (1Password, LastPass, etc.)
- Document in secure company wiki
- Share with DevOps team via secure channel
- **NEVER commit to Git or share publicly**

---

### Step 8: Test Account Setup (5 minutes)

```powershell
# Test 1: Create a test resource group
az group create --name test-rg --location eastus

# Test 2: List resource groups
az group list --output table

# Test 3: Delete test resource group
az group delete --name test-rg --yes --no-wait

# Test 4: Verify Azure CLI configuration
az account show
az configure --list-defaults

# All tests should pass without errors
```

---

## ✅ VERIFICATION CHECKLIST

Before proceeding to deployment, verify:

- [ ] Logged out from bizleeper@gmail.com account
- [ ] Created Azure account with davidleepeejr@owlbangroup.com
- [ ] Azure subscription is active and enabled
- [ ] Logged in to Azure CLI with new account
- [ ] Subscription shows in `az account list`
- [ ] Default subscription is set correctly
- [ ] Required resource providers are registered
- [ ] Service principal created and credentials saved
- [ ] Test resource group creation successful
- [ ] Azure Portal accessible at portal.azure.com
- [ ] Billing information configured
- [ ] Cost alerts set up (optional but recommended)

---

## 🎯 NEXT STEPS: READY FOR DEPLOYMENT

Once all verification steps pass, you're ready to deploy!

### Immediate Next Actions:

1. **Review Deployment Plan**:
   ```powershell
   # Read the deployment guide
   Get-Content c:\Users\bizle\Desktop\jpmorgan_financial_apis\AZURE_DEPLOYMENT_GUIDE.md
   ```

2. **Run Pre-Deployment Verification**:
   ```powershell
   cd c:\Users\bizle\Desktop\jpmorgan_financial_apis\scripts
   .\verify_production_readiness.ps1
   ```

3. **Execute Azure Deployment**:
   ```powershell
   # This will create all Azure resources and deploy the application
   .\deploy_azure.ps1
   
   # Estimated time: 45-60 minutes
   # Estimated cost: $550-600/month
   ```

---

## 💰 COST MANAGEMENT

### Expected Monthly Costs

| Resource | Configuration | Monthly Cost |
|----------|--------------|--------------|
| AKS Cluster | 3 nodes (D2s_v3) | $200 |
| PostgreSQL | GeneralPurpose D2s_v3 | $150 |
| Redis Cache | Standard C1 | $75 |
| Container Registry | Standard | $5 |
| Storage Account | 100GB LRS | $20 |
| Key Vault | Standard | $0.03 |
| Monitoring | Log Analytics | $50 |
| Load Balancer | Standard | $20 |
| Bandwidth | Outbound | $10 |
| Backup | Automated | $15 |
| DNS | Azure DNS | $5 |
| **TOTAL** | | **~$550-600/month** |

### Cost Control Measures

1. **Set Up Budget Alerts**:
   ```powershell
   # In Azure Portal:
   # Cost Management + Billing > Budgets > Create
   # Set alert at $500, $750, and $1000
   ```

2. **Enable Cost Analysis**:
   ```powershell
   # Monitor daily costs in Azure Portal
   # Cost Management + Billing > Cost Analysis
   ```

3. **Use Azure Cost Management**:
   ```powershell
   # View cost breakdown
   az consumption usage list `
       --start-date (Get-Date).AddDays(-7).ToString('yyyy-MM-dd') `
       --end-date (Get-Date).ToString('yyyy-MM-dd')
   ```

4. **Implement Auto-Shutdown** (for dev/test environments):
   ```powershell
   # Stop AKS cluster during off-hours
   az aks stop --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-aks
   
   # Start when needed
   az aks start --resource-group jpmorgan-financial-apis-rg --name jpmorgan-financial-aks
   ```

---

## 🔒 SECURITY BEST PRACTICES

### Account Security

1. **Enable Multi-Factor Authentication (MFA)**:
   - Go to: https://account.microsoft.com/security
   - Enable MFA for davidleepeejr@owlbangroup.com
   - Use authenticator app (Microsoft Authenticator, Google Authenticator)

2. **Use Strong Password**:
   - Minimum 16 characters
   - Mix of uppercase, lowercase, numbers, symbols
   - Use password manager

3. **Regular Security Reviews**:
   - Review access logs monthly
   - Audit resource permissions quarterly
   - Update service principal credentials annually

### Subscription Security

1. **Role-Based Access Control (RBAC)**:
   ```powershell
   # Assign roles to team members
   az role assignment create `
       --assignee user@owlbangroup.com `
       --role "Contributor" `
       --scope /subscriptions/$subscriptionId/resourceGroups/jpmorgan-financial-apis-rg
   ```

2. **Enable Azure Security Center**:
   ```powershell
   # Enable in Azure Portal
   # Security Center > Getting Started > Upgrade
   ```

3. **Configure Network Security**:
   - Use Network Security Groups (NSGs)
   - Enable Azure Firewall
   - Implement DDoS protection

---

## 📞 SUPPORT & RESOURCES

### Azure Support

- **Portal**: https://portal.azure.com
- **Documentation**: https://docs.microsoft.com/azure
- **Support Phone**: 1-800-642-7676
- **Support Email**: Available in Azure Portal
- **Status Page**: https://status.azure.com

### The Owlban Group Internal

- **IT Support**: [Internal contact]
- **Finance Team**: [For billing questions]
- **DevOps Team**: [For technical support]
- **Security Team**: [For security concerns]

### Community Resources

- **Azure Forums**: https://docs.microsoft.com/answers/
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/azure
- **GitHub**: https://github.com/Azure
- **YouTube**: Azure Friday, Azure Tips and Tricks

---

## 🆘 TROUBLESHOOTING

### Issue: Cannot Create Subscription

**Symptoms**: "No subscriptions found" after account creation

**Solutions**:
1. Wait 5-10 minutes for subscription provisioning
2. Refresh browser and check Azure Portal
3. Verify email confirmation received
4. Contact Azure Support: 1-800-642-7676

### Issue: Login Fails

**Symptoms**: `az login` fails or shows wrong account

**Solutions**:
```powershell
# Clear all cached credentials
az logout
az account clear

# Delete cached tokens
Remove-Item -Path "$env:USERPROFILE\.azure" -Recurse -Force

# Login again
az login
```

### Issue: Resource Provider Not Registered

**Symptoms**: "The subscription is not registered to use namespace..."

**Solutions**:
```powershell
# Register the provider
az provider register --namespace Microsoft.ContainerService

# Wait for registration (2-3 minutes)
az provider show --namespace Microsoft.ContainerService --query "registrationState"

# Should show: "Registered"
```

### Issue: Insufficient Permissions

**Symptoms**: "Authorization failed" or "Access denied"

**Solutions**:
1. Verify you're logged in with correct account
2. Check subscription role assignments
3. Contact subscription administrator
4. Verify service principal has correct permissions

---

## 📊 ACCOUNT STATUS TRACKING

### Current Status

| Item | Status | Notes |
|------|--------|-------|
| Azure Account Created | ⏳ Pending | davidleepeejr@owlbangroup.com |
| Subscription Active | ⏳ Pending | Waiting for account creation |
| Azure CLI Installed | ✅ Complete | v2.80.0 |
| Logged In | ⏳ Pending | Will login after account creation |
| Resource Providers | ⏳ Pending | Will register after login |
| Service Principal | ⏳ Pending | Will create after subscription |
| Ready for Deployment | ⏳ Pending | All above must be complete |

### Timeline

- **Account Creation**: 15-20 minutes
- **Subscription Activation**: Immediate (Free Trial) or 5-10 minutes (Pay-As-You-Go)
- **Azure CLI Login**: 2 minutes
- **Resource Provider Registration**: 5 minutes
- **Service Principal Creation**: 5 minutes
- **Total Setup Time**: ~30-40 minutes

---

## 🎉 SUCCESS CRITERIA

You'll know the setup is complete when:

1. ✅ `az account show` displays davidleepeejr@owlbangroup.com
2. ✅ `az account list` shows active subscription
3. ✅ All resource providers show "Registered" status
4. ✅ Service principal credentials saved securely
5. ✅ Test resource group creation succeeds
6. ✅ Azure Portal accessible and shows subscription
7. ✅ Cost alerts configured
8. ✅ MFA enabled on account

**When all criteria are met, proceed to deployment!**

---

## 📝 IMPORTANT NOTES

### For The Owlban Group Management

1. **Budget Approval Required**:
   - Monthly cost: $550-600
   - Annual cost: $6,600-7,200
   - One-time setup: $0 (using automated scripts)

2. **Business Continuity**:
   - Backup strategy included in deployment
   - 99.95% SLA for AKS
   - Multi-region failover available (additional cost)

3. **Compliance**:
   - Azure complies with SOC 2, ISO 27001, HIPAA, PCI DSS
   - Data residency: US East (configurable)
   - Audit logs retained for 90 days (configurable)

### For DevOps Team

1. **Access Management**:
   - Use service principal for CI/CD
   - Individual accounts for team members
   - Implement least privilege access

2. **Monitoring**:
   - Application Insights included
   - Log Analytics workspace created
   - Grafana dashboards available

3. **Deployment**:
   - Automated via PowerShell script
   - Infrastructure as Code (IaC) ready
   - Blue-green deployment supported

---

## 📋 QUICK REFERENCE

### Essential Commands

```powershell
# Login
az login

# Show account
az account show

# List subscriptions
az account list --output table

# Set subscription
az account set --subscription "subscription-name-or-id"

# List resource groups
az group list --output table

# Create resource group
az group create --name my-rg --location eastus

# Delete resource group
az group delete --name my-rg --yes

# Show costs
az consumption usage list --start-date 2024-01-01 --end-date 2024-01-31
```

### Important URLs

- **Azure Portal**: https://portal.azure.com
- **Free Trial**: https://azure.microsoft.com/free/
- **Pay-As-You-Go**: https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go/
- **Pricing Calculator**: https://azure.microsoft.com/pricing/calculator/
- **Status Page**: https://status.azure.com
- **Documentation**: https://docs.microsoft.com/azure

---

## 🚀 READY TO BEGIN?

Follow these steps in order:

1. ✅ Read this entire document
2. ✅ Get budget approval from management
3. ✅ Logout from current Azure account
4. ✅ Create new Azure account (davidleepeejr@owlbangroup.com)
5. ✅ Complete subscription setup
6. ✅ Login with Azure CLI
7. ✅ Verify subscription
8. ✅ Register resource providers
9. ✅ Create service principal
10. ✅ Run verification tests
11. ✅ Proceed to deployment

---

**Document Version**: 1.0.0  
**Created**: 2024-11-19  
**Account**: davidleepeejr@owlbangroup.com  
**Status**: Setup Guide Ready  
**Next Action**: Create Azure Account  

---

**LET'S GET STARTED! 🚀**

Create your Azure account now: https://azure.microsoft.com/free/

---

**END OF AZURE ACCOUNT SETUP GUIDE**
