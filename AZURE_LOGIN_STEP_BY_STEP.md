# 🔐 Azure CLI Login - Step-by-Step Guide

## Current Status
You have initiated `az login --tenant dc3405c4-651b-4650-8231-78739bd4f8c6` and need to complete the authentication.

---

## 📋 STEP-BY-STEP AUTHENTICATION PROCESS

### Step 1: Check Your Browser 🌐

**A browser window should have opened automatically.** Look for:

1. **Check all open browser windows/tabs** - The Azure login page might be hidden behind other windows
2. **Look for a tab titled**: "Sign in to your account" or "Microsoft Azure"
3. **The URL should start with**: **`https://login.microsoftonline.com/`**

**If you DON'T see a browser window:**
- Check your taskbar for a new browser window
- Check if your browser blocked the popup
- Look at the terminal for a URL you can manually copy and paste into your browser

---

### Step 2: Complete the Sign-In Process 🔑

Once you find the browser window, follow these steps:

#### A. Select Your Account
```
You'll see a page saying: "Pick an account"
```
- **Click on your Microsoft account** associated with tenant `dc3405c4-651b-4650-8231-78739bd4f8c6`
- This might be: davidleepeejr@owlbangroup.com or another account

#### B. Enter Your Password
```
Enter your Microsoft account password
```
- Type your password
- Click "Sign in"

#### C. Multi-Factor Authentication (if enabled)
```
You may be asked to verify your identity
```
- **Authenticator App**: Approve the notification on your phone
- **SMS Code**: Enter the code sent to your phone
- **Email Code**: Enter the code sent to your email
- **Security Key**: Use your physical security key

#### D. Stay Signed In? (Optional)
```
"Stay signed in?"
```
- Click "Yes" if you want to stay logged in
- Click "No" if this is a shared computer

#### E. Grant Permissions
```
"Azure CLI wants to access your account"
```
- Review the permissions requested
- Click "Accept" or "Yes" to grant access

---

### Step 3: Confirmation ✅

After completing authentication, you should see:

```
✅ "You have signed in to the Microsoft Azure Cross-platform Command Line Interface application on your device."

OR

✅ "Authentication complete. You can close this window."
```

**At this point:**
1. ✅ Close the browser window
2. ✅ Return to your terminal/VSCode
3. ✅ The terminal should now show your Azure subscription details in JSON format

---

## 🔧 TROUBLESHOOTING

### Problem 1: Browser Didn't Open

**Solution A: Find the URL in Terminal**
1. Look at your terminal output
2. Find a line that says: "To sign in, use a web browser to open the page..."
3. Copy the URL shown
4. Paste it into your browser manually

**Solution B: Use Device Code Authentication**
1. Press `Ctrl+C` in the terminal to cancel current login
2. Run this instead:
```powershell
az login --use-device-code --tenant dc3405c4-651b-4650-8231-78739bd4f8c6
```
3. You'll get a code like: `ABC123DEF`
4. Go to: https://microsoft.com/devicelogin
5. Enter the code
6. Complete authentication

---

### Problem 2: Wrong Account Showing

**Solution:**
1. In the browser, click "Use another account"
2. Enter the correct email address
3. Complete authentication with that account

---

### Problem 3: "No Subscriptions Found"

**After successful login, if you see "No subscriptions found":**

This means:
- ✅ Authentication succeeded
- ❌ But your account has no Azure subscriptions

**To fix:**
1. Go to: https://portal.azure.com
2. Click "Subscriptions" in the left menu
3. Click "+ Add" to create a subscription
4. Choose:
   - **Free Trial** (recommended for testing) - $200 credit for 30 days
   - **Pay-As-You-Go** (for production) - ~$600/month

---

### Problem 4: Authentication Timeout

**If the browser shows "Request timed out":**

1. Press `Ctrl+C` in terminal to cancel
2. Run the login command again:
```powershell
az login --tenant dc3405c4-651b-4650-8231-78739bd4f8c6
```
3. Complete authentication faster this time

---

### Problem 5: "Tenant Not Found"

**If you see "Tenant 'dc3405c4-651b-4650-8231-78739bd4f8c6' not found":**

**Solution:**
1. Try logging in without specifying tenant:
```powershell
az login
```
2. After login, check available tenants:
```powershell
az account list --output table
```

---

## 📱 WHAT YOU SHOULD SEE

### In Browser (During Authentication):
```
┌─────────────────────────────────────┐
│  Microsoft                          │
│  Sign in                            │
│                                     │
│  Email: [your-email@domain.com]     │
│  Password: [**********]             │
│                                     │
│  [Sign in]                          │
└─────────────────────────────────────┘
```

### In Browser (After Success):
```
┌─────────────────────────────────────┐
│  ✅ Authentication Complete          │
│                                     │
│  You have signed in to the          │
│  Microsoft Azure CLI                │
│                                     │
│  You can close this window          │
└─────────────────────────────────────┘
```

### In Terminal (After Success):
```json
[
  {
    "cloudName": "AzureCloud",
    "homeTenantId": "dc3405c4-651b-4650-8231-78739bd4f8c6",
    "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "isDefault": true,
    "name": "Azure subscription 1",
    "state": "Enabled",
    "tenantId": "dc3405c4-651b-4650-8231-78739bd4f8c6",
    "user": {
      "name": "your-email@domain.com",
      "type": "user"
    }
  }
]
```

---

## ✅ VERIFICATION STEPS

After you see the success message, verify the login:

### 1. Check Account Status
```powershell
az account show
```

**Expected Output:**
```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "dc3405c4-651b-4650-8231-78739bd4f8c6",
  "id": "subscription-id",
  "isDefault": true,
  "name": "Azure subscription 1",
  "state": "Enabled",
  "tenantId": "dc3405c4-651b-4650-8231-78739bd4f8c6",
  "user": {
    "name": "your-email@domain.com",
    "type": "user"
  }
}
```

### 2. List All Subscriptions
```powershell
az account list --output table
```

**Expected Output:**
```
Name                    CloudName    SubscriptionId                        State    IsDefault
----------------------  -----------  ------------------------------------  -------  -----------
Azure subscription 1    AzureCloud   xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  Enabled  True
```

### 3. Test Azure Access
```powershell
az group list --output table
```

**Expected Output:**
```
Name                          Location    Status
----------------------------  ----------  ---------
(empty if no resource groups created yet)
```

---

## 🎯 NEXT STEPS AFTER SUCCESSFUL LOGIN

Once you see your subscription details in the terminal:

### 1. Verify Subscription
```powershell
az account show --output table
```

### 2. Set Default Subscription (if multiple)
```powershell
az account set --subscription "subscription-name-or-id"
```

### 3. Register Resource Providers
```powershell
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.DBforPostgreSQL
az provider register --namespace Microsoft.Cache
az provider register --namespace Microsoft.KeyVault
```

### 4. Create Resource Group
```powershell
az group create --name jpmorgan-financial-apis-rg --location eastus
```

### 5. Proceed with Deployment
```powershell
cd c:\Users\bizle\Desktop\jpmorgan_financial_apis\scripts
.\deploy_azure.ps1
```

---

## 📞 NEED HELP?

If you're still stuck:

1. **Check the terminal** - Look for any error messages or URLs
2. **Check browser** - Make sure no popup blockers are active
3. **Try device code** - Use `az login --use-device-code` as alternative
4. **Check Azure Portal** - Go to https://portal.azure.com to verify account access

---

## 🔐 SECURITY NOTES

- ✅ Always verify the URL is `login.microsoftonline.com`
- ✅ Never enter credentials on suspicious pages
- ✅ Use MFA/2FA for additional security
- ✅ Don't share your authentication tokens
- ✅ Log out when done: `az logout`

---

**Document Created**: For Azure CLI Authentication
**Tenant ID**: dc3405c4-651b-4650-8231-78739bd4f8c6
**Status**: Waiting for browser authentication completion

---

## 📋 QUICK CHECKLIST

- [ ] Browser window opened (or manually opened URL)
- [ ] Selected correct Microsoft account
- [ ] Entered password
- [ ] Completed MFA verification (if required)
- [ ] Accepted permissions
- [ ] Saw "Authentication complete" message
- [ ] Closed browser window
- [ ] Returned to terminal
- [ ] Saw JSON output with subscription details
- [ ] Verified with `az account show`

**Once all items are checked, your Azure CLI is authenticated and ready!** ✅

---

**END OF GUIDE**
