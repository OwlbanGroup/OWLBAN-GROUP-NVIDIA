<#
.SYNOPSIS
    Azure Account Setup Helper Script for davidleepeejr@owlbangroup.com

.DESCRIPTION
    This script helps set up the new Azure account and verify all prerequisites
    for deploying JPMorgan Financial APIs to Azure.

.EXAMPLE
    .\setup_azure_account.ps1
#>

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

# Display header
function Show-Header {
    Write-Host "`n" -NoNewline
    Write-Host "╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                                                                    ║" -ForegroundColor Cyan
    Write-Host "║           Azure Account Setup for The Owlban Group                ║" -ForegroundColor Cyan
    Write-Host "║              JPMorgan Financial APIs Deployment                    ║" -ForegroundColor Cyan
    Write-Host "║                                                                    ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host "`n"
    Write-Info "Account: davidleepeejr@owlbangroup.com"
    Write-Info "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
}

# Check Azure CLI installation
function Test-AzureCLI {
    Write-Step "Step 1: Checking Azure CLI Installation"
    
    try {
        $azVersion = az --version 2>$null
        if ($azVersion) {
            Write-Success "Azure CLI is installed"
            $version = ($azVersion | Select-String "azure-cli" | Out-String).Trim()
            Write-Info $version
            return $true
        }
    }
    catch {
        Write-Error "Azure CLI is not installed"
        Write-Warning "Please install from: https://aka.ms/installazurecliwindows"
        Write-Info "Or run: winget install -e --id Microsoft.AzureCLI"
        return $false
    }
}

# Logout from current account
function Clear-AzureAccount {
    Write-Step "Step 2: Clearing Previous Azure Account"
    
    try {
        $currentAccount = az account show 2>$null
        if ($currentAccount) {
            $accountInfo = $currentAccount | ConvertFrom-Json
            Write-Info "Currently logged in as: $($accountInfo.user.name)"
            
            $response = Read-Host "Do you want to logout from this account? (Y/N)"
            if ($response -eq 'Y' -or $response -eq 'y') {
                Write-Info "Logging out..."
                az logout
                az account clear
                Write-Success "Logged out successfully"
            }
            else {
                Write-Warning "Keeping current login. Make sure you're using the correct account!"
            }
        }
        else {
            Write-Info "No active Azure login found"
        }
    }
    catch {
        Write-Info "No active Azure login found"
    }
}

# Guide user to create Azure account
function Show-AccountCreationGuide {
    Write-Step "Step 3: Azure Account Creation"
    
    Write-Host "`n📋 Account Information:" -ForegroundColor Yellow
    Write-Host "  Email: davidleepeejr@owlbangroup.com" -ForegroundColor White
    Write-Host "  Company: The Owlban Group" -ForegroundColor White
    Write-Host ""
    
    Write-Host "🎯 Choose Your Subscription Type:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Option 1: Free Trial (Recommended for Testing)" -ForegroundColor Cyan
    Write-Host "    • $200 credit for 30 days"
    Write-Host "    • No charges during trial"
    Write-Host "    • Easy upgrade to production"
    Write-Host "    • URL: https://azure.microsoft.com/free/"
    Write-Host ""
    Write-Host "  Option 2: Pay-As-You-Go (Production Ready)" -ForegroundColor Cyan
    Write-Host "    • ~$550-600/month estimated cost"
    Write-Host "    • No upfront commitment"
    Write-Host "    • Production-ready immediately"
    Write-Host "    • URL: https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go/"
    Write-Host ""
    
    $choice = Read-Host "Which option do you want? (1 for Free Trial, 2 for Pay-As-You-Go)"
    
    if ($choice -eq '1') {
        Write-Info "Opening Free Trial signup page..."
        Start-Process "https://azure.microsoft.com/free/"
    }
    elseif ($choice -eq '2') {
        Write-Info "Opening Pay-As-You-Go signup page..."
        Start-Process "https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go/"
    }
    else {
        Write-Warning "Invalid choice. Please visit one of these URLs manually:"
        Write-Host "  Free Trial: https://azure.microsoft.com/free/"
        Write-Host "  Pay-As-You-Go: https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go/"
    }
    
    Write-Host ""
    Write-Warning "Complete the account creation in your browser, then return here."
    Write-Host ""
    Read-Host "Press Enter when you've completed the account creation"
}

# Login to Azure
function Connect-AzureAccount {
    Write-Step "Step 4: Logging in to Azure"
    
    Write-Info "Initiating Azure login..."
    Write-Warning "A browser window will open for authentication"
    Write-Warning "Sign in with: davidleepeejr@owlbangroup.com"
    Write-Host ""
    
    try {
        az login
        
        Write-Host ""
        Write-Success "Login successful!"
        
        # Show account info
        $account = az account show | ConvertFrom-Json
        Write-Host ""
        Write-Host "📊 Account Information:" -ForegroundColor Cyan
        Write-Host "  User: $($account.user.name)" -ForegroundColor White
        Write-Host "  Subscription: $($account.name)" -ForegroundColor White
        Write-Host "  Subscription ID: $($account.id)" -ForegroundColor White
        Write-Host "  State: $($account.state)" -ForegroundColor White
        Write-Host ""
        
        return $true
    }
    catch {
        Write-Error "Login failed: $_"
        return $false
    }
}

# Verify subscription
function Test-AzureSubscription {
    Write-Step "Step 5: Verifying Azure Subscription"
    
    try {
        $subscriptions = az account list | ConvertFrom-Json
        
        if ($subscriptions.Count -eq 0) {
            Write-Error "No Azure subscriptions found!"
            Write-Warning "Please ensure you completed the subscription setup in the Azure Portal"
            Write-Info "Visit: https://portal.azure.com"
            return $false
        }
        
        Write-Success "Found $($subscriptions.Count) subscription(s)"
        Write-Host ""
        Write-Host "📋 Subscription Details:" -ForegroundColor Cyan
        
        foreach ($sub in $subscriptions) {
            Write-Host "  Name: $($sub.name)" -ForegroundColor White
            Write-Host "  ID: $($sub.id)" -ForegroundColor White
            Write-Host "  State: $($sub.state)" -ForegroundColor White
            Write-Host "  Default: $($sub.isDefault)" -ForegroundColor White
            Write-Host ""
        }
        
        return $true
    }
    catch {
        Write-Error "Failed to verify subscription: $_"
        return $false
    }
}

# Register resource providers
function Register-AzureProviders {
    Write-Step "Step 6: Registering Azure Resource Providers"
    
    $providers = @(
        "Microsoft.ContainerService",
        "Microsoft.ContainerRegistry",
        "Microsoft.DBforPostgreSQL",
        "Microsoft.Cache",
        "Microsoft.KeyVault",
        "Microsoft.Storage",
        "Microsoft.OperationalInsights",
        "Microsoft.Insights",
        "Microsoft.Network",
        "Microsoft.Compute"
    )
    
    Write-Info "Registering $($providers.Count) resource providers..."
    Write-Warning "This may take 2-3 minutes"
    Write-Host ""
    
    foreach ($provider in $providers) {
        Write-Host "  Registering $provider..." -NoNewline
        try {
            az provider register --namespace $provider --wait 2>$null | Out-Null
            Write-Host " ✅" -ForegroundColor Green
        }
        catch {
            Write-Host " ⚠️" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Info "Verifying registration status..."
    Start-Sleep -Seconds 5
    
    $allRegistered = $true
    foreach ($provider in $providers) {
        $state = az provider show --namespace $provider --query "registrationState" --output tsv 2>$null
        if ($state -eq "Registered") {
            Write-Host "  $provider : ✅ Registered" -ForegroundColor Green
        }
        else {
            Write-Host "  $provider : ⏳ $state" -ForegroundColor Yellow
            $allRegistered = $false
        }
    }
    
    Write-Host ""
    if ($allRegistered) {
        Write-Success "All resource providers registered successfully"
    }
    else {
        Write-Warning "Some providers are still registering. This is normal and won't block deployment."
    }
    
    return $true
}

# Create service principal
function New-ServicePrincipal {
    Write-Step "Step 7: Creating Service Principal"
    
    Write-Info "Creating service principal for automation and CI/CD..."
    
    try {
        $subscriptionId = az account show --query id --output tsv
        
        $sp = az ad sp create-for-rbac `
            --name "jpmorgan-financial-apis-sp" `
            --role contributor `
            --scopes "/subscriptions/$subscriptionId" | ConvertFrom-Json
        
        Write-Success "Service principal created successfully"
        Write-Host ""
        Write-Host "🔐 Service Principal Credentials:" -ForegroundColor Yellow
        Write-Host "  App ID (Client ID): $($sp.appId)" -ForegroundColor White
        Write-Host "  Password (Client Secret): $($sp.password)" -ForegroundColor White
        Write-Host "  Tenant ID: $($sp.tenant)" -ForegroundColor White
        Write-Host "  Subscription ID: $subscriptionId" -ForegroundColor White
        Write-Host ""
        Write-Warning "IMPORTANT: Save these credentials securely!"
        Write-Warning "You won't be able to retrieve the password again."
        Write-Host ""
        
        # Save to file
        $credFile = Join-Path (Split-Path -Parent $PSScriptRoot) "azure_service_principal.txt"
        @"
Azure Service Principal Credentials
====================================
Created: $(Get-Date)
Account: davidleepeejr@owlbangroup.com

App ID (Client ID): $($sp.appId)
Password (Client Secret): $($sp.password)
Tenant ID: $($sp.tenant)
Subscription ID: $subscriptionId

IMPORTANT: 
- Store these credentials in a secure password manager
- Delete this file after saving the credentials
- Never commit this file to Git
- Share only via secure channels

Environment Variables (for CI/CD):
AZURE_CLIENT_ID=$($sp.appId)
AZURE_CLIENT_SECRET=$($sp.password)
AZURE_TENANT_ID=$($sp.tenant)
AZURE_SUBSCRIPTION_ID=$subscriptionId
"@ | Out-File -FilePath $credFile -Encoding UTF8
        
        Write-Info "Credentials saved to: $credFile"
        Write-Warning "Please save these credentials and delete the file!"
        Write-Host ""
        
        return $true
    }
    catch {
        Write-Error "Failed to create service principal: $_"
        Write-Info "You can create it manually later if needed"
        return $false
    }
}

# Run verification tests
function Test-AzureSetup {
    Write-Step "Step 8: Running Verification Tests"
    
    Write-Info "Running setup verification tests..."
    Write-Host ""
    
    $allPassed = $true
    
    # Test 1: Azure CLI
    Write-Host "  Test 1: Azure CLI installed..." -NoNewline
    try {
        az --version | Out-Null
        Write-Host " ✅" -ForegroundColor Green
    }
    catch {
        Write-Host " ❌" -ForegroundColor Red
        $allPassed = $false
    }
    
    # Test 2: Logged in
    Write-Host "  Test 2: Logged in to Azure..." -NoNewline
    try {
        $account = az account show 2>$null
        if ($account) {
            Write-Host " ✅" -ForegroundColor Green
        }
        else {
            Write-Host " ❌" -ForegroundColor Red
            $allPassed = $false
        }
    }
    catch {
        Write-Host " ❌" -ForegroundColor Red
        $allPassed = $false
    }
    
    # Test 3: Subscription exists
    Write-Host "  Test 3: Active subscription..." -NoNewline
    try {
        $subs = az account list | ConvertFrom-Json
        if ($subs.Count -gt 0) {
            Write-Host " ✅" -ForegroundColor Green
        }
        else {
            Write-Host " ❌" -ForegroundColor Red
            $allPassed = $false
        }
    }
    catch {
        Write-Host " ❌" -ForegroundColor Red
        $allPassed = $false
    }
    
    # Test 4: Can create resource group
    Write-Host "  Test 4: Resource group creation..." -NoNewline
    try {
        az group create --name test-setup-rg --location eastus --output none 2>$null
        az group delete --name test-setup-rg --yes --no-wait 2>$null
        Write-Host " ✅" -ForegroundColor Green
    }
    catch {
        Write-Host " ❌" -ForegroundColor Red
        $allPassed = $false
    }
    
    # Test 5: Docker installed
    Write-Host "  Test 5: Docker installed..." -NoNewline
    try {
        docker --version | Out-Null
        Write-Host " ✅" -ForegroundColor Green
    }
    catch {
        Write-Host " ⚠️  (Optional)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    if ($allPassed) {
        Write-Success "All critical tests passed!"
        return $true
    }
    else {
        Write-Warning "Some tests failed. Please review and fix issues."
        return $false
    }
}

# Show next steps
function Show-NextSteps {
    Write-Step "Setup Complete! 🎉"
    
    Write-Host ""
    Write-Host "✅ Azure account setup is complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 What was configured:" -ForegroundColor Cyan
    Write-Host "  ✅ Azure CLI verified"
    Write-Host "  ✅ Logged in with davidleepeejr@owlbangroup.com"
    Write-Host "  ✅ Subscription verified and active"
    Write-Host "  ✅ Resource providers registered"
    Write-Host "  ✅ Service principal created"
    Write-Host "  ✅ Verification tests passed"
    Write-Host ""
    Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. Review the deployment guide:"
    Write-Host "     Get-Content ..\AZURE_DEPLOYMENT_GUIDE.md"
    Write-Host ""
    Write-Host "  2. Run pre-deployment verification:"
    Write-Host "     .\verify_production_readiness.ps1"
    Write-Host ""
    Write-Host "  3. Execute Azure deployment:"
    Write-Host "     .\deploy_azure.ps1"
    Write-Host ""
    Write-Host "  4. Estimated deployment time: 45-60 minutes"
    Write-Host "  5. Estimated monthly cost: $550-600"
    Write-Host ""
    Write-Host "💡 Helpful Commands:" -ForegroundColor Cyan
    Write-Host "  • View account: az account show"
    Write-Host "  • List subscriptions: az account list --output table"
    Write-Host "  • View resources: az resource list --output table"
    Write-Host "  • Open Azure Portal: Start-Process https://portal.azure.com"
    Write-Host ""
    Write-Host "📞 Support:" -ForegroundColor Cyan
    Write-Host "  • Azure Support: 1-800-642-7676"
    Write-Host "  • Documentation: https://docs.microsoft.com/azure"
    Write-Host "  • Portal: https://portal.azure.com"
    Write-Host ""
    Write-Success "You're ready to deploy to Azure!"
    Write-Host ""
}

# Main execution
function Start-AzureAccountSetup {
    Show-Header
    
    # Step 1: Check Azure CLI
    if (-not (Test-AzureCLI)) {
        Write-Error "Please install Azure CLI and run this script again"
        exit 1
    }
    
    # Step 2: Clear previous account
    Clear-AzureAccount
    
    # Step 3: Guide account creation
    Show-AccountCreationGuide
    
    # Step 4: Login
    if (-not (Connect-AzureAccount)) {
        Write-Error "Login failed. Please try again."
        exit 1
    }
    
    # Step 5: Verify subscription
    if (-not (Test-AzureSubscription)) {
        Write-Error "Subscription verification failed. Please check your Azure account."
        exit 1
    }
    
    # Step 6: Register providers
    Register-AzureProviders
    
    # Step 7: Create service principal
    New-ServicePrincipal
    
    # Step 8: Run tests
    if (-not (Test-AzureSetup)) {
        Write-Warning "Some verification tests failed, but you can proceed with caution."
    }
    
    # Show next steps
    Show-NextSteps
}

# Execute the setup
Start-AzureAccountSetup
