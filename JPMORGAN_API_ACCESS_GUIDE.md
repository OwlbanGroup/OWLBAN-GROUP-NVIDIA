# JP Morgan API Access Guide

## How to Get JP Morgan Developer API Access

This guide explains how to obtain API credentials for JP Morgan's developer platform.

---

## ⚠️ IMPORTANT DISCLAIMER

**This is a DEVELOPER/SANDBOX environment, NOT production banking access.**

JP Morgan provides developer APIs for:
- Testing and development
- Building integrations
- Proof of concepts
- Sandbox environments

**This does NOT provide access to:**
- Real bank accounts
- Live financial transactions
- Personal banking credentials
- Production banking systems

---

## Steps to Get API Access

### 1. Register for JP Morgan Developer Portal

**Visit:** https://developer.payments.jpmorgan.com/

**Steps:**
1. Click "Sign Up" or "Register"
2. Fill out the registration form:
   - Business email address
   - Company information
   - Use case description
3. Verify your email address
4. Complete your profile

### 2. Create a Developer Account

After registration:
1. Log in to the developer portal
2. Navigate to "My Applications" or "Projects"
3. Click "Create New Application"
4. Fill in application details:
   - Application name
   - Description
   - Intended use case

### 3. Generate API Credentials

For each project/API you want to use:

1. **Select the API Product:**
   - AI Accounts API
   - Corporate Login API
   - Payroll API
   - Petty Cash API
   - Data Integration API

2. **Generate Credentials:**
   - Client ID
   - Client Secret
   - API Key

3. **Save Credentials Securely:**
   - Store in password manager
   - Never commit to version control
   - Use environment variables

### 4. Configure Your Application

Add credentials to your `.env` file:

```bash
# JP Morgan API Configuration
JPMORGAN_BASE_URL=https://api-sandbox.payments.jpmorgan.com
JPMORGAN_AUTH_URL=https://auth-sandbox.payments.jpmorgan.com

# AI Accounts Project
JPMORGAN_AI_ACCOUNTS_CLIENT_ID=your_client_id_here
JPMORGAN_AI_ACCOUNTS_CLIENT_SECRET=your_client_secret_here
JPMORGAN_AI_ACCOUNTS_API_KEY=your_api_key_here

# Corporate Login Project
JPMORGAN_CORPORATE_CLIENT_ID=your_client_id_here
JPMORGAN_CORPORATE_CLIENT_SECRET=your_client_secret_here
JPMORGAN_CORPORATE_API_KEY=your_api_key_here

# Payroll Project
JPMORGAN_PAYROLL_CLIENT_ID=your_client_id_here
JPMORGAN_PAYROLL_CLIENT_SECRET=your_client_secret_here
JPMORGAN_PAYROLL_API_KEY=your_api_key_here

# Petty Cash Project
JPMORGAN_PETTY_CASH_CLIENT_ID=your_client_id_here
JPMORGAN_PETTY_CASH_CLIENT_SECRET=your_client_secret_here
JPMORGAN_PETTY_CASH_API_KEY=your_api_key_here

# Owl1 Integration Project
JPMORGAN_OWL1_CLIENT_ID=your_client_id_here
JPMORGAN_OWL1_CLIENT_SECRET=your_client_secret_here
JPMORGAN_OWL1_API_KEY=your_api_key_here
```

---

## API Environments

### Sandbox (Development)
- **Purpose:** Testing and development
- **URL:** https://api-sandbox.payments.jpmorgan.com
- **Data:** Mock/test data only
- **Transactions:** Simulated, not real

### Production (Live)
- **Purpose:** Real business operations
- **URL:** https://api.payments.jpmorgan.com
- **Requirements:**
  - Business verification
  - Compliance review
  - Legal agreements
  - Security audit
- **Access:** Requires formal business relationship with JP Morgan

---

## Getting Production Access

**Production access requires:**

1. **Business Relationship:**
   - Existing JP Morgan business account
   - Corporate banking relationship
   - Verified business entity

2. **Compliance Requirements:**
   - KYC (Know Your Customer) verification
   - AML (Anti-Money Laundering) compliance
   - Security certifications
   - Legal agreements

3. **Application Process:**
   - Contact JP Morgan business representative
   - Submit formal application
   - Undergo security review
   - Sign legal agreements
   - Complete onboarding process

4. **Contact Information:**
   - **Business Banking:** Contact your JP Morgan relationship manager
   - **Developer Support:** developer-support@jpmorgan.com
   - **Sales Inquiries:** https://www.jpmorgan.com/commercial-banking/contact-us

---

## Security Best Practices

### Credential Management

1. **Never Hardcode Credentials:**
   ```python
   # ❌ BAD
   client_id = "abc123"
   
   # ✅ GOOD
   client_id = os.getenv("JPMORGAN_CLIENT_ID")
   ```

2. **Use Environment Variables:**
   - Store in `.env` file
   - Add `.env` to `.gitignore`
   - Use different credentials per environment

3. **Rotate Credentials Regularly:**
   - Change API keys every 90 days
   - Rotate after team member changes
   - Update immediately if compromised

4. **Limit Access:**
   - Use least privilege principle
   - Separate dev/prod credentials
   - Monitor API usage

### API Security

1. **Use HTTPS Only**
2. **Implement Rate Limiting**
3. **Log API Calls**
4. **Monitor for Anomalies**
5. **Implement Timeout Handling**

---

## Testing Your Integration

### 1. Test Authentication

```python
from src.jpmorgan_client import JPMorganAPIClient

client = JPMorganAPIClient()
token = await client.get_access_token("ai_accounts")
print(f"Token obtained: {token[:20]}...")
```

### 2. Test API Endpoints

```python
# Get accounts
accounts = await client.get_accounts()
print(f"Found {len(accounts)} accounts")

# Get balance
balance = await client.get_account_balance("account_id")
print(f"Balance: {balance}")
```

### 3. Run Health Check

```bash
curl http://localhost:8000/api/jpmorgan/health
```

---

## Troubleshooting

### Common Issues

1. **"Invalid credentials"**
   - Verify credentials in `.env` file
   - Check for typos
   - Ensure credentials are for correct environment

2. **"API key not found"**
   - Generate API key in developer portal
   - Add to `.env` file
   - Restart application

3. **"Rate limit exceeded"**
   - Implement exponential backoff
   - Reduce request frequency
   - Contact support for limit increase

4. **"Unauthorized"**
   - Check token expiration
   - Verify OAuth flow
   - Ensure correct scopes

---

## Support Resources

### Documentation
- **Developer Portal:** https://developer.payments.jpmorgan.com/docs
- **API Reference:** https://developer.payments.jpmorgan.com/api-reference
- **Integration Guides:** https://developer.payments.jpmorgan.com/guides

### Support Channels
- **Developer Forum:** https://community.jpmorgan.com/developers
- **Email Support:** developer-support@jpmorgan.com
- **Status Page:** https://status.jpmorgan.com

### Additional Resources
- **GitHub Examples:** https://github.com/jpmorgan/api-examples
- **Postman Collection:** Available in developer portal
- **SDK Documentation:** Language-specific SDKs available

---

## Summary

To get started with JP Morgan APIs:

1. ✅ Register at developer.payments.jpmorgan.com
2. ✅ Create a developer account
3. ✅ Generate API credentials for each project
4. ✅ Add credentials to `.env` file
5. ✅ Test in sandbox environment
6. ✅ For production: Contact JP Morgan business banking

**Remember:** This is for DEVELOPMENT purposes. Production access requires a formal business relationship with JP Morgan.
