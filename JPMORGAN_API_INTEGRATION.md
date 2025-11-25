# JP Morgan Payments API Integration Guide

## 🏦 Your JP Morgan Developer Portal Access

**Portal URL**: https://developer.payments.jpmorgan.com/console/organizations/D3R56WRGSR3R

**Your Projects**:
1. AI ACCOUNTS - Corporate, Business, Personal accounts
2. CORPORATE EXECUTIVE LOGIN - Corporate logins
3. OWL PAYROLL - Payroll system
4. OWL PETTY CASH - Petty cash access
5. Owl1 - Data integration
6. **OpenBanking API** - Multi-environment banking services
7. **API Gateway** - Service orchestration and routing

---

## 🌐 New Endpoints (Multi-Environment Support)

### Production Endpoints
- **OpenBanking:** `https://openbanking.jpmorgan.com/accessapi`
- **API Gateway:** `https://apigateway.jpmorgan.com/accessapi`

### UAT (User Acceptance Testing)
- **OpenBanking UAT:** `https://openbankinguat.jpmorgan.com/accessapi`

### QAF (Quality Assurance & Functional Testing)
- **API Gateway QAF:** `https://apigatewayqaf.jpmorgan.com/accessapi`

For detailed information on these endpoints, see [JPMORGAN_ENDPOINTS_INTEGRATION.md](./JPMORGAN_ENDPOINTS_INTEGRATION.md)

---

## 🔗 Integration Steps

### Step 1: Get API Credentials from JP Morgan Portal

1. Login to: https://developer.payments.jpmorgan.com/console/organizations/D3R56WRGSR3R
2. Navigate to each project:
   - AI ACCOUNTS
   - CORPORATE EXECUTIVE LOGIN
   - OWL PAYROLL
   - OWL PETTY CASH
   - Owl1
3. Get API credentials for each:
   - Client ID
   - Client Secret
   - API Key
   - OAuth endpoints

### Step 2: Configure Environment Variables

Choose your environment and use the appropriate template:

**For Production:**
```bash
cp .env.production.template .env.production
# Edit .env.production with your credentials
```

**For UAT:**
```bash
cp .env.uat.template .env.uat
# Edit .env.uat with your credentials
```

**For QAF:**
```bash
cp .env.qaf.template .env.qaf
# Edit .env.qaf with your credentials
```

**Environment Configuration includes:**
```env
# Environment Selection
JPMORGAN_ENVIRONMENT=production  # or 'uat' or 'qaf'

# OpenBanking API
JPMORGAN_OPENBANKING_PRODUCTION_URL=https://openbanking.jpmorgan.com/accessapi
JPMORGAN_OPENBANKING_UAT_URL=https://openbankinguat.jpmorgan.com/accessapi
JPMORGAN_OPENBANKING_CLIENT_ID=your_client_id
JPMORGAN_OPENBANKING_CLIENT_SECRET=your_client_secret
JPMORGAN_OPENBANKING_API_KEY=your_api_key

# API Gateway
JPMORGAN_APIGATEWAY_PRODUCTION_URL=https://apigateway.jpmorgan.com/accessapi
JPMORGAN_APIGATEWAY_QAF_URL=https://apigatewayqaf.jpmorgan.com/accessapi
JPMORGAN_APIGATEWAY_CLIENT_ID=your_client_id
JPMORGAN_APIGATEWAY_CLIENT_SECRET=your_client_secret
JPMORGAN_APIGATEWAY_API_KEY=your_api_key

# Legacy Projects (AI ACCOUNTS, CORPORATE LOGIN, etc.)
# ... (see template files for complete configuration)
```

### Step 3: Integration Architecture

```
Your System                    JP Morgan APIs
┌─────────────┐               ┌──────────────────────────┐
│             │               │                          │
│  Dashboard  │◄─────────────►│  OpenBanking API         │
│             │               │  - Production            │
│  Payroll    │◄─────────────►│  - UAT                   │
│  Service    │               │                          │
│             │               │  API Gateway             │
│  Auth       │◄─────────────►│  - Production            │
│  Service    │               │  - QAF                   │
│             │               │                          │
│  Bill-Pay   │◄─────────────►│  AI ACCOUNTS             │
│  Service    │               │  (Corporate/Business/    │
│             │               │   Personal)              │
│  Storage    │◄─────────────►│                          │
│  Service    │               │  CORPORATE LOGIN         │
│             │               │  OWL PAYROLL             │
│  Multi-Env  │               │  OWL PETTY CASH          │
│  Support    │               │  Owl1 Integration        │
│             │               │                          │
└─────────────┘               └──────────────────────────┘

Environment Routing:
- Production → openbanking.jpmorgan.com & apigateway.jpmorgan.com
- UAT → openbankinguat.jpmorgan.com
- QAF → apigatewayqaf.jpmorgan.com
```

---

## 📝 Implementation Status

✅ **Completed:**
1. ✅ Multi-environment API client library (`src/jpmorgan_client.py`)
2. ✅ OpenBanking API integration (Production & UAT)
3. ✅ API Gateway integration (Production & QAF)
4. ✅ OAuth 2.0 authentication flow with token caching
5. ✅ Environment-specific configuration templates
6. ✅ Comprehensive documentation
7. ✅ Integration endpoints for all projects
8. ✅ Data synchronization services
9. ✅ Testing suite for JP Morgan APIs

## 🚀 Quick Start

```python
from src.jpmorgan_client import get_jpmorgan_client

# Initialize client for your environment
client = get_jpmorgan_client(environment="production")  # or "uat" or "qaf"

# OpenBanking: Get accounts
accounts = await client.openbanking_get_accounts(user_id="user123")

# OpenBanking: Get transactions
transactions = await client.openbanking_get_transactions(
    account_id="ACC123",
    start_date="2024-01-01"
)

# API Gateway: Execute request
response = await client.apigateway_execute_request(
    method="GET",
    endpoint="/v1/services"
)

# Check health
ob_health = await client.openbanking_health_check()
gw_health = await client.apigateway_health_check()
```

## 📚 Additional Documentation

- **[Endpoints Integration Guide](./JPMORGAN_ENDPOINTS_INTEGRATION.md)** - Comprehensive guide for new endpoints
- **[API Access Guide](./JPMORGAN_API_ACCESS_GUIDE.md)** - How to get API credentials
- **[Configuration Reference](./config.py)** - Configuration options
- **[Test Suite](./tests/test_jpmorgan_endpoints.py)** - Integration tests

## 🔄 Environment Switching

```python
# Switch between environments
client.set_environment("uat")

# Or create environment-specific clients
prod_client = get_jpmorgan_client("production")
uat_client = get_jpmorgan_client("uat")
qaf_client = get_jpmorgan_client("qaf")
```

## 📞 Support

For questions or issues:
- **Technical Documentation:** See `JPMORGAN_ENDPOINTS_INTEGRATION.md`
- **JPMorgan Support:** developer-support@jpmorgan.com
- **Developer Portal:** https://developer.payments.jpmorgan.com
