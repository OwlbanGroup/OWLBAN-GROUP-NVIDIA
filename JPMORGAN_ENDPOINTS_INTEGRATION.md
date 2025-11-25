# JPMorgan API Endpoints Integration Guide

## 🎯 Overview

This guide covers the integration of JPMorgan's OpenBanking and API Gateway endpoints across multiple environments (Production, UAT, QAF).

---

## 🌐 Available Endpoints

### Production Environment

#### OpenBanking API
- **URL:** `https://openbanking.jpmorgan.com/accessapi`
- **Purpose:** Production OpenBanking services
- **Environment:** `production`
- **Use Case:** Live banking operations, account access, transactions

#### API Gateway
- **URL:** `https://apigateway.jpmorgan.com/accessapi`
- **Purpose:** Production API Gateway for service orchestration
- **Environment:** `production`
- **Use Case:** Service routing, API management, authentication

### UAT Environment

#### OpenBanking API - UAT
- **URL:** `https://openbankinguat.jpmorgan.com/accessapi`
- **Purpose:** User Acceptance Testing for OpenBanking
- **Environment:** `uat`
- **Use Case:** Pre-production testing, user acceptance validation

### QAF Environment

#### API Gateway - QAF
- **URL:** `https://apigatewayqaf.jpmorgan.com/accessapi`
- **Purpose:** Quality Assurance and Functional Testing
- **Environment:** `qaf`
- **Use Case:** Comprehensive testing, quality assurance

---

## 🔧 Configuration

### Environment Setup

1. **Choose your environment template:**
   ```bash
   # For Production
   cp .env.production.template .env.production
   
   # For UAT
   cp .env.uat.template .env.uat
   
   # For QAF
   cp .env.qaf.template .env.qaf
   ```

2. **Fill in your credentials:**
   - OpenBanking Client ID, Secret, and API Key
   - API Gateway Client ID, Secret, and API Key
   - Database and Redis connection strings
   - Security keys

3. **Set the active environment:**
   ```bash
   # In your .env file
   JPMORGAN_ENVIRONMENT=production  # or 'uat' or 'qaf'
   ```

### Environment Variables

#### OpenBanking Configuration
```bash
# Production
JPMORGAN_OPENBANKING_PRODUCTION_URL=https://openbanking.jpmorgan.com/accessapi
JPMORGAN_OPENBANKING_CLIENT_ID=your_client_id
JPMORGAN_OPENBANKING_CLIENT_SECRET=your_client_secret
JPMORGAN_OPENBANKING_API_KEY=your_api_key

# UAT
JPMORGAN_OPENBANKING_UAT_URL=https://openbankinguat.jpmorgan.com/accessapi
```

#### API Gateway Configuration
```bash
# Production
JPMORGAN_APIGATEWAY_PRODUCTION_URL=https://apigateway.jpmorgan.com/accessapi
JPMORGAN_APIGATEWAY_CLIENT_ID=your_client_id
JPMORGAN_APIGATEWAY_CLIENT_SECRET=your_client_secret
JPMORGAN_APIGATEWAY_API_KEY=your_api_key

# QAF
JPMORGAN_APIGATEWAY_QAF_URL=https://apigatewayqaf.jpmorgan.com/accessapi
```

---

## 💻 Usage Examples

### Python Client Usage

#### Initialize Client

```python
from src.jpmorgan_client import get_jpmorgan_client

# Production environment
client = get_jpmorgan_client(environment="production")

# UAT environment
client_uat = get_jpmorgan_client(environment="uat")

# QAF environment
client_qaf = get_jpmorgan_client(environment="qaf")
```

#### OpenBanking API Examples

##### Health Check
```python
# Check OpenBanking API health
health = await client.openbanking_health_check()
print(f"Status: {health['status']}")
print(f"Environment: {health['environment']}")
```

##### Get Accounts
```python
# Retrieve user accounts
accounts = await client.openbanking_get_accounts(user_id="user123")
for account in accounts:
    print(f"Account: {account['account_id']} - {account['account_type']}")
```

##### Get Transactions
```python
# Get account transactions
transactions = await client.openbanking_get_transactions(
    account_id="ACC123456",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
print(f"Found {len(transactions)} transactions")
```

##### Get Balance
```python
# Check account balance
balance = await client.openbanking_get_balance(account_id="ACC123456")
print(f"Balance: {balance['amount']} {balance['currency']}")
```

#### API Gateway Examples

##### Health Check
```python
# Check API Gateway health
health = await client.apigateway_health_check()
print(f"Gateway Status: {health['status']}")
```

##### Execute Custom Request
```python
# Execute a custom API request through the gateway
response = await client.apigateway_execute_request(
    method="GET",
    endpoint="/v1/services/payments",
    params={"status": "active"}
)
```

##### Get Available Services
```python
# List all available services
services = await client.apigateway_get_services()
for service in services:
    print(f"Service: {service['name']} - {service['status']}")
```

##### Get API Status
```python
# Get gateway status and metrics
status = await client.apigateway_get_api_status()
print(f"Uptime: {status['uptime']}")
print(f"Request Count: {status['request_count']}")
```

### Environment Switching

```python
# Switch environments dynamically
client.set_environment("uat")
print(f"Now using: {client.environment}")

# Get service URLs for current environment
openbanking_url = client.get_service_url("openbanking")
apigateway_url = client.get_service_url("apigateway")
```

---

## 🔐 Authentication Flow

### OAuth 2.0 Client Credentials Flow

1. **Request Access Token:**
   ```
   POST {auth_url}/oauth/token
   Content-Type: application/x-www-form-urlencoded
   
   grant_type=client_credentials
   client_id={your_client_id}
   client_secret={your_client_secret}
   scope=payments
   ```

2. **Receive Token:**
   ```json
   {
     "access_token": "eyJhbGciOiJSUzI1NiIs...",
     "token_type": "Bearer",
     "expires_in": 3600
   }
   ```

3. **Use Token in Requests:**
   ```
   GET {service_url}/accounts
   Authorization: Bearer {access_token}
   X-API-Key: {your_api_key}
   Content-Type: application/json
   ```

### Token Caching

The client automatically caches tokens and refreshes them before expiration:
- Tokens are cached per project
- Automatic refresh 60 seconds before expiration
- Thread-safe token management

---

## 🧪 Testing

### Health Check Tests

```python
import asyncio
from src.jpmorgan_client import get_jpmorgan_client

async def test_all_endpoints():
    """Test all endpoint health checks"""
    
    # Test Production
    client_prod = get_jpmorgan_client("production")
    ob_health = await client_prod.openbanking_health_check()
    gw_health = await client_prod.apigateway_health_check()
    
    print(f"Production OpenBanking: {ob_health['status']}")
    print(f"Production API Gateway: {gw_health['status']}")
    
    # Test UAT
    client_uat = get_jpmorgan_client("uat")
    uat_health = await client_uat.openbanking_health_check()
    print(f"UAT OpenBanking: {uat_health['status']}")
    
    # Test QAF
    client_qaf = get_jpmorgan_client("qaf")
    qaf_health = await client_qaf.apigateway_health_check()
    print(f"QAF API Gateway: {qaf_health['status']}")
    
    # Cleanup
    await client_prod.close()
    await client_uat.close()
    await client_qaf.close()

# Run tests
asyncio.run(test_all_endpoints())
```

### Integration Tests

```bash
# Run comprehensive test suite
python -m pytest tests/test_jpmorgan_endpoints.py -v

# Run specific environment tests
python -m pytest tests/test_jpmorgan_endpoints.py::test_production_endpoints -v
python -m pytest tests/test_jpmorgan_endpoints.py::test_uat_endpoints -v
python -m pytest tests/test_jpmorgan_endpoints.py::test_qaf_endpoints -v
```

---

## 🚨 Error Handling

### Common Errors

#### 401 Unauthorized
```python
try:
    accounts = await client.openbanking_get_accounts(user_id="user123")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Authentication failed. Check credentials.")
```

#### 403 Forbidden
```python
try:
    response = await client.apigateway_execute_request("GET", "/admin")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 403:
        print("Access denied. Insufficient permissions.")
```

#### 429 Rate Limit
```python
import time

try:
    transactions = await client.openbanking_get_transactions("ACC123")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        retry_after = int(e.response.headers.get("Retry-After", 60))
        print(f"Rate limited. Retry after {retry_after} seconds")
        time.sleep(retry_after)
```

#### Network Errors
```python
try:
    health = await client.openbanking_health_check()
except httpx.ConnectError:
    print("Connection failed. Check network connectivity.")
except httpx.TimeoutException:
    print("Request timed out. Service may be slow.")
```

---

## 📊 Monitoring & Logging

### Enable Debug Logging

```python
import logging
import structlog

# Configure logging
logging.basicConfig(level=logging.DEBUG)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
)
```

### Monitor API Calls

```python
from src.jpmorgan_client import get_jpmorgan_client

client = get_jpmorgan_client("production")

# All API calls are automatically logged
accounts = await client.openbanking_get_accounts("user123")
# Logs: "Retrieved OpenBanking accounts" with user_id and count
```

---

## 🔒 Security Best Practices

### 1. Credential Management
- ✅ Store credentials in environment variables
- ✅ Use separate credentials per environment
- ✅ Never commit `.env` files to version control
- ✅ Rotate credentials every 90 days
- ✅ Use strong, unique API keys

### 2. Network Security
- ✅ Always use HTTPS
- ✅ Validate SSL certificates
- ✅ Implement request timeouts
- ✅ Use IP whitelisting when available

### 3. Access Control
- ✅ Follow principle of least privilege
- ✅ Separate production and non-production access
- ✅ Monitor API usage for anomalies
- ✅ Implement rate limiting

### 4. Data Protection
- ✅ Encrypt sensitive data at rest
- ✅ Use secure token storage
- ✅ Implement audit logging
- ✅ Comply with data retention policies

---

## 🐛 Troubleshooting

### Issue: "Missing credentials for project"
**Solution:** Ensure all required environment variables are set:
```bash
# Check if variables are set
echo $JPMORGAN_OPENBANKING_CLIENT_ID
echo $JPMORGAN_APIGATEWAY_CLIENT_ID
```

### Issue: "Connection refused"
**Solution:** Verify the endpoint URL and network connectivity:
```bash
# Test connectivity
curl -I https://openbanking.jpmorgan.com/accessapi/health
```

### Issue: "Invalid token"
**Solution:** Clear token cache and re-authenticate:
```python
client.tokens.clear()  # Clear cached tokens
token = await client.get_access_token("openbanking")
```

### Issue: "Environment not switching"
**Solution:** Create a new client instance:
```python
# Don't reuse client across environments
client_prod = get_jpmorgan_client("production")
client_uat = get_jpmorgan_client("uat")  # Creates new instance
```

---

## 📞 Support

### JPMorgan Support Channels
- **Developer Portal:** https://developer.payments.jpmorgan.com
- **Email Support:** developer-support@jpmorgan.com
- **Status Page:** https://status.jpmorgan.com
- **Documentation:** https://developer.payments.jpmorgan.com/docs

### Internal Support
- **Technical Issues:** Check logs in `logs/telemetry.log`
- **Configuration Help:** Review `.env.*.template` files
- **API Questions:** Consult `JPMORGAN_API_INTEGRATION.md`

---

## 📝 Changelog

### Version 2.0.0 (2024-01-XX)
- ✅ Added OpenBanking API integration
- ✅ Added API Gateway integration
- ✅ Multi-environment support (Production, UAT, QAF)
- ✅ Environment-specific configuration templates
- ✅ Enhanced error handling and logging
- ✅ Comprehensive documentation

### Version 1.0.0
- Initial release with legacy API support

---

## 📚 Additional Resources

- [JPMorgan API Access Guide](./JPMORGAN_API_ACCESS_GUIDE.md)
- [JPMorgan API Integration](./JPMORGAN_API_INTEGRATION.md)
- [Configuration Reference](./config.py)
- [API Client Source](./src/jpmorgan_client.py)
- [Test Suite](./tests/test_jpmorgan_endpoints.py)

---

**Last Updated:** 2024-01-XX  
**Maintained By:** Development Team  
**Version:** 2.0.0
