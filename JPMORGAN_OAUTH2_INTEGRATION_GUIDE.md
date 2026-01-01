# JPMorgan Payments API - OAuth2 Integration Guide

## 🚀 Complete Production-Ready NestJS Implementation

This guide covers the complete OAuth2 integration with JPMorgan Payments API, including token management, API calls, and Grafana integration.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Environment Setup](#environment-setup)
3. [OAuth2 Flow](#oauth2-flow)
4. [API Endpoints](#api-endpoints)
5. [Grafana Integration](#grafana-integration)
6. [Security Best Practices](#security-best-practices)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  JPMorgan OAuth2 Integration                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │  Your Backend    │      │  JPMorgan API    │           │
│  │  (NestJS)        │      │                  │           │
│  │                  │      │                  │           │
│  │  ┌────────────┐  │      │  ┌────────────┐ │           │
│  │  │ Token      │──┼──1──→│  │ OAuth2     │ │           │
│  │  │ Service    │←─┼──2───│  │ Token      │ │           │
│  │  └────────────┘  │      │  │ Endpoint   │ │           │
│  │        │         │      │  └────────────┘ │           │
│  │        │ token   │      │                  │           │
│  │        ▼         │      │                  │           │
│  │  ┌────────────┐  │      │  ┌────────────┐ │           │
│  │  │ JPMorgan   │──┼──3──→│  │ Accounts   │ │           │
│  │  │ Service    │←─┼──4───│  │ Balances   │ │           │
│  │  └────────────┘  │      │  │ Payments   │ │           │
│  │        │         │      │  └────────────┘ │           │
│  │        ▼         │      │                  │           │
│  │  ┌────────────┐  │      └──────────────────┘           │
│  │  │ Controller │  │                                      │
│  │  │ (REST API) │  │                                      │
│  │  └────────────┘  │                                      │
│  │        │         │                                      │
│  └────────┼─────────┘                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                      │
│  │    Grafana       │                                      │
│  │  (Visualization) │                                      │
│  └──────────────────┘                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Flow:
1. Request access token with client credentials
2. Receive and cache access token
3. Use token to call JPMorgan APIs
4. Return data to client/Grafana
```

---

## 🔧 Environment Setup

### 1. Environment Variables

Create or update your `.env` file:

```bash
# JPMorgan OAuth2 Configuration
JPM_CLIENT_ID=your_client_id_here
JPM_CLIENT_SECRET=your_client_secret_here
JPM_TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token
JPM_SCOPE=jpm:payments:sandbox
JPM_API_BASE_URL=https://api-sandbox.payments.jpmorgan.com

# For Production
# JPM_TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/access_token
# JPM_SCOPE=jpm:payments:production
# JPM_API_BASE_URL=https://api.payments.jpmorgan.com
```

### 2. Get JPMorgan Credentials

1. **Register Your Application:**
   - Visit JPMorgan Developer Portal
   - Create a new application
   - Note your `client_id` and `client_secret`

2. **Request API Access:**
   - Request access to required scopes:
     - `jpm:payments:sandbox` (for testing)
     - `jpm:payments:accounts:read`
     - `jpm:payments:balances:read`
     - `jpm:payments:transactions:read`
     - `jpm:payments:ach:write`

3. **Configure Sandbox:**
   - Use sandbox credentials for development
   - Test all flows before production

---

## 🔐 OAuth2 Flow

### Client Credentials Grant

JPMorgan uses the **OAuth2 Client Credentials** flow:

```typescript
// Automatic token management in JpmorganTokenService

POST https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token
Content-Type: application/x-www-form-urlencoded

client_id=YOUR_CLIENT_ID
&client_secret=YOUR_CLIENT_SECRET
&grant_type=client_credentials
&scope=jpm:payments:sandbox

Response:
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "jpm:payments:sandbox"
}
```

### Token Caching

The `JpmorganTokenService` automatically:
- ✅ Caches tokens in memory
- ✅ Refreshes before expiry (30s buffer)
- ✅ Handles token invalidation
- ✅ Thread-safe token management

---

## 📡 API Endpoints

### 1. Get Balances (Grafana Compatible)

```bash
GET http://localhost:4000/api/jpmorgan/balances
GET http://localhost:4000/api/jpmorgan/balances?connectionRef=conn_123

Response:
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": [
    {
      "accountId": "acc_123",
      "availableBalance": "50000.00",
      "currentBalance": "52000.00",
      "currency": "USD",
      "asOf": "2024-01-15T10:00:00.000Z"
    }
  ],
  "meta": {
    "count": 1,
    "connectionRef": "conn_123"
  }
}
```

### 2. Get Accounts

```bash
GET http://localhost:4000/api/jpmorgan/accounts
GET http://localhost:4000/api/jpmorgan/accounts?connectionRef=conn_123

Response:
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": [
    {
      "id": "acc_123",
      "accountNumber": "****1234",
      "accountName": "Business Checking",
      "accountType": "CHECKING",
      "currency": "USD"
    }
  ],
  "meta": {
    "count": 1,
    "connectionRef": "conn_123"
  }
}
```

### 3. Get Transactions

```bash
GET http://localhost:4000/api/jpmorgan/transactions
GET http://localhost:4000/api/jpmorgan/transactions?accountId=acc_123&startDate=2024-01-01&endDate=2024-01-31

Response:
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": [
    {
      "id": "txn_456",
      "accountId": "acc_123",
      "amount": "1500.00",
      "currency": "USD",
      "description": "Payment received",
      "postedAt": "2024-01-14T15:30:00.000Z",
      "type": "CREDIT"
    }
  ],
  "meta": {
    "count": 1,
    "accountId": "acc_123",
    "dateRange": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    }
  }
}
```

### 4. Get Payment Status

```bash
GET http://localhost:4000/api/jpmorgan/payments/pmt_789

Response:
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "paymentId": "pmt_789",
    "status": "SETTLED",
    "debitAccount": "acc_123",
    "creditAccount": "acc_456",
    "amount": "5000.00",
    "currency": "USD",
    "createdAt": "2024-01-14T10:00:00.000Z"
  }
}
```

---

## 📊 Grafana Integration

### Setup Grafana Data Source

1. **Install JSON API Plugin:**
   ```bash
   grafana-cli plugins install marcusolsson-json-datasource
   ```

2. **Add Data Source:**
   - Go to **Configuration → Data Sources**
   - Click **Add data source**
   - Select **JSON API**
   - Configure:
     ```
     Name: JPMorgan Balances
     URL: http://your-backend:4000/api/jpmorgan/balances
     ```

3. **Create Dashboard:**

```json
{
  "dashboard": {
    "title": "JPMorgan Account Balances",
    "panels": [
      {
        "title": "Available Balance",
        "type": "stat",
        "targets": [
          {
            "datasource": "JPMorgan Balances",
            "jsonPath": "$.data[*].availableBalance"
          }
        ]
      },
      {
        "title": "Balance Trend",
        "type": "graph",
        "targets": [
          {
            "datasource": "JPMorgan Balances",
            "jsonPath": "$.data[*]",
            "fields": [
              {
                "name": "time",
                "jsonPath": "$.asOf"
              },
              {
                "name": "balance",
                "jsonPath": "$.currentBalance"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### Query Examples

**Balance by Account:**
```
$.data[?(@.accountId=='acc_123')].availableBalance
```

**Total Balance:**
```
$.data[*].currentBalance
```

**Account Count:**
```
$.meta.count
```

---

## 🛡️ Security Best Practices

### 1. Environment Variables
- ✅ Never commit `.env` files
- ✅ Use `.env.example` for templates
- ✅ Rotate credentials regularly
- ✅ Use different credentials per environment

### 2. API Security
- ✅ Use HTTPS in production
- ✅ Implement rate limiting
- ✅ Add API key authentication for your endpoints
- ✅ Log only non-sensitive data
- ✅ Implement IP allowlisting

### 3. Token Management
- ✅ Tokens cached in memory only
- ✅ Auto-refresh before expiry
- ✅ No token logging
- ✅ Secure token transmission

### 4. Production Checklist
```bash
# ✅ Use production URLs
JPM_TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/access_token
JPM_API_BASE_URL=https://api.payments.jpmorgan.com

# ✅ Enable HTTPS
# ✅ Configure firewall rules
# ✅ Set up monitoring
# ✅ Enable audit logging
# ✅ Implement backup strategy
```

---

## 🧪 Testing

### 1. Test Token Service

```bash
# Start the backend
cd nestjs-backend
npm run start:dev

# Check logs for token acquisition
# Should see: "Successfully obtained new access token"
```

### 2. Test API Endpoints

```bash
# Test balances endpoint
curl http://localhost:4000/api/jpmorgan/balances

# Test accounts endpoint
curl http://localhost:4000/api/jpmorgan/accounts

# Test transactions endpoint
curl "http://localhost:4000/api/jpmorgan/transactions?startDate=2024-01-01"
```

### 3. Test Token Caching

```bash
# Make multiple requests quickly
for i in {1..5}; do
  curl http://localhost:4000/api/jpmorgan/balances
  echo ""
done

# Check logs - should see "Using cached access token" after first request
```

### 4. Test Error Handling

```bash
# Test with invalid credentials
# Update .env with wrong credentials
# Restart server
# Should see: "Failed to authenticate with JPMorgan API"
```

---

## 🔍 Troubleshooting

### Issue: "Failed to authenticate with JPMorgan API"

**Causes:**
- Invalid `client_id` or `client_secret`
- Incorrect `JPM_TOKEN_URL`
- Network connectivity issues
- Expired credentials

**Solutions:**
```bash
# 1. Verify credentials
echo $JPM_CLIENT_ID
echo $JPM_CLIENT_SECRET

# 2. Test token endpoint directly
curl -X POST https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=YOUR_ID&client_secret=YOUR_SECRET&grant_type=client_credentials&scope=jpm:payments:sandbox"

# 3. Check network
ping id.payments.jpmorgan.com

# 4. Review logs
tail -f logs/application.log
```

### Issue: "Token expired" errors

**Causes:**
- Token not refreshing properly
- System clock skew
- Token expiry too short

**Solutions:**
```typescript
// Check token validity
const isValid = tokenService.isTokenValid();
console.log('Token valid:', isValid);

// Force token refresh
tokenService.invalidateToken();
const newToken = await tokenService.getAccessToken();
```

### Issue: Grafana not showing data

**Causes:**
- Incorrect data source URL
- JSON path errors
- CORS issues
- Authentication problems

**Solutions:**
```bash
# 1. Test endpoint directly
curl http://localhost:4000/api/jpmorgan/balances

# 2. Check Grafana data source settings
# 3. Review browser console for CORS errors
# 4. Verify JSON path in query

# 5. Enable CORS if needed (main.ts)
app.enableCors({
  origin: 'http://localhost:3000', // Grafana URL
  credentials: true,
});
```

### Issue: Rate limiting errors

**Causes:**
- Too many requests
- JPMorgan API limits exceeded

**Solutions:**
```typescript
// Implement request throttling
import { ThrottlerModule } from '@nestjs/throttler';

@Module({
  imports: [
    ThrottlerModule.forRoot({
      ttl: 60,
      limit: 10, // 10 requests per minute
    }),
  ],
})
```

---

## 📈 Monitoring & Logging

### Log Levels

```typescript
// Development
LOG_LEVEL=debug

// Production
LOG_LEVEL=info
```

### Key Metrics to Monitor

1. **Token Acquisition:**
   - Success rate
   - Response time
   - Failure reasons

2. **API Calls:**
   - Request count
   - Response times
   - Error rates
   - Status codes

3. **Cache Performance:**
   - Hit rate
   - Miss rate
   - Token refresh frequency

### Sample Prometheus Metrics

```typescript
// Add to jpmorgan.service.ts
import { Counter, Histogram } from 'prom-client';

const apiCallCounter = new Counter({
  name: 'jpmorgan_api_calls_total',
  help: 'Total JPMorgan API calls',
  labelNames: ['endpoint', 'status'],
});

const apiCallDuration = new Histogram({
  name: 'jpmorgan_api_call_duration_seconds',
  help: 'JPMorgan API call duration',
  labelNames: ['endpoint'],
});
```

---

## 🚀 Next Steps

1. **Implement Additional Endpoints:**
   - Payments initiation
   - Account details
   - Transaction search
   - Webhook handlers

2. **Add Caching Layer:**
   - Redis for distributed caching
   - Cache invalidation strategies
   - TTL configuration

3. **Enhance Security:**
   - API key authentication
   - IP allowlisting
   - Request signing
   - Audit logging

4. **Improve Monitoring:**
   - Prometheus metrics
   - Grafana dashboards
   - Alert rules
   - Health checks

5. **Production Deployment:**
   - Load balancing
   - Auto-scaling
   - Backup strategies
   - Disaster recovery

---

## 📚 Additional Resources

- [JPMorgan Developer Portal](https://developer.jpmorgan.com)
- [OAuth2 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [NestJS Documentation](https://docs.nestjs.com)
- [Grafana JSON API Plugin](https://grafana.com/grafana/plugins/marcusolsson-json-datasource/)

---

## 💡 Support

For issues or questions:
1. Check the troubleshooting section
2. Review JPMorgan API documentation
3. Check application logs
4. Contact JPMorgan support

---

**Last Updated:** January 2024  
**Version:** 1.0.0  
**Status:** Production Ready ✅
