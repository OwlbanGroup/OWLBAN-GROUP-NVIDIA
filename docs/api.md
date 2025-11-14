# JPMorgan Financial APIs - API Documentation

## Overview

The JPMorgan Financial APIs provide RESTful access to financial data and services. All APIs require OAuth2 authentication and follow REST conventions.

## Authentication

### OAuth2 Client Credentials Flow

```bash
POST https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Using Access Tokens

Include the access token in the Authorization header:

```bash
Authorization: Bearer <access_token>
```

## Core APIs

### Account Management API

#### List Accounts

```http
GET /api/v1/accounts
Authorization: Bearer <token>
```

**Response:**
```json
{
  "accounts": [
    {
      "accountId": "000000004045701",
      "accountName": "TEST ACCOUNT NAME",
      "branchId": "",
      "bankId": "02100002",
      "bankName": "JPMORGAN CHASE",
      "currency": {
        "code": "USD",
        "description": "US DOLLAR"
      },
      "balanceList": [
        {
          "asOfDate": "2021-12-08",
          "currentDay": true,
          "openingAvailableAmount": 0.00,
          "endingAvailableAmount": 0.00
        }
      ]
    }
  ]
}
```

#### Get Account Details

```http
GET /api/v1/accounts/{accountId}
Authorization: Bearer <token>
```

**Parameters:**
- `accountId` (path): Account identifier

**Response:**
```json
{
  "accountId": "000000004045701",
  "accountName": "TEST ACCOUNT NAME",
  "details": {
    "accountType": "CHECKING",
    "status": "ACTIVE",
    "openedDate": "2020-01-01"
  }
}
```

#### Get Account Balance

```http
GET /api/v1/accounts/{accountId}/balance
Authorization: Bearer <token>
```

**Response:**
```json
{
  "accountId": "000000004045701",
  "balances": [
    {
      "type": "AVAILABLE",
      "amount": 33253003.18,
      "currency": "USD",
      "asOfDate": "2021-12-08"
    }
  ]
}
```

### Market Data API

#### Get Market Quotes

```http
GET /api/v1/market/quotes
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbols` (optional): Comma-separated list of symbols
- `fields` (optional): Fields to return (price, volume, etc.)

**Example:**
```http
GET /api/v1/market/quotes?symbols=AAPL,GOOGL&fields=price,volume
```

**Response:**
```json
{
  "quotes": [
    {
      "symbol": "AAPL",
      "price": 150.25,
      "volume": 45234123,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### Get Historical Data

```http
GET /api/v1/market/history
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbol` (required): Stock symbol
- `startDate` (required): Start date (YYYY-MM-DD)
- `endDate` (required): End date (YYYY-MM-DD)
- `interval` (optional): Data interval (1min, 5min, 1h, 1d)

**Response:**
```json
{
  "symbol": "AAPL",
  "data": [
    {
      "date": "2024-01-15",
      "open": 148.50,
      "high": 151.20,
      "low": 147.80,
      "close": 150.25,
      "volume": 45234123
    }
  ]
}
```

### Trading API

#### Place Order

```http
POST /api/v1/orders
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "accountId": "000000004045701",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "orderType": "MARKET",
  "price": null
}
```

**Response:**
```json
{
  "orderId": "ORD_123456",
  "status": "PENDING",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### List Orders

```http
GET /api/v1/orders
Authorization: Bearer <token>
```

**Query Parameters:**
- `accountId` (optional): Filter by account
- `status` (optional): Filter by status (PENDING, FILLED, CANCELLED)
- `limit` (optional): Maximum results (default: 50)

**Response:**
```json
{
  "orders": [
    {
      "orderId": "ORD_123456",
      "accountId": "000000004045701",
      "symbol": "AAPL",
      "side": "BUY",
      "quantity": 100,
      "status": "FILLED",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### Cancel Order

```http
DELETE /api/v1/orders/{orderId}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "orderId": "ORD_123456",
  "status": "CANCELLED",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

### Analytics API

#### Portfolio Analytics

```http
GET /api/v1/analytics/portfolio
Authorization: Bearer <token>
```

**Query Parameters:**
- `accountId` (required): Account identifier
- `period` (optional): Analysis period (1M, 3M, 6M, 1Y)

**Response:**
```json
{
  "accountId": "000000004045701",
  "period": "3M",
  "analytics": {
    "totalReturn": 5.23,
    "volatility": 0.15,
    "sharpeRatio": 1.85,
    "maxDrawdown": -2.1
  }
}
```

#### Risk Metrics

```http
GET /api/v1/analytics/risk
Authorization: Bearer <token>
```

**Response:**
```json
{
  "riskMetrics": {
    "valueAtRisk": -125000.00,
    "expectedShortfall": -180000.00,
    "beta": 1.15,
    "correlationMatrix": {
      "AAPL": {"MSFT": 0.75, "GOOGL": 0.65},
      "MSFT": {"AAPL": 0.75, "GOOGL": 0.70}
    }
  }
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Success
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid account ID format",
    "details": {
      "field": "accountId",
      "expected": "numeric string"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "requestId": "req_123456"
}
```

## Rate Limiting

API requests are subject to rate limits:

- **Authenticated Requests**: 1000 requests per minute
- **Market Data**: 500 requests per minute
- **Trading Operations**: 100 requests per minute

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1642242600
```

## Pagination

List endpoints support pagination:

```http
GET /api/v1/accounts?page=2&pageSize=50
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "pageSize": 50,
    "totalPages": 10,
    "totalItems": 500,
    "hasNext": true,
    "hasPrev": true
  }
}
```

## Webhooks

Register webhooks for real-time notifications:

```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["ORDER_FILLED", "PRICE_ALERT"],
  "secret": "your_webhook_secret"
}
```

## SDKs and Libraries

### Python SDK

```python
from jpmorgan_api import JPMorganAPI

api = JPMorganAPI(client_id="your_client_id", client_secret="your_client_secret")

# Get accounts
accounts = api.get_accounts()

# Place order
order = api.place_order({
    "accountId": "000000004045701",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100
})
```

### JavaScript SDK

```javascript
import { JPMorganAPI } from 'jpmorgan-api-sdk';

const api = new JPMorganAPI({
  clientId: 'your_client_id',
  clientSecret: 'your_client_secret'
});

// Get market data
const quotes = await api.getQuotes(['AAPL', 'GOOGL']);
```

## Versioning

API versions are specified in the URL path:

- `v1`: Current stable version
- `v2`: Next version (beta)

Breaking changes will be communicated 90 days in advance.

## Support

- **API Status**: [status.jpmorgan.com](https://status.jpmorgan.com)
- **Developer Portal**: [developer.jpmorgan.com](https://developer.jpmorgan.com)
- **Support**: [support.jpmorgan.com](https://support.jpmorgan.com)

---

**API Version**: v1.0.0
**Last Updated**: November 2024
