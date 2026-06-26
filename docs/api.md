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

---

# Payments API

The Payments API provides comprehensive banking suite with card loading, transactions, and instant pay functionality. All endpoints require authentication.

## Payment Method Management

### Add Payment Method

```http
POST /api/v1/payments/methods
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "type": "card",
  "provider": "visa",
  "last_four": "4242",
  "is_default": true
}
```

**Parameters:**
- `type` (required): Payment method type - `card`, `bank_account`, or `wallet`
- `provider` (optional): Payment provider (e.g., visa, mastercard, chase)
- `last_four` (optional): Last 4 digits of the payment method
- `is_default` (optional): Set as default payment method

**Response:**
```json
{
  "status": "success",
  "message": "Payment method added successfully",
  "payment_method": {
    "id": "pm_123",
    "user_id": "user_456",
    "type": "card",
    "provider": "visa",
    "last_four": "4242",
    "is_default": true,
    "is_active": true,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Get Payment Methods

```http
GET /api/v1/payments/methods
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "payment_methods": [...],
  "count": 1
}
```

### Delete Payment Method

```http
DELETE /api/v1/payments/methods/<method_id>
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "message": "Payment method deleted successfully"
}
```

## Card Loading

### Load Funds on Card

```http
POST /api/v1/payments/load
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "method_id": "pm_123",
  "amount": 500.00,
  "currency": "USD"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Card loaded successfully",
  "payment_id": "pay_abc123",
  "amount": 500.00,
  "currency": "USD",
  "new_balance": 1750.75
}
```

### Get Card Balance

```http
GET /api/v1/payments/cards/<card_id>/balance
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "balance": {
    "card_id": "card_123",
    "available_balance": 1250.75,
    "pending_balance": 0.0,
    "currency": "USD",
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

## Transaction Processing

### Process Payment

```http
POST /api/v1/payments/process
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "amount": 100.00,
  "payment_type": "card",
  "description": "Purchase at merchant",
  "currency": "USD",
  "method_id": "pm_123"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Payment processed successfully",
  "payment": {
    "id": "pay_abc123",
    "amount": 100.00,
    "currency": "USD",
    "status": "completed"
  }
}
```

### Get Transactions

```http
GET /api/v1/payments/transactions
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page, max 100 (default: 20)
- `status` (optional): Filter by status (PENDING, COMPLETED, FAILED)
- `type` (optional): Filter by payment type

**Response:**
```json
{
  "status": "success",
  "transactions": [...],
  "count": 20,
  "total": 100,
  "page": 1,
  "limit": 20,
  "pages": 5
}
```

### Get Transaction Details

```http
GET /api/v1/payments/transactions/<transaction_id>
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "transaction": {
    "id": "pay_abc123",
    "amount": 100.00,
    "currency": "USD",
    "status": "completed"
  }
}
```

## Instant Pay

### Quick Pay

```http
POST /api/v1/payments/quick-pay
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "recipient_id": "user_789",
  "amount": 50.00,
  "description": "Quick payment",
  "currency": "USD"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Quick pay processed instantly",
  "payment_id": "pay_quick123",
  "amount": 50.00,
  "recipient_id": "user_789",
  "processed_at": "2024-01-15T10:30:00Z"
}
```

### Transfer

```http
POST /api/v1/payments/transfer
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "to_account": "1234567890",
  "amount": 1000.00,
  "transfer_type": "instant",
  "description": "Transfer to savings"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Instant transfer processed successfully",
  "payment_id": "pay_trans123",
  "amount": 1000.00,
  "to_account": "1234567890",
  "transfer_type": "instant",
  "estimated_completion": "instant"
}
```

### Get Payment Status

```http
GET /api/v1/payments/status/<payment_id>
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "payment_status": {
    "payment_id": "pay_abc123",
    "status": "completed",
    "amount": 100.00,
    "currency": "USD",
    "created_at": "2024-01-15T10:29:00Z",
    "processed_at": "2024-01-15T10:29:05Z"
  }
}
```

## Dashboard & Analytics

### Payments Dashboard

```http
GET /api/v1/payments/dashboard
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "dashboard": {
    "total_payments": 50,
    "total_amount": 15000.00,
    "recent_transactions": [...],
    "status_summary": [
      {"status": "completed", "count": 45},
      {"status": "pending", "count": 3},
      {"status": "failed", "count": 2}
    ]
  }
}
```

### Payment Alerts

```http
GET /api/v1/payments/alerts
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "alerts": {
    "all": [...],
    "active": [...],
    "active_count": 2
  }
}
```

### Payment Statistics

```http
GET /api/v1/payments/stats
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "stats": {
    "user": {
      "total_payments": 50,
      "total_amount": 15000.00,
      "successful_payments": 45,
      "failed_payments": 2
    },
    "global": {...}
  }
}
```

---

# Banking API

The Banking API provides personal bank account management and transactions.

## Account Management

### List Accounts

```http
GET /api/v1/accounts
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "accounts": [
    {
      "account_id": 12345,
      "account_type": "checking",
      "balance": 5000.00,
      "currency": "USD"
    }
  ],
  "count": 1
}
```

### Create Account

```http
POST /api/v1/accounts
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "account_type": "checking",
  "initial_balance": 1000.00
}
```

**Response:**
```json
{
  "status": "success",
  "account": {...},
  "message": "Account created successfully"
}
```

### Get Account Details

```http
GET /api/v1/accounts/<account_id>
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "account": {...}
}
```

### Update Account

```http
PUT /api/v1/accounts/<account_id>
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "interest_rate": 0.015,
  "overdraft_limit": 500.00
}
```

### Validate Account

```http
POST /api/v1/accounts/<account_id>/validate
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "min_balance": 100.00
}
```

**Response:**
```json
{
  "status": "success",
  "validation": {
    "is_valid": true,
    "has_sufficient_funds": true
  }
}
```

## Account Transactions

### List Transactions

```http
GET /api/v1/accounts/<account_id>/transactions
Authorization: Bearer <token>
```

**Query Parameters:**
- `limit` (optional): Max results, max 100 (default: 50)

**Response:**
```json
{
  "status": "success",
  "transactions": [...],
  "count": 20
}
```

### Create Transaction

```http
POST /api/v1/accounts/<account_id>/transactions
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "type": "deposit",
  "amount": 100.00,
  "description": "Deposit check"
}
```

---

# Transfers API

The Transfers API provides endpoints for wire/ACH/RTP transfers.

## Transfer Operations

### Create Transfer

```http
POST /api/v1/transfers
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "transfer_type": "ach",
  "direction": "outgoing",
  "amount": 5000.00,
  "to_account_number": "1234567890",
  "description": "Monthly rent payment"
}
```

**Transfer Types:**
- `ach`: ACH transfer (0-3 business days)
- `wire`: Wire transfer (same day, fees apply)
- `rtp`: RTP instant transfer
- `internal`: Internal account transfer

**Response:**
```json
{
  "status": "success",
  "transfer": {
    "transfer_id": "ACH-A1B2C3D4",
    "transfer_type": "ach",
    "direction": "outgoing",
    "amount": 5000.00,
    "currency": "USD",
    "fee": 0,
    "status": "pending",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### List Transfers

```http
GET /api/v1/transfers
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "transfers": [...],
  "count": 10
}
```

### Get Transfer Details

```http
GET /api/v1/transfers/<transfer_id>
Authorization: Bearer <token>
```

### Cancel Transfer

```http
POST /api/v1/transfers/<transfer_id>/cancel
Authorization: Bearer <token>
```

### Complete Transfer

```http
POST /api/v1/transfers/<transfer_id>/complete
Authorization: Bearer <token>
```

## Fee Schedule

### Get Fees

```http
GET /api/v1/fees
```

**Response:**
```json
{
  "status": "success",
  "fees": {
    "internal": 0,
    "ach": 0,
    "domestic_wire": 25,
    "international_wire": 50,
    "rtp": 0
  },
  "limits": {
    "ach": {"min": 0.01, "max": 100000, "daily": 250000},
    "wire": {"min": 1, "max": 1000000, "daily": 5000000},
    "rtp": {"min": 0.01, "max": 100000, "daily": 250000}
  }
}
```

---

# Payroll API

The Payroll API provides employee management and payroll processing.

## Employee Management

### Create Employee

```http
POST /api/v1/employees
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "first_name": "John",
  "last_name": "Smith",
  "email": "john.smith@example.com",
  "phone": "555-0100",
  "department": "Engineering",
  "position": "Senior Developer",
  "salary": 120000,
  "pay_frequency": "biweekly"
}
```

**Response:**
```json
{
  "status": "success",
  "employee": {
    "employee_id": "emp_123",
    "first_name": "John",
    "last_name": "Smith",
    "email": "john.smith@example.com",
    "status": "active"
  }
}
```

### List Employees

```http
GET /api/v1/employees
Authorization: Bearer <token>
```

### Get Employee

```http
GET /api/v1/employees/<employee_id>
Authorization: Bearer <token>
```

### Update Employee

```http
PUT /api/v1/employees/<employee_id>
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "position": "Lead Developer",
  "salary": 130000
}
```

### Delete Employee

```http
DELETE /api/v1/employees/<employee_id>
Authorization: Bearer <token>
```

## Payroll Runs

### Create Payroll Run

```http
POST /api/v1/runs
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "pay_period_start": "2024-01-01",
  "pay_period_end": "2024-01-14",
  "payment_date": "2024-01-15"
}
```

### List Payroll Runs

```http
GET /api/v1/runs
Authorization: Bearer <token>
```

### Get Payroll Run

```http
GET /api/v1/runs/<run_id>
Authorization: Bearer <token>
```

### Process Payroll Run

```http
POST /api/v1/runs/<run_id>/process
Authorization: Bearer <token>
```

## Employee Payments

### Get Employee Payments

```http
GET /api/v1/employees/<employee_id>/payments
Authorization: Bearer <token>
```

### Calculate Employee Taxes

```http
GET /api/v1/employees/<employee_id>/taxes
Authorization: Bearer <token>
```

---

# Legacy API Endpoints

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
