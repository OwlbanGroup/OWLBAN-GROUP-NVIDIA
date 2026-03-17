# JPMorgan Financial APIs - Developer Transactions Guide

## Overview

This guide provides developers with complete instructions for **accessing accounts** and **performing transactions** using the JPMorgan Financial APIs. Covers authentication, account management, transfers, payments, and banking operations.

**Base URL**: `http://localhost:8000` (dev)  
**Docs**: `/api/docs` (Swagger/OpenAPI)  
**Auth**: JWT Bearer token (see [AUTH_GUIDE.md](AUTH_GUIDE.md))

**See also**: [USER_GUIDE.md](USER_GUIDE.md) for business/PFM features.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Authentication](#authentication)
- [Account Access & Management](#accounts)
- [Transactions](#transactions)
  - [Banking Transactions](#banking)
  - [Transfers](#transfers)
  - [Payments](#payments)
- [Examples](#examples)
- [Rate Limits & Fees](#limits)
- [Error Handling](#errors)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Prerequisites {#prerequisites}

1. **Start the API server**:
   ```bash
   cd jpmorgan_financial_apis
   python app.py
   ```
   Server runs at `http://localhost:8000`.

2. **HTTP Client**: curl, Postman, or code.

3. **Test Account**: Register/login for token.

## Authentication {#authentication}

All endpoints require `Authorization: Bearer <token>`.

### Step 1: Register
```bash
curl -X POST http://localhost:8000/user/register \
  -H 'Content-Type: application/json' \
  -d '{\"username\": \"devuser\", \"password\": \"securepass123\"}'
```

### Step 2: Login (get token)
```bash
curl -X POST http://localhost:8000/user/login \
  -H 'Content-Type: application/json' \
  -d '{\"username\": \"devuser\", \"password\": \"securepass123\"}' | jq -r '.token'
```
Save token: `YOUR_TOKEN_HERE`.

**Token expires**: 24h. Re-login when 401.

## Account Access & Management {#accounts}

Banking blueprint: `/banking/accounts`

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/banking/accounts` | List accounts | 20/min |
| POST | `/banking/accounts` | Create account | 5/min |
| GET | `/banking/accounts/:id` | Get account | 20/min |
| PUT | `/banking/accounts/:id` | Update account | 5/min |
| POST | `/banking/accounts/:id/validate` | Validate balance | 20/min |

### Examples

**List Accounts**:
```bash
curl -H \"Authorization: Bearer YOUR_TOKEN\" http://localhost:8000/banking/accounts
```

**Create Checking Account**:
```bash
curl -X POST http://localhost:8000/banking/accounts \\
  -H \"Authorization: Bearer YOUR_TOKEN\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"account_type\": \"checking\", \"initial_balance\": 1000.0}'
```

## Transactions {#transactions}

### Banking Transactions {#banking}
/banking/accounts/:id/transactions - Deposits, withdrawals, transfers.

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/banking/accounts/:id/transactions` | List txs | 30/min |
| POST | `/banking/accounts/:id/transactions` | Create deposit/withdrawal/transfer | 10/min |

**Deposit $500**:
```bash
curl -X POST http://localhost:8000/banking/accounts/1/transactions \\
  -H \"Authorization: Bearer YOUR_TOKEN\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"type\": \"deposit\", \"amount\": 500.0, \"description\": \"Salary\"}'
```

### Transfers {#transfers}
/transfers - ACH/WIRE/RTP.

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/transfers` | Create transfer | Default |
| GET | `/transfers` | List transfers | Default |
| GET | `/transfers/:id` | Get transfer | Default |
| POST | `/transfers/:id/cancel` | Cancel | Default |

**ACH Outgoing**:
```bash
curl -X POST http://localhost:8000/transfers \\
  -H \"Authorization: Bearer YOUR_TOKEN\" \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"transfer_type\": \"ach\", 
    \"direction\": \"outgoing\", 
    \"amount\": 100.0, 
    \"to_account_number\": \"123456789\",
    \"description\": \"Rent\"
  }'
```

**Fees**: GET `/transfers/fees`.

### Payments {#payments}
/payments - Cards, quick-pay, loads.

Key: POST `/payments/process`, POST `/payments/quick-pay`, GET `/payments/transactions`.

**Quick Pay**:
```bash
curl -X POST http://localhost:8000/payments/quick-pay \\
  -H \"Authorization: Bearer YOUR_TOKEN\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"recipient_id\": \"user123\", \"amount\": 50.0}'
```

## Code Examples {#examples}

### Python Client
```python
import requests

class JPMTransactions:
    def __init__(self, base_url='http://localhost:8000', token=None):
        self.base_url = base_url
        self.session = requests.Session()
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'

    def login(self, username, password):
        resp = self.session.post(f'{self.base_url}/user/login',
                                json={'username': username, 'password': password})
        if resp.ok:
            token = resp.json()['token']
            self.session.headers['Authorization'] = f'Bearer {token}'
            return token

    def create_transfer(self, amount, to_account, transfer_type='ach'):
        resp = self.session.post(f'{self.base_url}/transfers',
                                json={'transfer_type': transfer_type, 'direction': 'outgoing',
                                      'amount': amount, 'to_account_number': to_account})
        return resp.json()

# Usage
client = JPMTransactions()
client.login('devuser', 'securepass123')
print(client.create_transfer(100.0, '123456789'))
```

### JavaScript (Fetch)
```javascript
const API_BASE = 'http://localhost:8000';

async function login(username, password) {
  const resp = await fetch(`${API_BASE}/user/login`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username, password})
  });
  const data = await resp.json();
  return data.token;
}

async function createTransfer(token, amount, toAccount) {
  const resp = await fetch(`${API_BASE}/transfers`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      transfer_type: 'ach',
      direction: 'outgoing',
      amount,
      to_account_number: toAccount
    })
  });
  return resp.json();
}
```

## Rate Limits & Fees {#limits}

**Limits** (per minute):
- Accounts: 5-30
- Transfers/Payments: 10-30

**Transfer Fees**:
| Type | Fee |
|------|-----|
| Internal | $0 |
| ACH | $0 |
| Domestic Wire | $25 |
| International Wire | $50 |

## Error Handling {#errors}

| Code | Meaning | Action |
|------|---------|--------|
| 401 | Invalid/expired token | Re-login |
| 403 | Insufficient permissions | Check role |
| 400 | Validation error | Fix payload |
| 429 | Rate limit | Retry later |

## Best Practices {#best-practices}
- Use HTTPS in production.
- Implement token refresh.
- Exponential backoff for retries.
- Validate responses.
- Monitor `/metrics`.

## Troubleshooting {#troubleshooting}
- **No token?** Check login response.
- **401?** Token expired - re-login.
- **Server down?** `curl http://localhost:8000/health`.
- Logs: Check console output.

**Questions?** See [AUTH_GUIDE.md](AUTH_GUIDE.md) or USER_GUIDE.md.

---

*Last Updated: Current Date*

