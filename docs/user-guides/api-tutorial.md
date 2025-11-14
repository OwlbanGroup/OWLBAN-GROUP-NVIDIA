# API Usage Tutorial - JPMorgan Financial APIs

This tutorial guides you through using the JPMorgan Financial APIs for common financial operations including account management, market data retrieval, and trading.

## Prerequisites

- Valid API credentials (Client ID and Client Secret)
- Python 3.8+ or preferred programming language
- `requests` library for Python examples

```bash
pip install requests
```

## Authentication

All API requests require OAuth2 authentication. First, obtain an access token:

### Python Example

```python
import requests

# Token endpoint
token_url = "https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token"

# Your credentials
client_id = "your_client_id"
client_secret = "your_client_secret"

# Request token
response = requests.post(token_url, data={
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret
})

token_data = response.json()
access_token = token_data["access_token"]

# Use token in headers
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}
```

## Getting Account Information

### List All Accounts

```python
import requests

base_url = "http://localhost:8000"  # Change to production URL
endpoint = "/api/v1/accounts"

response = requests.get(f"{base_url}{endpoint}", headers=headers)
accounts = response.json()

print("Your Accounts:")
for account in accounts["accounts"]:
    print(f"- {account['accountName']}: {account['accountId']}")
```

### Get Account Balance

```python
account_id = "000000004045701"  # Replace with actual account ID

endpoint = f"/api/v1/accounts/{account_id}/balance"
response = requests.get(f"{base_url}{endpoint}", headers=headers)

balance_data = response.json()
print(f"Account Balance: ${balance_data['balances'][0]['amount']}")
```

## Retrieving Market Data

### Get Current Quotes

```python
endpoint = "/api/v1/market/quotes"
params = {
    "symbols": "AAPL,GOOGL,MSFT",
    "fields": "price,volume"
}

response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
quotes = response.json()

print("Market Quotes:")
for quote in quotes["quotes"]:
    print(f"{quote['symbol']}: ${quote['price']} (Vol: {quote['volume']})")
```

### Get Historical Data

```python
endpoint = "/api/v1/market/history"
params = {
    "symbol": "AAPL",
    "startDate": "2024-01-01",
    "endDate": "2024-01-31",
    "interval": "1d"
}

response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
history = response.json()

print("Historical Data for AAPL:")
for day in history["data"][:5]:  # Show first 5 days
    print(f"{day['date']}: Open ${day['open']}, Close ${day['close']}")
```

## Placing Trades

### Market Order Example

```python
endpoint = "/api/v1/orders"

order_data = {
    "accountId": "000000004045701",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10,
    "orderType": "MARKET"
}

response = requests.post(f"{base_url}{endpoint}", headers=headers, json=order_data)
order_result = response.json()

print(f"Order placed: {order_result['orderId']}")
```

### Limit Order Example

```python
order_data = {
    "accountId": "000000004045701",
    "symbol": "GOOGL",
    "side": "SELL",
    "quantity": 5,
    "orderType": "LIMIT",
    "price": 2800.00
}

response = requests.post(f"{base_url}{endpoint}", headers=headers, json=order_data)
order_result = response.json()

print(f"Limit order placed: {order_result['orderId']}")
```

## Managing Orders

### List Active Orders

```python
endpoint = "/api/v1/orders"
params = {"status": "PENDING"}

response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
orders = response.json()

print("Active Orders:")
for order in orders["orders"]:
    print(f"{order['orderId']}: {order['symbol']} {order['side']} {order['quantity']}")
```

### Cancel an Order

```python
order_id = "ORD_123456"  # Replace with actual order ID
endpoint = f"/api/v1/orders/{order_id}"

response = requests.delete(f"{base_url}{endpoint}", headers=headers)
cancel_result = response.json()

print(f"Order cancelled: {cancel_result['status']}")
```

## Portfolio Analytics

### Get Portfolio Performance

```python
endpoint = "/api/v1/analytics/portfolio"
params = {
    "accountId": "000000004045701",
    "period": "3M"
}

response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params)
analytics = response.json()

print("Portfolio Analytics:")
print(f"Total Return: {analytics['analytics']['totalReturn']}%")
print(f"Volatility: {analytics['analytics']['volatility']}")
print(f"Sharpe Ratio: {analytics['analytics']['sharpeRatio']}")
```

## Business Asset Management

### List Businesses

```python
endpoint = "/businesses"

response = requests.get(f"{base_url}{endpoint}", headers=headers)
businesses = response.json()

print("Businesses:")
for business in businesses:
    print(f"- {business['name']}: {business['id']}")
```

### Create New Asset

```python
endpoint = "/assets"

asset_data = {
    "name": "Office Building",
    "type": "Real Estate",
    "value": 2500000.00,
    "businessId": "business_123"
}

response = requests.post(f"{base_url}{endpoint}", headers=headers, json=asset_data)
new_asset = response.json()

print(f"Asset created: {new_asset['id']}")
```

## Telemetry Processing

### Process Telemetry Event

```python
endpoint = "/telemetry"

telemetry_data = {
    "eventType": "app_launch",
    "timestamp": "2024-01-15T10:30:00Z",
    "userId": "user_123",
    "deviceInfo": {
        "os": "Windows",
        "version": "11",
        "deviceType": "desktop"
    },
    "metadata": {
        "sessionId": "sess_456",
        "location": "New York"
    }
}

response = requests.post(f"{base_url}{endpoint}", headers=headers, json=telemetry_data)
result = response.json()

print(f"Telemetry processed: {result['status']}")
```

## Error Handling

Always check response status and handle errors:

```python
response = requests.get(f"{base_url}/api/v1/accounts", headers=headers)

if response.status_code == 200:
    accounts = response.json()
    print("Success!")
elif response.status_code == 401:
    print("Authentication failed. Check your token.")
elif response.status_code == 429:
    print("Rate limit exceeded. Wait before retrying.")
else:
    error = response.json()
    print(f"Error: {error['error']['message']}")
```

## Rate Limiting

The API has rate limits. Monitor headers:

```python
response = requests.get(f"{base_url}/api/v1/accounts", headers=headers)

print(f"Rate Limit: {response.headers.get('X-RateLimit-Limit')}")
print(f"Remaining: {response.headers.get('X-RateLimit-Remaining')}")
print(f"Reset: {response.headers.get('X-RateLimit-Reset')}")
```

## Best Practices

1. **Reuse Access Tokens**: Tokens are valid for 1 hour. Cache and reuse them.
2. **Handle Rate Limits**: Implement exponential backoff for retries.
3. **Validate Data**: Always validate input data before sending requests.
4. **Use HTTPS**: Never send credentials over HTTP.
5. **Monitor Usage**: Track your API usage to avoid hitting limits.
6. **Error Recovery**: Implement proper error handling and recovery logic.

## Complete Example Script

```python
import requests
import time

class JPMorganAPIClient:
    def __init__(self, client_id, client_secret, base_url="http://localhost:8000"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url
        self.token = None
        self.token_expires = 0

    def get_token(self):
        if time.time() < self.token_expires:
            return self.token

        token_url = "https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token"
        response = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        })

        token_data = response.json()
        self.token = token_data["access_token"]
        self.token_expires = time.time() + 3500  # 1 hour - 100 seconds buffer
        return self.token

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Content-Type": "application/json"
        }

    def get_accounts(self):
        endpoint = "/api/v1/accounts"
        response = requests.get(f"{self.base_url}{endpoint}", headers=self.get_headers())
        return response.json()

    def get_quotes(self, symbols):
        endpoint = "/api/v1/market/quotes"
        params = {"symbols": ",".join(symbols)}
        response = requests.get(f"{self.base_url}{endpoint}", headers=self.get_headers(), params=params)
        return response.json()

# Usage
client = JPMorganAPIClient("your_client_id", "your_client_secret")
accounts = client.get_accounts()
quotes = client.get_quotes(["AAPL", "GOOGL"])
```

## Next Steps

- Explore the [API Documentation](../api.md) for complete endpoint details
- Check the [Deployment Guide](../DEPLOYMENT.md) for production setup
- Review [Security Best Practices](../security.md) for secure implementation

---

**Last Updated**: November 2024
**Version**: 1.0.0
