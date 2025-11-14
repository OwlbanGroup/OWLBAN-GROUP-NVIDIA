# Getting Started Guide - JPMorgan Financial APIs

## Welcome

Welcome to the JPMorgan Financial APIs! This guide will help you get started with accessing and using our comprehensive financial data services.

## Prerequisites

Before you begin, ensure you have:

- **API Credentials**: Client ID and Client Secret from JPMorgan
- **Development Environment**:
  - Python 3.8+
  - pip package manager
  - Git
- **Basic Knowledge**:
  - RESTful APIs
  - OAuth2 authentication
  - Financial data concepts

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jpmorgan/jpmorgan-financial-apis.git
cd jpmorgan-financial-apis
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Add your API credentials:

```bash
TOKEN_CLIENT_ID=your_client_id_here
TOKEN_CLIENT_SECRET=your_client_secret_here
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 4. Run the Application

```bash
# Development mode
python app.py

# Or using Docker
docker-compose up -d
```

### 5. Test Your Setup

```bash
# Get an access token
curl -X POST "https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -u "$TOKEN_CLIENT_ID:$TOKEN_CLIENT_SECRET" \
  -d "grant_type=client_credentials"

# Test API access
curl -H "Authorization: Bearer <your_token>" \
  http://localhost:8000/api/v1/accounts
```

## Understanding the API

### Authentication Flow

The JPMorgan Financial APIs use OAuth2 Client Credentials flow:

1. **Request Token**: Exchange client credentials for access token
2. **Use Token**: Include token in API requests
3. **Token Refresh**: Tokens expire; implement automatic refresh

```python
import requests

# Step 1: Get access token
token_response = requests.post(
    'https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token',
    auth=(CLIENT_ID, CLIENT_SECRET),
    data={'grant_type': 'client_credentials'}
)

access_token = token_response.json()['access_token']

# Step 2: Use token in API calls
headers = {'Authorization': f'Bearer {access_token}'}
response = requests.get('https://api.jpmorgan.com/v1/accounts', headers=headers)
```

### Core Concepts

#### Accounts
Financial accounts represent banking relationships:
- **Account ID**: Unique identifier for each account
- **Account Type**: CHECKING, SAVINGS, INVESTMENT, etc.
- **Currency**: USD, EUR, GBP, etc.
- **Balance**: Current and available amounts

#### Transactions
Financial transactions track money movement:
- **Transaction ID**: Unique transaction identifier
- **Amount**: Transaction amount and currency
- **Date**: When transaction occurred
- **Description**: Transaction details
- **Category**: Categorized transaction type

#### Market Data
Real-time and historical market information:
- **Symbols**: Stock tickers (AAPL, GOOGL, MSFT)
- **Prices**: Bid, ask, last trade prices
- **Volume**: Trading volume
- **Time & Sales**: Individual trade records

## Basic API Usage

### Python SDK Example

```python
from jpmorgan_api import JPMorganAPI

# Initialize client
api = JPMorganAPI(
    client_id='your_client_id',
    client_secret='your_client_secret'
)

# Get account list
accounts = api.get_accounts()
print(f"Found {len(accounts)} accounts")

# Get specific account details
account = api.get_account('000000004045701')
print(f"Account balance: ${account['balance']}")

# Get market quotes
quotes = api.get_quotes(['AAPL', 'GOOGL'])
for symbol, data in quotes.items():
    print(f"{symbol}: ${data['price']}")
```

### REST API Examples

#### Get Accounts

```bash
curl -X GET "https://api.jpmorgan.com/v1/accounts" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "accounts": [
    {
      "accountId": "000000004045701",
      "accountName": "Primary Checking",
      "accountType": "CHECKING",
      "currency": "USD",
      "balance": 33253003.18,
      "availableBalance": 33253003.18
    }
  ]
}
```

#### Get Account Transactions

```bash
curl -X GET "https://api.jpmorgan.com/v1/accounts/000000004045701/transactions?limit=10" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json"
```

#### Get Market Data

```bash
curl -X GET "https://api.jpmorgan.com/v1/market/quotes?symbols=AAPL,GOOGL" \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json"
```

## Advanced Usage

### Pagination

Large result sets are paginated:

```python
# Get first page
response = api.get_transactions(account_id, limit=50)
transactions = response['transactions']

# Get next page
if response['pagination']['hasNext']:
    next_page = api.get_transactions(
        account_id,
        limit=50,
        offset=response['pagination']['nextOffset']
    )
```

### Filtering and Sorting

```python
# Filter transactions by date
transactions = api.get_transactions(
    account_id,
    start_date='2024-01-01',
    end_date='2024-01-31',
    transaction_type='DEBIT'
)

# Sort market data
quotes = api.get_quotes(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    sort_by='price',
    sort_order='desc'
)
```

### Error Handling

```python
try:
    accounts = api.get_accounts()
except AuthenticationError:
    print("Invalid credentials - check client ID and secret")
    # Refresh token or re-authenticate
except RateLimitError:
    print("Rate limit exceeded - implement backoff")
    time.sleep(60)  # Wait before retrying
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
    # Handle specific error codes
```

### Rate Limiting

API requests are subject to rate limits:

- **Authenticated Requests**: 1000 per minute
- **Market Data**: 500 per minute
- **Trading Operations**: 100 per minute

```python
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Implement retry with backoff
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=1
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# Use session for all requests
response = http.get(url, headers=headers)
```

## Development Best Practices

### Environment Management

```bash
# Use virtual environments
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration Management

```python
# Use environment variables for sensitive data
import os

class Config:
    CLIENT_ID = os.getenv('JPMORGAN_CLIENT_ID')
    CLIENT_SECRET = os.getenv('JPMORGAN_CLIENT_SECRET')
    BASE_URL = os.getenv('JPMORGAN_BASE_URL', 'https://api.jpmorgan.com')

    @classmethod
    def validate(cls):
        if not cls.CLIENT_ID or not cls.CLIENT_SECRET:
            raise ValueError("JPMorgan credentials not configured")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log API calls
logger.info(f"Making API call to {endpoint}")
try:
    response = make_api_call(endpoint, data)
    logger.info(f"API call successful: {response.status_code}")
except Exception as e:
    logger.error(f"API call failed: {e}")
```

### Testing

```python
import unittest
from unittest.mock import Mock, patch

class TestJPMorganAPI(unittest.TestCase):
    def setUp(self):
        self.api = JPMorganAPI('test_id', 'test_secret')

    @patch('jpmorgan_api.requests.post')
    def test_get_token_success(self, mock_post):
        # Mock successful token response
        mock_response = Mock()
        mock_response.json.return_value = {'access_token': 'test_token'}
        mock_post.return_value = mock_response

        token = self.api.get_token()
        self.assertEqual(token, 'test_token')

    @patch('jpmorgan_api.requests.get')
    def test_get_accounts_with_token(self, mock_get):
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {'accounts': []}
        mock_get.return_value = mock_response

        accounts = self.api.get_accounts()
        self.assertIsInstance(accounts, list)
```

## Troubleshooting

### Common Issues

#### Authentication Failures

**Problem**: Getting 401 Unauthorized errors

**Solutions**:
1. Verify client ID and secret are correct
2. Check token hasn't expired (tokens last 1 hour)
3. Ensure proper OAuth2 flow implementation
4. Validate token format in Authorization header

#### Connection Timeouts

**Problem**: Requests timing out

**Solutions**:
1. Check network connectivity
2. Implement retry logic with exponential backoff
3. Increase timeout values for slow operations
4. Verify API endpoints are accessible

#### Rate Limit Exceeded

**Problem**: Getting 429 Too Many Requests

**Solutions**:
1. Implement rate limiting in your application
2. Use caching to reduce API calls
3. Implement request queuing
4. Monitor your request patterns

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Enable requests debugging
import http.client
http.client.HTTPConnection.debuglevel = 1
```

### Health Checks

```bash
# Check API health
curl https://api.jpmorgan.com/health

# Check your application health
curl http://localhost:8000/health

# Validate database connectivity
curl http://localhost:8000/health/database
```

## Next Steps

### Explore Advanced Features

- **Webhooks**: Real-time notifications for account changes
- **Bulk Operations**: Process multiple items efficiently
- **Analytics**: Advanced financial analytics and reporting
- **Trading APIs**: Programmatic trading capabilities

### Join the Community

- **Documentation**: [docs.jpmorgan.com](https://docs.jpmorgan.com)
- **Developer Forum**: [community.jpmorgan.com](https://community.jpmorgan.com)
- **Support**: [support.jpmorgan.com](https://support.jpmorgan.com)

### Stay Updated

- Follow our [changelog](https://github.com/jpmorgan/jpmorgan-financial-apis/blob/main/CHANGELOG.md)
- Subscribe to [release notifications](https://github.com/jpmorgan/jpmorgan-financial-apis/releases)
- Watch for API updates and new features

---

**Need Help?**
- Check the [troubleshooting guide](troubleshooting.md)
- Review the [API documentation](api.md)
- Contact [developer support](https://support.jpmorgan.com)

Happy coding! 🚀
