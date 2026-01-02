# Environment Configuration Guide

This document describes all environment variables required for the NestJS backend application.

## Required Environment Variables

Create a `.env` file in the `nestjs-backend` directory with the following variables:

```bash
# ==============================================
# APPLICATION CONFIGURATION
# ==============================================
NODE_ENV=development
PORT=4000

# ==============================================
# DATABASE CONFIGURATION
# ==============================================
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_database_password
DB_NAME=jpmorgan_financial_db
DB_POOL_SIZE=10
DB_CONNECTION_TIMEOUT=30000

# ==============================================
# JWT CONFIGURATION
# ==============================================
JWT_SECRET=your_super_secret_jwt_key_change_this_in_production
JWT_EXPIRATION=1h

# ==============================================
# API CONFIGURATION
# ==============================================
API_PREFIX=api
API_VERSION=v1

# ==============================================
# RATE LIMITING
# ==============================================
THROTTLE_TTL=60
THROTTLE_LIMIT=10

# ==============================================
# CORS CONFIGURATION
# ==============================================
CORS_ORIGIN=*

# ==============================================
# LOGGING
# ==============================================
LOG_LEVEL=info

# ==============================================
# JPMORGAN OAUTH2 CONFIGURATION
# ==============================================
# Get these from JPMorgan Developer Portal
JPM_CLIENT_ID=your_jpmorgan_client_id
JPM_CLIENT_SECRET=your_jpmorgan_client_secret

# JPMorgan API URLs (Sandbox)
JPM_TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token
JPM_SCOPE=jpm:payments:sandbox
JPM_API_BASE_URL=https://api-sandbox.payments.jpmorgan.com
JPM_BALANCES_URL=https://api-sandbox.payments.jpmorgan.com/tsapi/v1/accounts/balances
JPM_ACCOUNTS_URL=https://api-sandbox.payments.jpmorgan.com/tsapi/v1/accounts
JPM_TRANSACTIONS_URL=https://api-sandbox.payments.jpmorgan.com/tsapi/v1/transactions

# JPMorgan API URLs (Production)
# JPM_TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/access_token
# JPM_SCOPE=jpm:payments:production
# JPM_API_BASE_URL=https://api.payments.jpmorgan.com
# JPM_BALANCES_URL=https://api.payments.jpmorgan.com/tsapi/v1/accounts/balances
# JPM_ACCOUNTS_URL=https://api.payments.jpmorgan.com/tsapi/v1/accounts
# JPM_TRANSACTIONS_URL=https://api.payments.jpmorgan.com/tsapi/v1/transactions

# ==============================================
# DASHBOARD API KEYS (Role-Based Access Control)
# ==============================================
# Generate secure random keys for production
# Example: openssl rand -hex 32

# Admin API Key (full access)
DASHBOARD_ADMIN_API_KEY=admin_key_replace_with_secure_random_string_in_production

# Viewer API Key (read-only access)
DASHBOARD_VIEWER_API_KEY=viewer_key_replace_with_secure_random_string_in_production
```

## Environment Variable Descriptions

### Application Configuration

- **NODE_ENV**: Application environment (`development`, `production`, `test`, `staging`)
- **PORT**: Port number for the application server (default: 4000)

### Database Configuration

- **DB_HOST**: PostgreSQL database host
- **DB_PORT**: PostgreSQL database port (default: 5432)
- **DB_USER**: Database username
- **DB_PASSWORD**: Database password
- **DB_NAME**: Database name
- **DB_POOL_SIZE**: Maximum number of database connections in the pool (default: 10)
- **DB_CONNECTION_TIMEOUT**: Database connection timeout in milliseconds (default: 30000)

### JWT Configuration

- **JWT_SECRET**: Secret key for JWT token signing (use a strong random string)
- **JWT_EXPIRATION**: JWT token expiration time (e.g., '1h', '7d', '30m')

### API Configuration

- **API_PREFIX**: Global API prefix (default: 'api')
- **API_VERSION**: API version (default: 'v1')

### Rate Limiting

- **THROTTLE_TTL**: Time-to-live for rate limiting in seconds (default: 60)
- **THROTTLE_LIMIT**: Maximum number of requests per TTL (default: 10)

### CORS Configuration

- **CORS_ORIGIN**: Allowed CORS origins (use '*' for development, specific domains for production)

### Logging

- **LOG_LEVEL**: Logging level (`error`, `warn`, `info`, `debug`, `verbose`)

### JPMorgan OAuth2 Configuration

- **JPM_CLIENT_ID**: Your JPMorgan API client ID (from Developer Portal)
- **JPM_CLIENT_SECRET**: Your JPMorgan API client secret (from Developer Portal)
- **JPM_TOKEN_URL**: OAuth2 token endpoint URL
- **JPM_SCOPE**: OAuth2 scope for API access
- **JPM_API_BASE_URL**: Base URL for JPMorgan API
- **JPM_BALANCES_URL**: Endpoint for fetching account balances
- **JPM_ACCOUNTS_URL**: Endpoint for fetching account information
- **JPM_TRANSACTIONS_URL**: Endpoint for fetching transactions

### Dashboard API Keys

- **DASHBOARD_ADMIN_API_KEY**: API key for admin access (full permissions)
- **DASHBOARD_VIEWER_API_KEY**: API key for viewer access (read-only)

## Generating Secure API Keys

For production, generate secure random API keys:

```bash
# Using OpenSSL
openssl rand -hex 32

# Using Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"

# Using Python
python -c "import secrets; print(secrets.token_hex(32))"
```

## API Key Usage

### In Grafana

When configuring Grafana JSON API datasource:

1. Go to Configuration → Data Sources → Add data source → JSON API
2. Set URL: `http://localhost:4000/api/jpmorgan`
3. Add custom HTTP header:
   - Header: `x-api-key`
   - Value: `your_viewer_or_admin_api_key`

### In cURL

```bash
# Using viewer key
curl -H "x-api-key: your_viewer_api_key" \
  http://localhost:4000/api/jpmorgan/balances

# Using admin key
curl -H "x-api-key: your_admin_api_key" \
  http://localhost:4000/api/jpmorgan/balances
```

### In Postman

1. Open request
2. Go to Headers tab
3. Add header:
   - Key: `x-api-key`
   - Value: `your_api_key`

## Role-Based Access Control

### Admin Role
- Full access to all endpoints
- Can view and modify data
- Access to admin-only endpoints (future)

### Viewer Role
- Read-only access to data endpoints
- Can view balances, accounts, transactions
- Cannot modify data

## Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use strong, random API keys** in production
3. **Rotate API keys regularly** (every 90 days recommended)
4. **Use environment-specific keys** (different keys for dev/staging/prod)
5. **Store secrets securely** (use AWS Secrets Manager, Azure Key Vault, etc.)
6. **Enable HTTPS** in production
7. **Restrict CORS origins** to specific domains in production
8. **Monitor API key usage** and set up alerts for suspicious activity

## Validation

The application validates all environment variables on startup. If any required variables are missing or invalid, the application will fail to start with a detailed error message.

## Example Production Configuration

```bash
NODE_ENV=production
PORT=4000

# Use managed database service
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_USER=prod_user
DB_PASSWORD=strong_random_password
DB_NAME=jpmorgan_prod_db

# Strong JWT secret
JWT_SECRET=use_a_very_long_random_string_here_at_least_32_characters

# Production JPMorgan credentials
JPM_CLIENT_ID=prod_client_id_from_jpmorgan
JPM_CLIENT_SECRET=prod_client_secret_from_jpmorgan
JPM_TOKEN_URL=https://id.payments.jpmorgan.com/am/oauth2/access_token
JPM_SCOPE=jpm:payments:production
JPM_API_BASE_URL=https://api.payments.jpmorgan.com

# Secure API keys
DASHBOARD_ADMIN_API_KEY=generated_with_openssl_rand_hex_32
DASHBOARD_VIEWER_API_KEY=another_generated_secure_key

# Restrict CORS
CORS_ORIGIN=https://yourdomain.com,https://grafana.yourdomain.com

# Production logging
LOG_LEVEL=warn
```

## Troubleshooting

### Application won't start

Check that all required variables are set:
- Database credentials
- JWT secret
- JPMorgan credentials (can be empty for testing without JPM API)

### API key authentication fails

- Verify the API key is correctly set in the request header
- Check that the key matches one defined in your `.env` file
- Ensure the header name is exactly `x-api-key` (case-sensitive)

### JPMorgan API calls fail

- Verify your JPMorgan credentials are correct
- Check that you're using the correct environment (sandbox vs production)
- Ensure your JPMorgan account has the necessary permissions
- Check the OAuth2 token is being acquired successfully (check logs)

## Support

For issues or questions:
1. Check the application logs for detailed error messages
2. Verify all environment variables are correctly set
3. Review the JPMorgan API documentation
4. Contact your JPMorgan API representative for credential issues
