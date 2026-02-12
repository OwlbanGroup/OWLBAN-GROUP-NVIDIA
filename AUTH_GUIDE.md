# JPMorgan Financial APIs - Authentication Guide

## Overview

The JPMorgan Financial APIs use a comprehensive authentication system that supports both JWT-based authentication and legacy username/password authentication. This guide covers all authentication methods, token management, and security best practices.

## Authentication Methods

### 1. JWT Authentication (Recommended)

The primary authentication method uses JSON Web Tokens (JWT) with role-based access control (RBAC).

#### User Roles and Permissions

| Role | Description | Permissions |
|------|-------------|-------------|
| **ADMIN** | Full system access | All permissions including user management |
| **MANAGER** | Business management | Business, assets, telemetry, audit (read-only) |
| **USER** | Standard user | Read access to business, assets, telemetry |
| **AUDITOR** | Audit access only | Read access to audit logs and metrics |

#### Permission Matrix

| Permission | ADMIN | MANAGER | USER | AUDITOR |
|------------|-------|---------|------|---------|
| read:users | ✅ | ❌ | ❌ | ❌ |
| write:users | ✅ | ❌ | ❌ | ❌ |
| read:businesses | ✅ | ✅ | ✅ | ❌ |
| write:businesses | ✅ | ✅ | ❌ | ❌ |
| read:assets | ✅ | ✅ | ✅ | ❌ |
| write:assets | ✅ | ✅ | ❌ | ❌ |
| read:telemetry | ✅ | ✅ | ✅ | ❌ |
| write:telemetry | ✅ | ✅ | ❌ | ❌ |
| read:audit | ✅ | ❌ | ❌ | ✅ |
| write:audit | ✅ | ❌ | ❌ | ❌ |
| read:revenue | ✅ | ✅ | ✅ | ❌ |
| write:revenue | ✅ | ✅ | ❌ | ❌ |
| read:private_bank | ✅ | ✅ | ✅ | ❌ |
| write:private_bank | ✅ | ✅ | ❌ | ❌ |
| read:ml | ✅ | ❌ | ❌ | ❌ |
| write:ml | ✅ | ❌ | ❌ | ❌ |
| read:metrics | ✅ | ✅ | ✅ | ✅ |
| write:metrics | ✅ | ❌ | ❌ | ❌ |

### 2. Legacy Authentication

For backward compatibility, username/password authentication is available through `/user/register` and `/user/login` endpoints.

## Getting Started with Authentication

### Step 1: Register a User Account

#### Using Legacy Authentication

```bash
# Register a new user
curl -X POST http://localhost:8000/user/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "securepassword123"
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "User created successfully"
}
```

#### Using JWT Authentication

The JWT authentication requires database-backed user management. Contact your system administrator to create accounts.

### Step 2: Login and Obtain Token

#### Legacy Login

```bash
# Login to get authentication token
curl -X POST http://localhost:8000/user/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "securepassword123"
  }'
```

**Response:**
```json
{
  "status": "success",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "username": "johndoe",
  "created_at": "2024-01-01T10:00:00Z",
  "token_created_at": "2024-01-01T10:00:00Z"
}
```

#### JWT Login

JWT authentication is handled through the database-backed auth service. The token contains user information and permissions.

### Step 3: Use Token in API Requests

Include the token in the `Authorization` header for all authenticated requests:

```bash
# Example: Get user profile
curl -X GET http://localhost:8000/user/profile \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

**Response:**
```json
{
  "status": "success",
  "username": "johndoe",
  "created_at": "2024-01-01T10:00:00Z",
  "token_created_at": "2024-01-01T10:00:00Z"
}
```

## API Endpoints

### Authentication Endpoints

#### Legacy Authentication

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/user/register` | Register new user | 5/minute |
| POST | `/user/login` | User login | 10/minute |
| GET | `/user/profile` | Get user profile | 10/minute |

#### JWT Authentication

JWT authentication is handled automatically by the `@require_auth` decorator on protected endpoints.

### Protected Endpoints

All business operations require authentication:

```bash
# List businesses (requires authentication)
curl -X GET http://localhost:8000/businesses \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Create business (requires authentication)
curl -X POST http://localhost:8000/businesses \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Example Corp",
    "description": "Financial services company",
    "industry": "Finance"
  }'
```

## Token Management

### Token Expiration

- **Legacy tokens**: Simple tokens (no expiration)
- **JWT tokens**: 24-hour expiration by default

### Token Refresh

When a JWT token expires, you need to re-authenticate:

```bash
# Re-login to get new token
curl -X POST http://localhost:8000/user/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "securepassword123"
  }'
```

### Token Validation

Tokens are validated on each request. Invalid or expired tokens return:

```json
{
  "error": "Invalid or expired token",
  "status": "error"
}
```

## Security Features

### Rate Limiting

- Registration: 5 requests per minute
- Login: 10 requests per minute
- Profile access: 10 requests per minute
- Other endpoints: Varies by operation

### Password Requirements

- Minimum 8 characters
- Maximum 128 characters
- Stored with bcrypt hashing

### Session Management

- Automatic logout on token expiration
- Secure token storage recommended
- No concurrent session limits

## Error Handling

### Common Authentication Errors

#### Invalid Credentials
```json
{
  "detail": "Invalid username or password"
}
```

#### Missing Token
```json
{
  "detail": "Authentication required",
  "headers": {"WWW-Authenticate": "Bearer"}
}
```

#### Insufficient Permissions
```json
{
  "error": "Insufficient permissions",
  "status": "error"
}
```

#### Rate Limit Exceeded
```json
{
  "detail": "Too Many Requests"
}
```

## Best Practices

### 1. Token Storage

**Never store tokens in:**
- Local storage (insecure)
- Cookies without security flags
- Plain text files

**Recommended approaches:**
- Secure HTTP-only cookies
- Memory-based storage (for SPAs)
- Encrypted local storage with proper flags

### 2. Token Refresh Strategy

```javascript
// Example token refresh logic
async function makeAuthenticatedRequest(url, options = {}) {
  let token = getStoredToken();

  if (isTokenExpired(token)) {
    token = await refreshToken();
    storeToken(token);
  }

  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });
}
```

### 3. Error Handling

```javascript
// Example error handling
async function apiCall(endpoint) {
  try {
    const response = await fetch(endpoint, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (response.status === 401) {
      // Token expired or invalid
      redirectToLogin();
      return;
    }

    if (response.status === 403) {
      // Insufficient permissions
      showPermissionError();
      return;
    }

    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    showNetworkError();
  }
}
```

### 4. Security Headers

Always validate these security headers in responses:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`

## Troubleshooting

### Common Issues

#### 1. "Invalid or expired token"

**Cause:** Token has expired or is malformed
**Solution:** Re-authenticate to get a new token

#### 2. "Authentication required"

**Cause:** Missing or malformed Authorization header
**Solution:** Ensure header format: `Authorization: Bearer <token>`

#### 3. "Insufficient permissions"

**Cause:** User role doesn't have required permissions
**Solution:** Contact administrator to upgrade role or check endpoint requirements

#### 4. Rate Limiting

**Cause:** Too many requests in time window
**Solution:** Implement exponential backoff retry logic

### Debug Mode

Enable debug logging to troubleshoot authentication issues:

```bash
export FLASK_ENV=development
export LOG_LEVEL=DEBUG
python app_async.py
```

### Health Check

Verify authentication system health:

```bash
# Check if auth endpoints are responding
curl -f http://localhost:8000/health

# Test authentication flow
curl -X POST http://localhost:8000/user/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass"}'
```

## Integration Examples

### JavaScript (Fetch API)

```javascript
class ApiClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('auth_token');
  }

  async login(username, password) {
    const response = await fetch(`${this.baseUrl}/user/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });

    if (response.ok) {
      const data = await response.json();
      this.token = data.token;
      localStorage.setItem('auth_token', this.token);
      return data;
    } else {
      throw new Error('Login failed');
    }
  }

  async request(endpoint, options = {}) {
    const headers = { ...options.headers };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers
    });

    if (response.status === 401) {
      // Token expired, redirect to login
      this.logout();
      window.location.href = '/login';
      return;
    }

    return response;
  }

  logout() {
    this.token = null;
    localStorage.removeItem('auth_token');
  }
}
```

### Python (Requests)

```python
import requests
from typing import Optional

class JPMorganAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.session = requests.Session()

    def login(self, username: str, password: str) -> dict:
        """Login and store token"""
        response = self.session.post(
            f"{self.base_url}/user/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()

        data = response.json()
        self.token = data.get("token")
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}"
        })
        return data

    def register(self, username: str, password: str) -> dict:
        """Register new user"""
        response = self.session.post(
            f"{self.base_url}/user/register",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        return response.json()

    def get_businesses(self) -> dict:
        """Get list of businesses"""
        response = self.session.get(f"{self.base_url}/businesses")
        response.raise_for_status()
        return response.json()

    def create_business(self, business_data: dict) -> dict:
        """Create a new business"""
        response = self.session.post(
            f"{self.base_url}/businesses",
            json=business_data
        )
        response.raise_for_status()
        return response.json()
```

## Support

For authentication issues or questions:

1. Check this guide first
2. Review API logs for error details
3. Contact system administrator
4. Create issue in project repository

## Version History

- **v1.0**: Initial authentication system with JWT and legacy support
- **v1.1**: Added role-based access control
- **v1.2**: Enhanced security with rate limiting and improved error handling

---

**Note:** This authentication system is designed for enterprise-grade security. For production deployments, ensure proper SSL/TLS configuration and regular security audits.
