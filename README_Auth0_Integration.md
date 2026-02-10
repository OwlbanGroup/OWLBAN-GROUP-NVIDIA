# Auth0 Integration for JPMorgan Financial APIs

This document describes the Auth0 authentication integration added to the JPMorgan Financial APIs system.

## Overview

The application now supports Auth0 authentication as the primary authentication mechanism, replacing the previous in-memory user management system. Auth0 provides enterprise-grade authentication, authorization, and user management capabilities.

## Features Added

### 1. Auth0 Authentication Module (`src/auth0_auth.py`)
- JWT token verification using Auth0's public keys
- User information retrieval from Auth0
- Permission-based access control
- Integration with Flask application

### 2. Configuration Updates (`config.py`)
- Added Auth0-specific configuration variables
- Environment variable support for Auth0 settings
- JWKS URL and issuer configuration

### 3. API Endpoints

#### Authentication Endpoints
- `GET /auth/login` - Returns Auth0 Universal Login URL
- `GET /auth/callback` - Handles Auth0 callback (for web applications)
- `GET /auth/userinfo` - Returns current authenticated user information
- `GET /auth/logout` - Returns Auth0 logout URL

#### Protected Endpoints
- `GET /businesses` - Now requires Auth0 authentication
- `POST /businesses` - Now requires Auth0 authentication

## Configuration

### Environment Variables

Add the following environment variables to your `.env` file:

```bash
# Auth0 Configuration
AUTH0_DOMAIN=your-auth0-domain.auth0.com
AUTH0_CLIENT_ID=your-auth0-client-id
AUTH0_CLIENT_SECRET=your-auth0-client-secret
AUTH0_AUDIENCE=https://your-api-identifier
```

### Auth0 Setup

1. **Create an Auth0 Application**:
   - Go to your Auth0 Dashboard
   - Create a new Application (Regular Web Application or SPA)
   - Note the Domain, Client ID, and Client Secret

2. **Configure API**:
   - Create an API in Auth0
   - Set the Identifier (Audience)
   - Configure permissions/scopes as needed

3. **Update Allowed Origins**:
   - Add your application domains to Allowed Origins in Auth0

## Usage

### Authentication Flow

1. **Get Login URL**:
   ```bash
   GET /auth/login
   ```

2. **Authenticate User**:
   - Redirect user to the returned login URL
   - User authenticates with Auth0
   - Auth0 redirects back with authorization code

3. **Exchange Code for Tokens**:
   - Use the authorization code to get access/id tokens
   - Include access token in API requests

4. **Make Authenticated Requests**:
   ```bash
   Authorization: Bearer <access_token>
   GET /businesses
   ```

### Example API Usage

```python
import requests

# Get login URL
response = requests.get('http://localhost:5000/auth/login')
login_url = response.json()['login_url']

# After authentication, use access token
headers = {'Authorization': f'Bearer {access_token}'}
response = requests.get('http://localhost:5000/businesses', headers=headers)
```

## Security Features

### JWT Verification
- Tokens are verified using Auth0's public keys from JWKS endpoint
- Validates issuer, audience, and expiration
- Supports RS256 algorithm

### Permission-Based Access
- Decorator `@require_permission('permission_name')` for fine-grained access control
- User permissions stored in JWT token

### Rate Limiting
- Maintains existing rate limiting for authenticated endpoints
- Conditional limits based on authentication status

## Migration Notes

### Legacy Endpoints
- Legacy user registration/login endpoints (`/user/register`, `/user/login`) remain available for backward compatibility
- In-memory user store still functional for testing (`TESTING=1`)

### Authentication Decorators
- `@auth0_required`: Requires valid Auth0 JWT token
- `@require_auth`: Legacy authentication (still available)
- `@token_auth_required`: Legacy token authentication

## Testing

### With Auth0
1. Set up Auth0 application and API
2. Configure environment variables
3. Obtain access token from Auth0
4. Use token in API requests

### Testing Mode
- Set `TESTING=1` to bypass authentication
- Legacy authentication remains functional

## Dependencies

Added dependency:
```
auth0-python==4.7.1
```

Install with:
```bash
pip install -r requirements.txt
```

## Error Handling

### Common Errors
- `401 Unauthorized`: Invalid or missing token
- `403 Forbidden`: Insufficient permissions
- `500 Internal Server Error`: Configuration issues

### Logging
- Authentication failures logged with context
- Token verification errors tracked
- User activity audited

## Best Practices

1. **Token Storage**: Store tokens securely (HttpOnly cookies for web apps)
2. **Token Refresh**: Implement token refresh logic for long-lived sessions
3. **Permissions**: Use Auth0 roles and permissions for access control
4. **Monitoring**: Monitor authentication success/failure rates
5. **Security**: Rotate client secrets regularly

## Troubleshooting

### Configuration Issues
- Verify Auth0 domain format (should end with `.auth0.com`)
- Check client credentials are correct
- Ensure API audience matches Auth0 API configuration

### Token Issues
- Verify token hasn't expired
- Check token audience matches API audience
- Ensure correct Authorization header format

### Permission Issues
- Verify user has required permissions in Auth0
- Check JWT contains permissions claim
- Ensure permission names match between Auth0 and application

## Support

For Auth0-specific issues, refer to:
- [Auth0 Documentation](https://auth0.com/docs)
- [Auth0 Community](https://community.auth0.com)

For application-specific issues, check the application logs and configuration.
