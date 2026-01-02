# API Key Authentication & Role-Based Access Control - Implementation Guide

## Overview

This document describes the API key authentication and role-based access control (RBAC) system implemented for the JPMorgan Financial APIs backend.

## Architecture

### Components

1. **Roles Enum** (`src/auth/roles.enum.ts`)
   - Defines available roles: `ADMIN`, `VIEWER`

2. **API Key Configuration** (`src/auth/api-key-roles.config.ts`)
   - Maps API keys to roles
   - Loaded from environment variables

3. **Auth Decorator** (`src/auth/auth.decorator.ts`)
   - Metadata decorator for specifying required roles

4. **API Key Guard** (`src/auth/api-key.guard.ts`)
   - Validates API keys
   - Enforces role-based access control

5. **Protected Controllers**
   - JPMorgan controller with role-based endpoints

## Roles

### Admin Role
- **Access Level**: Full access
- **Permissions**:
  - Read all data (balances, accounts, transactions)
  - Write operations (future: create payments, modify settings)
  - Access to admin-only endpoints (future)
- **Use Case**: Internal dashboards, administrative tools

### Viewer Role
- **Access Level**: Read-only
- **Permissions**:
  - Read balances, accounts, transactions
  - Cannot modify data
  - Cannot access admin endpoints
- **Use Case**: Grafana dashboards, reporting tools, external integrations

## Implementation Details

### 1. Roles Enum

```typescript
export enum Role {
  ADMIN = 'admin',
  VIEWER = 'viewer',
}
```

### 2. API Key Configuration

```typescript
import { Role } from './roles.enum';

export const ApiKeys: Record<string, Role> = {
  [process.env.DASHBOARD_ADMIN_API_KEY || 'admin-key-placeholder']: Role.ADMIN,
  [process.env.DASHBOARD_VIEWER_API_KEY || 'viewer-key-placeholder']: Role.VIEWER,
};
```

**Environment Variables:**
```bash
DASHBOARD_ADMIN_API_KEY=your_secure_admin_key_here
DASHBOARD_VIEWER_API_KEY=your_secure_viewer_key_here
```

### 3. Auth Decorator

```typescript
import { SetMetadata } from '@nestjs/common';
import { Role } from './roles.enum';

export const ROLES_KEY = 'roles';
export const Roles = (...roles: Role[]) => SetMetadata(ROLES_KEY, roles);
```

**Usage:**
```typescript
@Get('balances')
@Roles(Role.ADMIN, Role.VIEWER)  // Both roles can access
async getBalances() { ... }

@Post('admin/settings')
@Roles(Role.ADMIN)  // Only admin can access
async updateSettings() { ... }
```

### 4. API Key Guard

```typescript
@Injectable()
export class ApiKeyGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    // 1. Get required roles from decorator
    const requiredRoles = this.reflector.get<Role[]>(ROLES_KEY, context.getHandler()) || [];

    // 2. Extract API key from request header
    const request = context.switchToHttp().getRequest();
    const apiKey = request.headers['x-api-key'];

    // 3. Validate API key exists
    if (!apiKey) {
      throw new UnauthorizedException('Missing API key');
    }

    // 4. Look up role for API key
    const role = ApiKeys[apiKey];
    if (!role) {
      throw new UnauthorizedException('Invalid API key');
    }

    // 5. Check if role is authorized
    if (requiredRoles.length && !requiredRoles.includes(role)) {
      throw new ForbiddenException('Insufficient role');
    }

    // 6. Attach role to request for logging
    request.userRole = role;

    return true;
  }
}
```

### 5. Protected Controller

```typescript
@Controller('api/jpmorgan')
@UseGuards(ApiKeyGuard)  // Apply guard to all routes
export class JpmorganController {
  @Get('balances')
  @Roles(Role.ADMIN, Role.VIEWER)  // Both roles allowed
  async getBalances() {
    // Implementation
  }

  @Post('admin/refresh')
  @Roles(Role.ADMIN)  // Only admin allowed
  async forceRefresh() {
    // Implementation
  }
}
```

## Usage Examples

### 1. cURL

```bash
# Using viewer key
curl -H "x-api-key: your_viewer_key" \
  http://localhost:4000/api/jpmorgan/balances

# Using admin key
curl -H "x-api-key: your_admin_key" \
  http://localhost:4000/api/jpmorgan/balances
```

### 2. Grafana JSON API Datasource

**Configuration:**
1. Go to Configuration → Data Sources
2. Add data source → JSON API
3. Set URL: `http://localhost:4000/api/jpmorgan`
4. Add custom HTTP header:
   - Header: `x-api-key`
   - Value: `your_viewer_key`

**Dashboard Query:**
```json
{
  "target": "balances",
  "type": "timeseries"
}
```

### 3. Postman

**Headers:**
```
x-api-key: your_api_key
Content-Type: application/json
```

**Request:**
```
GET http://localhost:4000/api/jpmorgan/balances
```

### 4. JavaScript/TypeScript

```typescript
const response = await fetch('http://localhost:4000/api/jpmorgan/balances', {
  headers: {
    'x-api-key': 'your_api_key',
    'Content-Type': 'application/json',
  },
});

const data = await response.json();
```

### 5. Python

```python
import requests

headers = {
    'x-api-key': 'your_api_key',
    'Content-Type': 'application/json'
}

response = requests.get(
    'http://localhost:4000/api/jpmorgan/balances',
    headers=headers
)

data = response.json()
```

## Security Best Practices

### 1. Generate Secure API Keys

```bash
# Using OpenSSL (recommended)
openssl rand -hex 32

# Using Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"

# Using Python
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Key Management

- **Never commit API keys** to version control
- **Use environment variables** or secrets management services
- **Rotate keys regularly** (every 90 days recommended)
- **Use different keys** for different environments (dev/staging/prod)
- **Monitor key usage** and set up alerts for suspicious activity

### 3. Production Configuration

```bash
# Strong, unique keys
DASHBOARD_ADMIN_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
DASHBOARD_VIEWER_API_KEY=z6y5x4w3v2u1t0s9r8q7p6o5n4m3l2k1j0i9h8g7f6e5d4c3b2a1

# Restrict CORS in production
CORS_ORIGIN=https://yourdomain.com,https://grafana.yourdomain.com

# Use HTTPS
# Enable rate limiting
# Set up monitoring and alerting
```

### 4. Key Storage Options

**Development:**
- `.env` file (not committed)
- Local environment variables

**Production:**
- AWS Secrets Manager
- Azure Key Vault
- Google Cloud Secret Manager
- HashiCorp Vault
- Kubernetes Secrets

## Error Responses

### Missing API Key

**Request:**
```bash
curl http://localhost:4000/api/jpmorgan/balances
```

**Response:**
```json
{
  "statusCode": 401,
  "message": "Missing API key",
  "error": "Unauthorized"
}
```

### Invalid API Key

**Request:**
```bash
curl -H "x-api-key: invalid_key" \
  http://localhost:4000/api/jpmorgan/balances
```

**Response:**
```json
{
  "statusCode": 401,
  "message": "Invalid API key",
  "error": "Unauthorized"
}
```

### Insufficient Permissions

**Request:**
```bash
# Viewer trying to access admin endpoint
curl -H "x-api-key: viewer_key" \
  http://localhost:4000/api/jpmorgan/admin/settings
```

**Response:**
```json
{
  "statusCode": 403,
  "message": "Insufficient role",
  "error": "Forbidden"
}
```

## Monitoring & Logging

### Request Logging

The guard attaches the user role to the request object:

```typescript
request.userRole = role;  // 'admin' or 'viewer'
```

This can be used for:
- Audit logging
- Usage analytics
- Security monitoring

### Example Logging Interceptor

```typescript
@Injectable()
export class AuditLogInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const { method, url, userRole } = request;

    console.log(`[${userRole}] ${method} ${url}`);

    return next.handle();
  }
}
```

## Future Enhancements

### 1. Database-Backed API Keys

Store API keys in database with additional metadata:

```typescript
interface ApiKey {
  id: string;
  key: string;  // hashed
  role: Role;
  name: string;
  createdAt: Date;
  expiresAt: Date;
  lastUsedAt: Date;
  isActive: boolean;
}
```

### 2. Multiple Roles Per Key

```typescript
interface ApiKey {
  key: string;
  roles: Role[];
  permissions: string[];
}
```

### 3. Rate Limiting Per Key

```typescript
@UseGuards(ApiKeyGuard, ThrottlerGuard)
@Throttle(100, 60)  // 100 requests per minute
export class JpmorganController { ... }
```

### 4. Key Expiration

```typescript
interface ApiKey {
  key: string;
  role: Role;
  expiresAt: Date;
}

// In guard
if (apiKey.expiresAt < new Date()) {
  throw new UnauthorizedException('API key expired');
}
```

### 5. IP Whitelisting

```typescript
interface ApiKey {
  key: string;
  role: Role;
  allowedIPs: string[];
}

// In guard
const clientIP = request.ip;
if (!apiKey.allowedIPs.includes(clientIP)) {
  throw new ForbiddenException('IP not whitelisted');
}
```

## Testing

### Unit Tests

```typescript
describe('ApiKeyGuard', () => {
  it('should allow access with valid admin key', () => {
    // Test implementation
  });

  it('should allow access with valid viewer key', () => {
    // Test implementation
  });

  it('should deny access with invalid key', () => {
    // Test implementation
  });

  it('should deny access when viewer tries admin endpoint', () => {
    // Test implementation
  });
});
```

### Integration Tests

```typescript
describe('JPMorgan API (e2e)', () => {
  it('/api/jpmorgan/balances (GET) with admin key', () => {
    return request(app.getHttpServer())
      .get('/api/jpmorgan/balances')
      .set('x-api-key', adminKey)
      .expect(200);
  });

  it('/api/jpmorgan/balances (GET) with viewer key', () => {
    return request(app.getHttpServer())
      .get('/api/jpmorgan/balances')
      .set('x-api-key', viewerKey)
      .expect(200);
  });

  it('/api/jpmorgan/balances (GET) without key', () => {
    return request(app.getHttpServer())
      .get('/api/jpmorgan/balances')
      .expect(401);
  });
});
```

## Troubleshooting

### Issue: "Missing API key"

**Cause:** No `x-api-key` header in request

**Solution:** Add header to request:
```bash
curl -H "x-api-key: your_key" http://localhost:4000/api/jpmorgan/balances
```

### Issue: "Invalid API key"

**Cause:** API key not found in configuration

**Solution:** 
1. Check `.env` file has correct keys
2. Restart application to reload environment variables
3. Verify key matches exactly (no extra spaces)

### Issue: "Insufficient role"

**Cause:** User role doesn't have permission for endpoint

**Solution:**
1. Use admin key for admin endpoints
2. Check endpoint's `@Roles()` decorator
3. Verify role assignment in `api-key-roles.config.ts`

## Summary

The API key authentication system provides:

✅ **Simple authentication** via HTTP headers  
✅ **Role-based access control** (Admin/Viewer)  
✅ **Easy integration** with Grafana and other tools  
✅ **Secure by default** with environment-based configuration  
✅ **Extensible** for future enhancements  
✅ **Production-ready** with proper error handling  

For questions or issues, refer to the main documentation or contact the development team.
