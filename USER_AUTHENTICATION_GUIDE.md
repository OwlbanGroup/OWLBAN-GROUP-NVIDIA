# OWLBAN GROUP - User Authentication Guide

## Overview

This guide covers authentication across all OWLBAN GROUP platforms:

| Platform | URL | Company |
|----------|-----|---------|
| OWLBAN GROUP | http://localhost:3000 | OWLBAN_GROUP |
| OSCAR BROOME Revenue | http://localhost:3001 | OSCAR_BROOME |
| BLACKBOX AI | http://localhost:3002 | BLACKBOX_AI |
| API Server | http://localhost:8000 | NVIDIA_INTEGRATION |
| Web Dashboard | http://localhost:8501 | All |

---

## Getting Started

### 1. Creating an Account

**Web (OWLBAN GROUP):**
1. Navigate to `http://localhost:3000/register.html`
2. Enter your email, username, and password
3. Click "Create Account"
4. Sign in with your credentials

**API:**
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","username":"yourname","password":"SecurePass1!","company":"OWLBAN_GROUP"}'
```

### 2. Logging In

**Web:** Navigate to the login page and enter your credentials.

**API:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"SecurePass1!"}'
```

Response:
```json
{
  "access_token": "eyJhbG...",
  "refresh_token": "eyJhbG...",
  "token_type": "bearer"
}
```

### 3. Using Access Tokens

Include the access token in API requests:
```bash
curl http://localhost:8000/protected-endpoint \
  -H "Authorization: Bearer eyJhbG..."
```

---

## Authentication Methods

### JWT Bearer Token (Recommended)
- Short-lived access tokens (15 min) + long-lived refresh tokens (7 days)
- Used by all web applications and API clients

### API Keys
- Generate from your profile page or via API
- Long-lived, revocable
- Used for server-to-server communication

```bash
curl -X POST http://localhost:8000/auth/api-keys \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name":"my-server-key"}'
```

Use the API key:
```bash
curl http://localhost:8000/protected \
  -H "Authorization: Bearer owlban_abc123..."
```

### HTTP Basic Auth (Legacy)
- Used for admin endpoints on the API Server
- Username/password via HTTP Basic Authentication
- Set via `API_USERNAME` and `API_PASSWORD` environment variables

---

## Password Reset

1. Click "Forgot password?" on the login page
2. Enter your email address
3. A reset token will be sent (demo mode shows token directly)
4. Enter your new password

**API:**
```bash
# Request reset
curl -X POST http://localhost:8000/auth/reset-request \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com"}'

# Reset password
curl -X POST http://localhost:8000/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{"token":"<reset-token>","new_password":"NewSecurePass1!"}'
```

---

## Password Policy

- Minimum 8 characters
- Must contain uppercase letter
- Must contain lowercase letter
- Must contain a number
- Account locks after 5 failed attempts (15 min)

---

## Role-Based Access Control (RBAC)

| Role | Permissions |
|------|------------|
| user | Standard access to own resources |
| admin | Full user management, all resources |
| executive | Read-only access to reports and analytics |
| developer | API access, development tools |
| analyst | Data analysis, read-only system access |

---

## Rate Limiting

All API endpoints are rate-limited:
- Default: 10 requests/second with burst of 50
- Configured via `RATE_LIMIT_RATE` and `RATE_LIMIT_BURST` env vars
- Exceeding limit returns HTTP 429 with `Retry-After` header

---

## Security Headers

All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: default-src 'self'`

---

## Single Sign-On (SSO)

Users registered via the Python `auth_lib` can authenticate across all platforms:
- Same credentials work for OWLBAN GROUP, OSCAR BROOME, BLACKBOX AI
- Sessions are managed independently per platform
- Centralized user store (`users.json`)

---

## Running the Servers

```powershell
# OWLBAN GROUP
cd owlbangroup.io && npm install express jsonwebtoken bcryptjs
node src/server.js

# OSCAR BROOME
cd OSCAR-BROOME-REVENUE && npm install express jsonwebtoken bcryptjs
node server_with_auth.js

# BLACKBOX AI
cd BLACKBOX-AI && npm install express jsonwebtoken bcryptjs
node src/server.js

# Python API Server
.\.venv\Scripts\python.exe -m uvicorn api_server:fastapi_app --host 0.0.0.0 --port 8000

# Streamlit Dashboard
.\.venv\Scripts\python.exe -m streamlit run web_dashboard.py --server.port 8501
```

---

## Testing

```powershell
# Run auth tests
.\.venv\Scripts\python.exe -m pytest tests/test_auth_system.py -v

# Run all tests
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Security Hardening: MFA

Multi-Factor Authentication (TOTP, RFC 6238) protects accounts with a
dependency-free stdlib implementation compatible with Google/Microsoft
authenticator apps.

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/mfa/setup` | Generate a TOTP secret + provisioning URI (scan with an authenticator) |
| POST | `/auth/mfa/enable` | Verify a code and enable MFA (`{"code":"123456"}`) |
| POST | `/auth/mfa/disable` | Verify a code and disable MFA |
| GET  | `/auth/mfa/status` | Return whether MFA is required |

**MFA login flow:**

1. `POST /auth/login` with email + password.
2. If MFA is enabled and no `mfa_code` is present, the API returns `HTTP 428`
   (`{"detail":"MFA code required"}`).
3. Re-submit login including the `mfa_code` from the authenticator app.
4. On success, JWT access/refresh tokens are returned.

**Pairing (library):**

```python
from auth_lib import TOTP, auth_manager
setup = auth_manager.setup_mfa("user@owlban.com")
print(setup["secret"])
print(setup["provisioning_uri"])  # scan with authenticator, then:
ok, msg = auth_manager.enable_mfa("user@owlban.com", "123456")
```

---

## Security Hardening: CSRF

Cookie-authenticated web surfaces use the double-submit-cookie pattern
(stateless). A `csrf_token` cookie (`samesite=lax`) is set on every response;
state-changing methods must echo it in the `X-CSRF-Token` header (or a
`csrf_token` form field). API auth routes (`/auth/*`) and `/prometheus/metrics`
are exempt because they use stateless bearer tokens.

```python
from middleware.csrf import generate_csrf_token, validate_csrf_token
token = generate_csrf_token()
assert validate_csrf_token(token, token)     # True
assert validate_csrf_token(token, "attack")  # False (timing-safe)
```
