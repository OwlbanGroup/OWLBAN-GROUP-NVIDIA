"""
CSRF Protection for OWLBAN GROUP web applications.

Implements the double-submit cookie pattern (stateless, no server-side
session store required). A random token is placed in a cookie AND delivered to
the client; state-changing requests must echo the token in the
`X-CSRF-Token` header (or `csrfmiddlewaretoken` form field). Since an attacker
cannot read/forge the cookie value from an untrusted origin, a mismatch proves
cross-origin forgery.

For the JWT/bearer API (auth via Authorization header), CSRF is not strictly
required because the browser does not auto-attach bearer tokens. This module
guards the cookie/session-authenticated web surfaces (OWLBAN GROUP site,
OSCAR BROOME, BLACKBOX AI) included in the platform.
"""

import hmac
import secrets
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Base64url-safe alphabet for cookie/token values (no '=' padding needed)
_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


def generate_csrf_token() -> str:
    """Return a cryptographically random CSRF token (32 bytes, urlsafe)."""
    return secrets.token_urlsafe(32)


def validate_csrf_token(cookie_value, header_value) -> bool:
    """Return True if the cookie and submitted token match (timing-safe)."""
    if not cookie_value or not header_value:
        return False
    return hmac.compare_digest(cookie_value, header_value)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    Enforces double-submit CSRF protection for state-changing HTTP methods.

    - Sets the `csrf_token` cookie on responses when absent.
    - For unsafe methods (POST/PUT/PATCH/DELETE), requires the submitted
      token (X-CSRF-Token header or `csrf_token` form field) to match the
      cookie value.
    - Skips safe methods (GET/HEAD/OPTIONS) and, by default, the API auth
      routes that authenticate via the stateless Authorization header.
    """

    COOKIE_NAME = "csrf_token"
    HEADER_NAME = "X-CSRF-Token"
    UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    # Bearer-token-authenticated API routes do not need cookie-CSRF.
    SKIPPED_PATHS = ("/auth/", "/prometheus/", "/metrics", "/health", "/status")

    def __init__(self, app, exempt_paths=None):
        super().__init__(app)
        if exempt_paths:
            self.SKIPPED_PATHS = tuple(exempt_paths + list(self.SKIPPED_PATHS))

    async def dispatch(self, request, call_next):
        method = request.method.upper()

        # Provide a CSRF cookie for all responses lacking one (double-submit).
        response = None
        csrf_cookie = request.cookies.get(self.COOKIE_NAME)

        if method in self.UNSAFE_METHODS and not self._is_exempt(request.url.path):
            submitted = request.headers.get(self.HEADER_NAME)
            if submitted is None:
                # Allow form-encoded csrf_token field as a fallback.
                form = await self._read_form(request)
                submitted = form.get("csrf_token")
            if csrf_cookie is None or not validate_csrf_token(csrf_cookie, submitted):
                logger.warning(
                    "CSRF validation failed for %s %s (origin=%s)",
                    method, request.url.path, request.headers.get("origin"),
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed"},
                )

        response = await call_next(request)
        if csrf_cookie is None:
            response.set_cookie(
                self.COOKIE_NAME,
                generate_csrf_token(),
                httponly=False,      # read by frontend JS for the header
                samesite="lax",      # CSRF-safe while still allowing top-nav nav
                secure=request.url.scheme == "https",
                max_age=3600,
            )
        return response

    def _is_exempt(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.SKIPPED_PATHS)

    async def _read_form(self, request):
        try:
            import json
            ct = request.headers.get("content-type", "")
            if "application/json" in ct:
                body = await request.body()
                data = json.loads(body or b"{}")
                if isinstance(data, dict):
                    return data
            elif "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
                form = await request.form()
                return dict(form)
        except Exception:
            logger.exception("Failed to read form for CSRF token")
        return {}