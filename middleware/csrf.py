"""CSRF token helpers and middleware."""

from __future__ import annotations

import hmac
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def validate_csrf_token(expected: str | None, supplied: str | None) -> bool:
    if not expected or not supplied:
        return False
    return hmac.compare_digest(expected, supplied)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    COOKIE_NAME = "csrf_token"
    HEADER_NAME = "X-CSRF-Token"
    UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    SKIPPED_PATHS: tuple[str, ...] = (
        "/auth/",
        "/inference",
        "/revenue/",
        "/rl/",
        "/quantum/",
        "/catalog/",
        "/gpu/",
        "/logs",
        "/metrics",
        "/status",
        "/health",
        "/prometheus/",
    )

    def _is_exempt(self, path: str) -> bool:
        return any(path == prefix or path.startswith(prefix) for prefix in self.SKIPPED_PATHS)

    async def dispatch(self, request, call_next):
        if request.method not in self.UNSAFE_METHODS or self._is_exempt(request.url.path):
            return await call_next(request)

        cookie_token = request.cookies.get(self.COOKIE_NAME)
        header_token = request.headers.get(self.HEADER_NAME)
        if not validate_csrf_token(cookie_token, header_token):
            return JSONResponse({"detail": "CSRF token missing or invalid"}, status_code=403)
        return await call_next(request)
