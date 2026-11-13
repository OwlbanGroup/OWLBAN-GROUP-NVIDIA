"""
Tests for OWLBAN GROUP middleware components.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from middleware.rate_limiter import TokenBucket, RateLimiterMiddleware
from middleware.security_headers import SecurityHeadersMiddleware
from middleware.csrf import (
    generate_csrf_token,
    validate_csrf_token,
    CSRFProtectionMiddleware,
)


class TestTokenBucket:
    """Test token bucket rate limiter."""

    def test_consume_success(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        assert bucket.consume() is True

    def test_consume_exhausted(self):
        bucket = TokenBucket(rate=0.1, capacity=1)
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refill_over_time(self):
        import time
        bucket = TokenBucket(rate=100.0, capacity=1)
        assert bucket.consume() is True
        assert bucket.consume() is False
        time.sleep(0.02)  # Allow refill at 100 tokens/sec
        assert bucket.consume() is True


class TestRateLimiterMiddleware:
    """Test rate limiter middleware configuration."""

    def test_default_rate_and_burst(self):
        mw = RateLimiterMiddleware.__new__(RateLimiterMiddleware)
        mw.__init__(None)  # app=None for test
        assert mw.rate == 10.0
        assert mw.burst == 50

    def test_custom_rate_and_burst(self):
        mw = RateLimiterMiddleware.__new__(RateLimiterMiddleware)
        mw.__init__(None, rate=5.0, burst=10)
        assert mw.rate == 5.0
        assert mw.burst == 10


class TestSecurityHeadersMiddleware:
    """Test security headers middleware."""

    def test_security_headers_present(self):
        mw = SecurityHeadersMiddleware.__new__(SecurityHeadersMiddleware)

        class FakeResponse:
            def __init__(self):
                self.headers = {}

        async def fake_call_next(request):
            resp = FakeResponse()
            return resp

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                mw.dispatch(None, fake_call_next)
            )
        finally:
            loop.close()
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Strict-Transport-Security"].startswith("max-age=")
        assert "Content-Security-Policy" in response.headers
        assert "X-Request-Id" in response.headers


class TestCSRFUtils:
    """Test CSRF utility functions."""

    def test_generate_and_validate(self):
        token = generate_csrf_token()
        assert token and len(token) >= 32
        assert validate_csrf_token(token, token) is True
        assert validate_csrf_token(token, "attacker") is False
        assert validate_csrf_token(None, token) is False
        assert validate_csrf_token(token, None) is False

    def test_middleware_exemptions(self):
        mw = CSRFProtectionMiddleware.__new__(CSRFProtectionMiddleware)
        mw.SKIPPED_PATHS = ("/auth/", "/metrics", "/health", "/status")
        assert mw._is_exempt("/auth/login")
        assert mw._is_exempt("/status")
        assert not mw._is_exempt("/api/payment")
