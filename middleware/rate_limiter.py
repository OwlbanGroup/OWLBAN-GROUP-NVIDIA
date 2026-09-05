"""
Rate Limiting Middleware for OWLBAN GROUP API Server.
Provides token-bucket rate limiting per IP and per user.
"""

import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens per second
        self.capacity = capacity  # max burst
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    Limits requests per IP address using token bucket algorithm.
    Configurable via environment variables:
    - RATE_LIMIT_RATE: tokens per second (default: 10)
    - RATE_LIMIT_BURST: max burst capacity (default: 50)
    """

    def __init__(self, app, rate: float = None, burst: int = None):
        super().__init__(app)
        self.rate = rate or float(__import__('os').getenv('RATE_LIMIT_RATE', '10'))
        self.burst = burst or int(__import__('os').getenv('RATE_LIMIT_BURST', '50'))
        self.buckets = defaultdict(lambda: TokenBucket(self.rate, self.burst))

    async def dispatch(self, request, call_next):
        client_ip = request.client.host
        bucket = self.buckets[client_ip]
        if not bucket.consume():
            logger.warning(f"Rate limit exceeded for {client_ip}")
            try:
                from monitoring.auth_metrics import auth_metrics
                auth_metrics.record_rate_limit(request.url.path)
            except Exception:
                pass
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": "1"},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.burst)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        return response
