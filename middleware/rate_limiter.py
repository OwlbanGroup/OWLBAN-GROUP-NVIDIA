"""In-memory token-bucket and ASGI rate-limiter middleware."""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class TokenBucket:
    def __init__(self, rate: float = 10.0, capacity: int = 50) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)
        self.updated_at = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: float = 10.0, burst: int = 50) -> None:
        super().__init__(app)
        self.rate = rate
        self.burst = burst
        self._buckets = defaultdict(lambda: TokenBucket(rate=self.rate, capacity=self.burst))

    async def dispatch(self, request, call_next):
        client = request.client.host if request.client else "unknown"
        if not self._buckets[client].consume():
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        return await call_next(request)
