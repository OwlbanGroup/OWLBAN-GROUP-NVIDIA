"""
Shared rate limiting utilities for JPMorgan Financial APIs.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class RateLimiter:
    """Simple endpoint rate limiter configuration."""
    limit: int
    window_seconds: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "limit": self.limit,
            "window_seconds": self.window_seconds,
        }


def get_rate_limiter_for_endpoint(endpoint: str) -> RateLimiter:
    """Get a default rate limiter for an endpoint."""
    endpoint = (endpoint or "").lower()

    if endpoint in {"/health", "/status", "/metrics"}:
        return RateLimiter(limit=300, window_seconds=60)

    if endpoint.startswith("/auth"):
        return RateLimiter(limit=60, window_seconds=60)

    return RateLimiter(limit=120, window_seconds=60)
