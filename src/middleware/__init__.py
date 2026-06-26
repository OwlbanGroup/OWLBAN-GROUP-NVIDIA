"""
Phase 7 Advanced Features - Middleware Package
JPMorgan Financial APIs

This package contains advanced middleware for:
- Redis caching (cache_middleware.py)
- Istio service mesh integration (istio_middleware.py)
"""

from .cache_middleware import (
    CacheConfig,
    RedisCache,
    CacheMiddleware,
    api_cache,
    ml_cache,
    user_cache,
    business_cache,
    ml_prediction_cache,
    cache_result,
    invalidate_cache,
)

from .istio_middleware import (
    IstioHeaders,
    IstioMiddleware,
    IstioServiceDiscovery,
    CircuitBreaker,
    RetryPolicy,
    IstioTimeout,
    IstioHealthCheck,
    service_discovery,
    circuit_breaker,
    retry_policy,
    istio_timeout,
    create_istio_middleware,
    register_services,
)

__all__ = [
    # Cache
    "CacheConfig",
    "RedisCache",
    "CacheMiddleware",
    "api_cache",
    "ml_cache",
    "user_cache",
    "business_cache",
    "ml_prediction_cache",
    "cache_result",
    "invalidate_cache",
    # Istio
    "IstioHeaders",
    "IstioMiddleware",
    "IstioServiceDiscovery",
    "CircuitBreaker",
    "RetryPolicy",
    "IstioTimeout",
    "IstioHealthCheck",
    "service_discovery",
    "circuit_breaker",
    "retry_policy",
    "istio_timeout",
    "create_istio_middleware",
    "register_services",
]

__version__ = "1.0.0"
