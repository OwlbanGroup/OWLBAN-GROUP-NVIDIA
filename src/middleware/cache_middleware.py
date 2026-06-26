"""
Advanced Caching Middleware for JPMorgan Financial APIs
Phase 7 - Advanced Caching Strategies

Implements:
- Redis cache-aside pattern
- Cache invalidation policies
- Distributed locking
- API response caching
- ML prediction caching
"""

import hashlib
import json
import logging
import time
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

import redis
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class CacheConfig:
    """Cache configuration settings"""
    
    # Redis connection settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Cache TTL settings (in seconds)
    DEFAULT_TTL: int = 300  # 5 minutes
    API_RESPONSE_TTL: int = 60  # 1 minute
    ML_PREDICTION_TTL: int = 3600  # 1 hour
    USER_DATA_TTL: int = 1800  # 30 minutes
    BUSINESS_DATA_TTL: int = 600  # 10 minutes
    REVENUE_DATA_TTL: int = 300  # 5 minutes
    
    # Cache key prefixes
    API_PREFIX: str = "api:"
    ML_PREFIX: str = "ml:"
    USER_PREFIX: str = "user:"
    BUSINESS_PREFIX: str = "biz:"
    REVENUE_PREFIX: str = "rev:"
    LOCK_PREFIX: str = "lock:"
    
    # Cache settings
    MAX_CACHE_SIZE: int = 10000
    CACHE_COMPRESSION: bool = True
    COMPRESSION_THRESHOLD: int = 1024
    
    # Distributed lock settings
    LOCK_TIMEOUT: int = 30  # seconds
    LOCK_RETRY_DELAY: float = 0.2  # seconds
    LOCK_MAX_RETRIES: int = 5


class RedisCache:
    """Redis cache manager with advanced features"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._client: Optional[redis.Redis] = None
        self._lock_client: Optional[redis.Redis] = None
    
    @property
    def client(self) -> redis.Redis:
        """Lazy initialization of Redis client"""
        if self._client is None:
            self._client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                password=self.config.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
        return self._client
    
    @property
    def lock_client(self) -> redis.Redis:
        """Separate client for distributed locking"""
        if self._lock_client is None:
            self._lock_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                password=self.config.REDIS_PASSWORD,
                decode_responses=False,
            )
        return self._lock_client
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = [prefix]
        for arg in args:
            if arg is not None:
                key_parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        return ":".join(key_parts)
    
    def _compress(self, data: str) -> bytes:
        """Compress data if above threshold"""
        if self.config.CACHE_COMPRESSION and len(data) > self.config.COMPRESSION_THRESHOLD:
            import gzip
            return gzip.compress(data.encode())
        return data.encode()
    
    def _decompress(self, data: bytes) -> str:
        """Decompress data"""
        try:
            import gzip
            return gzip.decompress(data).decode()
        except Exception:
            return data.decode()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(self._decompress(value))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return self._decompress(value)
            return None
        except redis.RedisError as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        """Set value in cache with optional TTL"""
        try:
            ttl = ttl or self.config.DEFAULT_TTL
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
                encoded = self._compress(value)
            else:
                encoded = self._compress(str(value))
            
            if nx:
                return self.client.setex(key, ttl, encoded)
            else:
                self.client.setex(key, ttl, encoded)
                return True
        except redis.RedisError as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return self.client.delete(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Redis delete pattern error: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return self.client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key"""
        try:
            return self.client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis ttl error: {e}")
            return -1
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            return self.client.incrby(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis increment error: {e}")
            return 0
    
    # Distributed Locking
    
    def acquire_lock(
        self,
        lock_name: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """Acquire distributed lock"""
        timeout = timeout or self.config.LOCK_TIMEOUT
        lock_key = f"{self.config.LOCK_PREFIX}{lock_name}"
        try:
            return self.lock_client.set(
                lock_key,
                "1",
                nx=True,
                ex=timeout,
            )
        except redis.RedisError as e:
            logger.error(f"Redis lock acquire error: {e}")
            return False
    
    def release_lock(self, lock_name: str) -> bool:
        """Release distributed lock"""
        lock_key = f"{self.config.LOCK_PREFIX}{lock_name}"
        try:
            return self.lock_client.delete(lock_key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis lock release error: {e}")
            return False
    
    def acquire_lock_with_retry(
        self,
        lock_name: str,
        max_retries: Optional[int] = None,
    ) -> bool:
        """Acquire lock with retry logic"""
        max_retries = max_retries or self.config.LOCK_MAX_RETRIES
        for _ in range(max_retries):
            if self.acquire_lock(lock_name):
                return True
            time.sleep(self.config.LOCK_RETRY_DELAY)
        return False
    
    # Cache invalidation by tag
    
    def add_to_cache_tags(self, key: str, *tags: str) -> bool:
        """Add cache key to tags"""
        try:
            for tag in tags:
                tag_key = f"tag:{tag}"
                self.client.sadd(tag_key, key)
                self.client.expire(tag_key, self.config.DEFAULT_TTL)
            return True
        except redis.RedisError as e:
            logger.error(f"Redis add to tags error: {e}")
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all keys with tag"""
        try:
            tag_key = f"tag:{tag}"
            keys = self.client.smembers(tag_key)
            if keys:
                for key in keys:
                    self.client.delete(key)
                self.client.delete(tag_key)
                return len(keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Redis invalidate tag error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.client.info("stats")
            memory = self.client.info("memory")
            return {
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "memory_used": memory.get("used_memory_human", "0"),
                "connected": True,
            }
        except redis.RedisError as e:
            logger.error(f"Redis stats error: {e}")
            return {"connected": False, "error": str(e)}
    
    def ping(self) -> bool:
        """Check if Redis is available"""
        try:
            return self.client.ping()
        except redis.RedisError:
            return False
    
    def close(self):
        """Close Redis connections"""
        try:
            if self._client:
                self._client.close()
            if self._lock_client:
                self._lock_client.close()
        except Exception as e:
            logger.error(f"Redis close error: {e}")


class CacheMiddleware:
    """FastAPI middleware for caching"""
    
    def __init__(self, app: FastAPI, cache: RedisCache):
        self.app = app
        self.cache = cache
        self._excluded_paths: Set[str] = {"/health", "/metrics", "/docs", "/openapi.json"}
        self._cacheable_methods: Set[str] = {"GET", "POST"}
    
    def add_excluded_path(self, path: str):
        """Add path to exclude from caching"""
        self._excluded_paths.add(path)
    
    def is_cacheable(self, request: Request) -> bool:
        """Check if request should be cached"""
        # Check method
        if request.method not in self._cacheable_methods:
            return False
        
        # Check path
        path = request.url.path
        for excluded in self._excluded_paths:
            if path.startswith(excluded):
                return False
        
        return True
    
    def generate_request_key(self, request: Request) -> str:
        """Generate cache key from request"""
        # Create key from path and query params
        key_parts = [
            self.cache.config.API_PREFIX,
            request.url.path.lstrip("/").replace("/", ":"),
        ]
        
        # Add query parameters
        query_params = sorted(request.query_params.items())
        if query_params:
            for k, v in query_params:
                key_parts.append(f"{k}:{v}")
        
        # Hash long keys
        key = ":".join(key_parts)
        if len(key) > 200:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            key = f"{self.cache.config.API_PREFIX}hash:{key_hash}"
        
        return key
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request with caching"""
        if not self.is_cacheable(request):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self.generate_request_key(request)
        
        # Try to get from cache
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            # Update cache stats
            self.cache.increment("cache:hits")
            
            logger.debug(f"Cache hit: {cache_key}")
            return JSONResponse(content=cached_response)
        
        self.cache.increment("cache:misses")
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                # Get response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Parse JSON
                try:
                    content = json.loads(body)
                    ttl = self._determine_ttl(request)
                    self.cache.set(cache_key, content, ttl)
                    logger.debug(f"Cached: {cache_key} (TTL: {ttl}s)")
                except json.JSONDecodeError:
                    pass
                
                # Return new response
                return JSONResponse(
                    content=json.loads(body),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except Exception as e:
                logger.error(f"Cache middleware error: {e}")
        
        return response
    
    def _determine_ttl(self, request: Request) -> int:
        """Determine TTL based on endpoint"""
        path = request.url.path
        
        if "/user/" in path:
            return self.cache.config.USER_DATA_TTL
        elif "/business/" in path:
            return self.cache.config.BUSINESS_DATA_TTL
        elif "/revenue/" in path:
            return self.cache.config.REVENUE_DATA_TTL
        elif "/ai/" in path or "/ml/" in path:
            return self.cache.config.ML_PREDICTION_TTL
        else:
            return self.cache.config.API_RESPONSE_TTL


def cache_result(
    prefix: str,
    ttl: Optional[int] = None,
    key_builder: Optional[Callable] = None,
):
    """Decorator for caching function results"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = cache._generate_cache_key(prefix, *args, **kwargs)
            
            # Check cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = cache._generate_cache_key(prefix, *args, **kwargs)
            
            # Check cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def invalidate_cache(prefix: str, pattern: str = "*"):
    """Decorator to invalidate cache after function execution"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache = RedisCache()
            cache.delete_pattern(f"{prefix}{pattern}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate cache
            cache = RedisCache()
            cache.delete_pattern(f"{prefix}{pattern}")
            
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Cache instances for different use cases
api_cache = RedisCache()
ml_cache = RedisCache()
user_cache = RedisCache()
business_cache = RedisCache()


# Cache for ML predictions
class MLPredictionCache:
    """Specialized cache for ML model predictions"""
    
    def __init__(self):
        self.cache = RedisCache()
    
    def get_prediction(self, model_name: str, input_hash: str) -> Optional[Dict]:
        """Get cached prediction"""
        key = f"{self.cache.config.ML_PREFIX}{model_name}:{input_hash}"
        return self.cache.get(key)
    
    def cache_prediction(
        self,
        model_name: str,
        input_hash: str,
        prediction: Dict,
    ) -> bool:
        """Cache prediction"""
        key = f"{self.cache.config.ML_PREFIX}{model_name}:{input_hash}"
        return self.cache.set(key, prediction, self.cache.config.ML_PREDICTION_TTL)
    
    def invalidate_model(self, model_name: str) -> int:
        """Invalidate all predictions for a model"""
        return self.cache.delete_pattern(
            f"{self.cache.config.ML_PREFIX}{model_name}:*"
        )


# Global instances
ml_prediction_cache = MLPredictionCache()
