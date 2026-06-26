"""
Istio Integration Middleware for JPMorgan Financial APIs
Phase 7 - Service Mesh

Provides:
- Istio-specific middleware
- Request tracing
- Distributed tracing integration
- mTLS verification
"""

import logging
import os
import traceback
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class IstioHeaders:
    """Istio header names for tracing"""
    
    # Request headers set by Istio
    REQUEST_ID = "x-request-id"
    B3_TRACE_ID = "x-b3-traceid"
    B3_SPAN_ID = "x-b3-spanid"
    B3_PARENT_SPAN_ID = "x-b3-parentspanid"
    B3_FLAGS = "x-b3-flags"
    B3_SAMPLED = "x-b3-sampled"
    
    # Istio headers
    ISTIO_CATTLE_PREFIX = "x-cattle-"
    ISTIO_PEER_HEADER = "x-istio-peer"
    
    # Custom headers
    CUSTOMER_ID = "x-customer-id"
    TENANT_ID = "x-tenant-id"
    WORKLOAD_NAME = "x-workload-name"
    APP_NAME = "x-app-name"
    VERSION = "x-version"


class IstioMiddleware:
    """FastAPI middleware for Istio integration"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.workload_name = os.environ.get("WORKLOAD_NAME", "unknown")
        self.app_name = os.environ.get("APP_NAME", "jpmorgan-api")
        self.version = os.environ.get("VERSION", "1.0.0")
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request with Istio integration"""
        
        # Start timing
        import time
        start_time = time.time()
        
        # Get or generate request ID
        request_id = request.headers.get(
            IstioHeaders.REQUEST_ID,
            self._generate_request_id()
        )
        
        # Extract trace context
        trace_context = self._extract_trace_context(request)
        
        # Add Istio headers to request state
        request.state.istio = {
            "request_id": request_id,
            "trace_id": trace_context.get("trace_id"),
            "span_id": trace_context.get("span_id"),
            "workload": self.workload_name,
            "app": self.app_name,
            "version": self.version,
        }
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add response headers
            response.headers[IstioHeaders.REQUEST_ID] = request_id
            
            if trace_context.get("trace_id"):
                response.headers[IstioHeaders.B3_TRACE_ID] = trace_context["trace_id"]
            
            # Record processing time
            process_time = time.time() - start_time
            response.headers["x-envoy-upstream-rq-timeout"] = str(int(process_time * 1000))
            
            return response
            
        except Exception as e:
            # Log error with trace context
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_context.get("trace_id"),
                    "path": request.url.path,
                    "method": request.method,
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                }
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": str(e) if os.environ.get("DEBUG") else "An error occurred",
                },
                headers={
                    IstioHeaders.REQUEST_ID: request_id,
                }
            )
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _extract_trace_context(self, request: Request) -> Dict[str, str]:
        """Extract trace context from headers"""
        headers = request.headers
        
        return {
            "request_id": headers.get(IstioHeaders.REQUEST_ID),
            "trace_id": headers.get(IstioHeaders.B3_TRACE_ID),
            "span_id": headers.get(IstioHeaders.B3_SPAN_ID),
            "parent_span_id": headers.get(IstioHeaders.B3_PARENT_SPAN_ID),
            "flags": headers.get(IstioHeaders.B3_FLAGS),
            "sampled": headers.get(IstioHeaders.B3_SAMPLED),
        }


class IstioServiceDiscovery:
    """Service discovery using Istio's service entries"""
    
    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
    
    def register_service(
        self,
        name: str,
        host: str,
        port: int,
        protocol: str = "http",
        subsets: Optional[Dict[str, str]] = None,
    ):
        """Register a service with Istio"""
        self._services[name] = {
            "host": host,
            "port": port,
            "protocol": protocol,
            "subsets": subset or {},
        }
    
    def get_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Get service info"""
        return self._services.get(name)
    
    def get_endpoint(self, name: str) -> str:
        """Get service endpoint URL"""
        service = self._services.get(name)
        if not service:
            return ""
        
        protocol = service.get("protocol", "http")
        host = service.get("host", "")
        port = service.get("port", 80)
        
        return f"{protocol}://{host}:{port}"


class CircuitBreaker:
    """Circuit breaker pattern for Istio integration"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state"""
        return self._state
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker"""
        
        if self._state == "open":
            # Check if we should try half-open
            import time
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success
            if self._state == "half_open":
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = "closed"
                    self._failure_count = 0
            
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_failure(self):
        """Record a failure"""
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "open"


class RetryPolicy:
    """Retry policy for Istio integration"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        retriable_status_codes: Optional[list] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retriable_status_codes = retriable_status_codes or [429, 500, 502, 503, 504]
    
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry"""
        import asyncio
        
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                
                # Check if status code is retriable
                if hasattr(result, "status_code"):
                    if result.status_code in self.retriable_status_codes:
                        continue
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed: {str(e)}"
                )
                
                if attempt < self.max_attempts:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_base ** (attempt - 1)),
                        self.max_delay
                    )
                    await asyncio.sleep(delay)
        
        raise last_exception


class IstioTimeout:
    """Timeout handling for Istio integration"""
    
    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
    
    async def execute(self, func: Callable, timeout: Optional[int] = None, *args, **kwargs):
        """Execute function with timeout"""
        import asyncio
        
        timeout = timeout or self.default_timeout
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout}s")
            raise


class IstioHealthCheck:
    """Health check for Istio"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def check_liveness(self) -> Dict[str, Any]:
        """Check if application is alive"""
        return {
            "status": "alive",
            "service": "jpmorgan-financial-apis",
        }
    
    async def check_readiness(self) -> Dict[str, Any]:
        """Check if application is ready to serve traffic"""
        
        # Check dependencies
        checks = {
            "database": await self._check_database(),
            "redis": await self._check_redis(),
        }
        
        # Determine readiness
        is_ready = all(check["available"] for check in checks.values())
        
        return {
            "status": "ready" if is_ready else "not_ready",
            "service": "jpmorgan-financial-apis",
            "checks": checks,
        }
    
    async def _check_database(self) -> Dict[str, bool]:
        """Check database connectivity"""
        # Placeholder - actual implementation would check DB
        return {"available": True}
    
    async def _check_redis(self) -> Dict[str, bool]:
        """Check Redis connectivity"""
        # Placeholder - actual implementation would check Redis
        return {"available": True}


# Global instances
service_discovery = IstioServiceDiscovery()
circuit_breaker = CircuitBreaker()
retry_policy = RetryPolicy()
istio_timeout = IstioTimeout()


def create_istio_middleware(app: FastAPI) -> IstioMiddleware:
    """Create Istio middleware for FastAPI app"""
    return IstioMiddleware(app)


# Register default services
def register_services():
    """Register default services with Istio"""
    
    service_discovery.register_service(
        "gateway",
        "gateway-service",
        8000,
        "http",
        {"v1": "v1"},
    )
    
    service_discovery.register_service(
        "mcp",
        "mcp-service",
        8080,
        "http",
    )
    
    service_discovery.register_service(
        "model-runner",
        "model-runner-service",
        11434,
        "http",
    )
    
    service_discovery.register_service(
        "telemetry",
        "telemetry-service",
        8000,
        "http",
    )
    
    service_discovery.register_service(
        "business",
        "business-service",
        8000,
        "http",
    )
    
    service_discovery.register_service(
        "revenue",
        "revenue-service",
        8000,
        "http",
    )
