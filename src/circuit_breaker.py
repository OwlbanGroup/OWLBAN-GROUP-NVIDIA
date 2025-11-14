"""
Circuit Breaker Pattern Implementation for JPMorgan Financial APIs
"""
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from .logger import telemetry_logger

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """
    Circuit Breaker implementation with configurable failure thresholds,
    timeout, and recovery mechanisms.
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Exception = Exception,
                 success_threshold: int = 3,
                 timeout: float = 10.0,
                 name: str = "default"):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type to count as failure
            success_threshold: Number of successes needed in half-open state
            timeout: Timeout for individual calls
            name: Name for logging and monitoring
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.name = name

        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._lock = threading.RLock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function"""
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function call

        Raises:
            CircuitBreakerOpenException: If circuit is open
        """
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._success_count = 0
                    telemetry_logger.get_logger().info(
                        f"Circuit breaker '{self.name}' entering HALF_OPEN state"
                    )
                else:
                    telemetry_logger.get_logger().warning(
                        f"Circuit breaker '{self.name}' is OPEN, failing fast"
                    )
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is open"
                    )

        try:
            # Execute with timeout
            result = self._call_with_timeout(func, *args, **kwargs)

            with self._lock:
                self._on_success()

            return result

        except self.expected_exception as e:
            with self._lock:
                self._on_failure()
            raise e
        except Exception as e:
            # Re-raise unexpected exceptions without counting as circuit failure
            raise e

    def _call_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Call to {func.__name__} timed out after {self.timeout}s")

        # Set up signal handler for timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._reset()
        # In CLOSED state, just log success
        telemetry_logger.get_logger().info(
            f"Circuit breaker '{self.name}' call succeeded"
        )

    def _on_failure(self):
        """Handle failed call"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery, go back to open
            self._state = CircuitBreakerState.OPEN
            telemetry_logger.get_logger().error(
                f"Circuit breaker '{self.name}' failed in HALF_OPEN, returning to OPEN"
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN
            telemetry_logger.get_logger().error(
                f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
            )

    def _reset(self):
        """Reset circuit breaker to closed state"""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        telemetry_logger.get_logger().info(
            f"Circuit breaker '{self.name}' reset to CLOSED state"
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count"""
        return self._failure_count

    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure_time': self._last_failure_time,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout,
            'success_threshold': self.success_threshold,
            'timeout': self.timeout
        }

# Global circuit breaker instances for different services
jpmorgan_api_circuit_breaker = CircuitBreaker(
    name="jpmorgan_api",
    failure_threshold=5,
    recovery_timeout=60.0,
    timeout=30.0
)

database_circuit_breaker = CircuitBreaker(
    name="database",
    failure_threshold=3,
    recovery_timeout=30.0,
    timeout=10.0
)

redis_circuit_breaker = CircuitBreaker(
    name="redis",
    failure_threshold=3,
    recovery_timeout=30.0,
    timeout=5.0
)

storage_circuit_breaker = CircuitBreaker(
    name="storage",
    failure_threshold=5,
    recovery_timeout=120.0,
    timeout=60.0
)
