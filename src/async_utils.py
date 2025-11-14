"""
Async utilities for improved performance
"""
import asyncio
import concurrent.futures
from typing import Any, Callable, Awaitable, TypeVar, Optional
from functools import wraps
import time
from prometheus_client import Histogram
import structlog

from .logger import telemetry_logger

# Type variable for generic function types
T = TypeVar('T')

# Async execution metrics
ASYNC_EXECUTION_TIME = Histogram(
    'async_execution_duration_seconds',
    'Time spent executing async operations',
    ['operation']
)

logger = structlog.get_logger()


def async_executor(max_workers: int = 4) -> concurrent.futures.ThreadPoolExecutor:
    """Create a thread pool executor for CPU-bound tasks"""
    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="telemetry-async")


def run_in_executor(executor: Optional[concurrent.futures.ThreadPoolExecutor] = None) -> Callable:
    """Decorator to run synchronous functions in a thread pool"""
    def decorator(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            loop = asyncio.get_event_loop()
            if executor is None:
                # Use default executor
                return await loop.run_in_executor(None, func, *args, **kwargs)
            else:
                return await loop.run_in_executor(executor, func, *args, **kwargs)
        return wrapper
    return decorator


def timed_async_operation(operation_name: str) -> Callable:
    """Decorator to time async operations and record metrics"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                ASYNC_EXECUTION_TIME.labels(operation=operation_name).observe(execution_time)

                logger.info(
                    f"Async operation completed",
                    operation=operation_name,
                    execution_time=execution_time,
                    success=True
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                ASYNC_EXECUTION_TIME.labels(operation=operation_name).observe(execution_time)

                logger.error(
                    f"Async operation failed",
                    operation=operation_name,
                    execution_time=execution_time,
                    error=str(e),
                    success=False
                )
                raise
        return wrapper
    return decorator


class AsyncTaskManager:
    """Manager for async tasks with proper cleanup"""

    def __init__(self):
        self.tasks: set = set()
        self.executor = async_executor()

    def create_task(self, coro: Awaitable, name: str = None) -> asyncio.Task:
        """Create a background task"""
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def gather_with_timeout(self, *coros: Awaitable, timeout: float = 30.0) -> tuple:
        """Gather multiple coroutines with timeout"""
        try:
            return await asyncio.wait_for(asyncio.gather(*coros), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Async gather timed out after {timeout} seconds")
            raise

    async def run_in_executor(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run function in thread pool executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all pending tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Shutdown executor
        self.executor.shutdown(wait=True)


# Global task manager instance
task_manager = AsyncTaskManager()


async def process_batch_async(telemetry_data_list: list, batch_size: int = 10) -> list:
    """Process telemetry data in async batches"""
    results = []

    # Process in batches to avoid overwhelming the system
    for i in range(0, len(telemetry_data_list), batch_size):
        batch = telemetry_data_list[i:i + batch_size]

        # Process batch concurrently
        batch_tasks = [
            task_manager.run_in_executor(
                _process_single_telemetry_item,
                item
            )
            for item in batch
        ]

        try:
            batch_results = await task_manager.gather_with_timeout(*batch_tasks, timeout=10.0)
            results.extend(batch_results)
        except asyncio.TimeoutError:
            logger.error("Batch processing timed out", batch_size=len(batch))
            # Continue with next batch

    return results


def _process_single_telemetry_item(telemetry_item: dict) -> dict:
    """Process a single telemetry item (CPU-bound operation)"""
    # Simulate processing time
    time.sleep(0.01)  # 10ms processing time

    # Add processing metadata
    telemetry_item['_processed_at'] = time.time()
    telemetry_item['_processing_status'] = 'completed'

    return telemetry_item


async def parallel_database_operations(operations: list) -> list:
    """Execute multiple database operations in parallel"""
    tasks = [
        task_manager.run_in_executor(op['func'], *op.get('args', []), **op.get('kwargs', {}))
        for op in operations
    ]

    return await task_manager.gather_with_timeout(*tasks, timeout=15.0)


async def async_health_check() -> dict:
    """Perform async health checks"""
    health_checks = [
        _check_database_health(),
        _check_cache_health(),
        _check_external_services_health()
    ]

    results = await task_manager.gather_with_timeout(*health_checks, timeout=5.0)

    # Aggregate results
    overall_health = all(result['healthy'] for result in results)

    return {
        'healthy': overall_health,
        'checks': results,
        'timestamp': time.time()
    }


async def _check_database_health() -> dict:
    """Check database health"""
    try:
        # Import here to avoid circular imports
        from .database import db_manager
        healthy = db_manager.health_check()
        return {
            'service': 'database',
            'healthy': healthy,
            'response_time': 0.001  # Mock response time
        }
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return {
            'service': 'database',
            'healthy': False,
            'error': str(e)
        }


async def _check_cache_health() -> dict:
    """Check cache health"""
    try:
        # Mock cache health check
        return {
            'service': 'cache',
            'healthy': True,
            'response_time': 0.001
        }
    except Exception as e:
        logger.error("Cache health check failed", error=str(e))
        return {
            'service': 'cache',
            'healthy': False,
            'error': str(e)
        }


async def _check_external_services_health() -> dict:
    """Check external services health"""
    try:
        # Mock external services health check
        return {
            'service': 'external_services',
            'healthy': True,
            'response_time': 0.001
        }
    except Exception as e:
        logger.error("External services health check failed", error=str(e))
        return {
            'service': 'external_services',
            'healthy': False,
            'error': str(e)
        }


# Cleanup on shutdown
async def shutdown_async_resources():
    """Cleanup async resources on application shutdown"""
    await task_manager.cleanup()
