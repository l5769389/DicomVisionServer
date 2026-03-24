import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from app.core.logging import get_logger


logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Log execution time for sync and async functions."""

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.info("timer func=%s elapsed_ms=%.2f", func.__qualname__, elapsed_ms)

        return cast(F, async_wrapper)

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info("timer func=%s elapsed_ms=%.2f", func.__qualname__, elapsed_ms)

    return cast(F, sync_wrapper)
