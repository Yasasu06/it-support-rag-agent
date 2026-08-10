"""
Centralized, reusable failure-handling utilities for the IT Support RAG Agent.

Provides consistent patterns instead of scattered ad-hoc try/excepts:
  - with_retry:          retry transient failures (API timeouts) with a delay
  - safe_fallback:       swallow any exception and return a safe fallback value
  - validate_user_input: reject empty / oversized input before the pipeline
"""

import logging
import functools
import time
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class ServiceUnavailableError(Exception):
    """Raised when an external service (OpenAI, ChromaDB, ServiceNow) is
    unavailable after retries."""
    pass


def with_retry(
    max_attempts: int = 2,
    delay_seconds: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator that retries a function on failure with a short delay. Use for
    transient failures like API timeouts - NOT for logic errors.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} failed "
                        f"(attempt {attempt}/"
                        f"{max_attempts}): {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay_seconds)
            raise ServiceUnavailableError(
                f"{func.__name__} failed after "
                f"{max_attempts} attempts: "
                f"{last_exception}"
            )
        return wrapper
    return decorator


def safe_fallback(
    fallback_value: Any = None,
    log_message: str = "Operation failed, using fallback"
):
    """
    Decorator that catches ANY exception and returns a safe fallback value
    instead of crashing. Use this for non-critical paths where a degraded
    response is better than a crash (e.g. live eval scoring, feedback logging -
    things that should never break the main user-facing answer).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{log_message}: {e}")
                return fallback_value
        return wrapper
    return decorator


def validate_user_input(
    question: str,
    max_length: int = 2000
) -> tuple[bool, Optional[str]]:
    """
    Validates user input before it enters the pipeline.
    Returns (is_valid, error_message).
    """
    if not question or not question.strip():
        return False, "Question cannot be empty."

    if len(question) > max_length:
        return False, (
            f"Question is too long ({len(question)} "
            f"characters). Please limit to "
            f"{max_length} characters."
        )

    return True, None
