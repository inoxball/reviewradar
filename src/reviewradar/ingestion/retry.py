"""Retry policy for transient provider failures (network errors, throttling, 5xx)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with jitter, shared by all sources."""

    max_attempts: int = 5
    initial_wait_seconds: float = 1.0
    max_wait_seconds: float = 30.0
    jitter_seconds: float = 1.0

    def retrying(self, is_retryable: Callable[[BaseException], bool]) -> AsyncRetrying:
        """Build a tenacity controller; use as ``async for attempt in policy.retrying(...)``."""
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(
                initial=self.initial_wait_seconds,
                max=self.max_wait_seconds,
                jitter=self.jitter_seconds,
            ),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )


def is_transient_http_error(exc: BaseException) -> bool:
    """True for failures worth retrying against an HTTP API."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUSES
    return isinstance(exc, httpx.TransportError)
