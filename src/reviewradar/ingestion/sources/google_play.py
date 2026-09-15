"""Google Play reviews via the ``google-play-scraper`` library.

Provider characteristics that shape this implementation:

* Reviews are served per language; ``country`` does not change the result set
  (verified: ``us``/``en`` and ``gb``/``en`` return identical reviews). Partitions are
  therefore per language, and reviews carry no country rather than a guessed one.
* Pagination uses an opaque continuation token; a token whose inner value is
  ``None`` marks the end of the available history.
* The library is synchronous and returns *naive local-time* datetimes (it calls
  ``datetime.fromtimestamp``). Calls run in a worker thread and timestamps are
  converted to UTC at this boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from google_play_scraper import Sort
from google_play_scraper import reviews as scrape_reviews
from google_play_scraper.exceptions import ExtraHTTPError, NotFoundError

from reviewradar.domain import FetchedReview, Market, ReviewPage, Store, pseudonymize_author
from reviewradar.ingestion.retry import RetryPolicy

logger = logging.getLogger(__name__)

PageFetcher = Callable[[str, Market, int, Any], tuple[list[dict[str, Any]], Any]]
"""Blocking page fetch: ``(app_id, market, page_size, continuation_token) -> (records, token)``."""

_PERSONAL_FIELDS = frozenset({"userName", "userImage"})


def scrape_reviews_page(
    app_id: str, market: Market, page_size: int, continuation_token: Any
) -> tuple[list[dict[str, Any]], Any]:
    """Fetch one page of the newest reviews through google-play-scraper."""
    return scrape_reviews(  # type: ignore[no-any-return]
        app_id,
        lang=market.language,
        country=market.country,
        sort=Sort.NEWEST,
        count=page_size,
        continuation_token=continuation_token,
    )


class GooglePlaySource:
    """Fetches Google Play reviews for one language partition at a time."""

    store = Store.GOOGLE_PLAY

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy,
        page_size: int = 200,
        page_delay_seconds: float = 0.0,
        fetch_page: PageFetcher = scrape_reviews_page,
    ) -> None:
        self._retry_policy = retry_policy
        self._page_size = page_size
        self._page_delay_seconds = page_delay_seconds
        self._fetch_page = fetch_page

    def partition_key(self, market: Market) -> str:
        # One stream per language; markets that differ only by country share it.
        return market.language

    async def fetch_pages(self, external_app_id: str, market: Market) -> AsyncGenerator[ReviewPage]:
        token: Any = None
        while True:
            records, token = await self._fetch_with_retry(external_app_id, market, token)
            if not records:
                return
            yield ReviewPage(reviews=parse_records(records, market))
            if getattr(token, "token", None) is None:
                return
            await asyncio.sleep(self._page_delay_seconds)

    async def _fetch_with_retry(
        self, app_id: str, market: Market, token: Any
    ) -> tuple[list[dict[str, Any]], Any]:
        async for attempt in self._retry_policy.retrying(is_transient_scraper_error):
            with attempt:
                return await asyncio.to_thread(
                    self._fetch_page, app_id, market, self._page_size, token
                )
        raise AssertionError("unreachable: tenacity re-raises after the final attempt")


def is_transient_scraper_error(exc: BaseException) -> bool:
    """Network failures, unexpected HTTP statuses and throttling pages are worth retrying."""
    if isinstance(exc, NotFoundError):
        return False
    return isinstance(exc, OSError | ExtraHTTPError | json.JSONDecodeError)


def parse_records(records: Iterable[Mapping[str, Any]], market: Market) -> list[FetchedReview]:
    """Convert scraper records into reviews, skipping (and logging) malformed ones."""
    reviews: list[FetchedReview] = []
    for record in records:
        try:
            reviews.append(_parse_record(record, market))
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "skipping malformed Google Play review id=%s", record.get("reviewId"), exc_info=True
            )
    return reviews


def _parse_record(record: Mapping[str, Any], market: Market) -> FetchedReview:
    author = record.get("userName")
    replied_at = record.get("repliedAt")
    return FetchedReview(
        external_id=str(record["reviewId"]),
        rating=int(record["score"]),
        body=record.get("content") or "",
        reviewed_at=_to_utc(record["at"]),
        country=None,  # not exposed by Google Play
        language=market.language,
        author_hash=pseudonymize_author(Store.GOOGLE_PLAY, author) if author else None,
        app_version=record.get("reviewCreatedVersion") or record.get("appVersion"),
        helpful_count=record.get("thumbsUpCount"),
        developer_reply=record.get("replyContent"),
        developer_replied_at=_to_utc(replied_at) if replied_at is not None else None,
        raw=_json_safe(record),
    )


def _to_utc(value: datetime) -> datetime:
    """Convert the scraper's naive local-time datetimes to aware UTC values."""
    if not isinstance(value, datetime):
        raise TypeError(f"expected datetime, got {type(value).__name__}")
    return value.astimezone(UTC)  # naive values are interpreted as local time


def _json_safe(record: Mapping[str, Any]) -> dict[str, Any]:
    """Raw payload for reprocessing, minus personal data, with datetimes as ISO strings."""
    return {
        key: _to_utc(value).isoformat() if isinstance(value, datetime) else value
        for key, value in record.items()
        if key not in _PERSONAL_FIELDS
    }
