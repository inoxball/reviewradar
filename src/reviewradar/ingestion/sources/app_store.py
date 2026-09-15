"""Apple App Store reviews via the public iTunes customer-reviews RSS feed.

Provider characteristics that shape this implementation:

* The feed is partitioned by storefront country and mixes review languages.
* It serves at most 10 pages of 50 reviews (500 per storefront); page 11 is an
  HTTP 400. For a high-volume app like Duolingo in the US that is roughly three days
  of history, so this source must run frequently to avoid coverage gaps.
* ``updated`` is the time of the latest edit, so edited reviews resurface at the top.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from datetime import UTC, datetime
from typing import Any, cast

import httpx

from reviewradar.domain import FetchedReview, Market, ReviewPage, Store, pseudonymize_author
from reviewradar.ingestion.retry import RetryPolicy, is_transient_http_error

logger = logging.getLogger(__name__)

FEED_URL = (
    "https://itunes.apple.com/{country}/rss/customerreviews"
    "/page={page}/id={app_id}/sortby=mostrecent/json"
)
MAX_PAGES = 10


class AppStoreSource:
    """Fetches App Store reviews for one storefront at a time."""

    store = Store.APP_STORE

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        retry_policy: RetryPolicy,
        page_delay_seconds: float = 0.0,
    ) -> None:
        self._client = client
        self._retry_policy = retry_policy
        self._page_delay_seconds = page_delay_seconds

    def partition_key(self, market: Market) -> str:
        # One feed per storefront, regardless of the language requested.
        return market.country

    async def fetch_pages(self, external_app_id: str, market: Market) -> AsyncGenerator[ReviewPage]:
        for page_number in range(1, MAX_PAGES + 1):
            if page_number > 1:
                await asyncio.sleep(self._page_delay_seconds)

            payload = await self._get_feed_page(external_app_id, market.country, page_number)
            reviews = parse_feed(payload, country=market.country)
            if not reviews:
                return
            yield ReviewPage(reviews=reviews, hit_provider_limit=page_number == MAX_PAGES)

    async def _get_feed_page(self, app_id: str, country: str, page: int) -> Mapping[str, Any]:
        url = FEED_URL.format(country=country, page=page, app_id=app_id)
        async for attempt in self._retry_policy.retrying(is_transient_http_error):
            with attempt:
                response = await self._client.get(url)
                response.raise_for_status()
                return cast(Mapping[str, Any], response.json())
        raise AssertionError("unreachable: tenacity re-raises after the final attempt")


def parse_feed(payload: Mapping[str, Any], *, country: str) -> list[FetchedReview]:
    """Convert one feed page into reviews, skipping (and logging) malformed entries."""
    entries = payload.get("feed", {}).get("entry", [])
    if isinstance(entries, Mapping):  # a single-entry feed is serialized as an object
        entries = [entries]

    reviews: list[FetchedReview] = []
    for entry in entries:
        if "im:rating" not in entry:  # legacy feeds lead with an app-metadata entry
            continue
        try:
            reviews.append(_parse_entry(entry, country))
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "skipping malformed App Store entry id=%s",
                _optional_label(entry, "id"),
                exc_info=True,
            )
    return reviews


def _parse_entry(entry: Mapping[str, Any], country: str) -> FetchedReview:
    author_uri = _optional_label(entry.get("author", {}), "uri")
    helpful_votes = _optional_label(entry, "im:voteSum")
    return FetchedReview(
        external_id=_label(entry, "id"),
        rating=int(_label(entry, "im:rating")),
        title=_optional_label(entry, "title"),
        body=_label(entry, "content"),
        reviewed_at=datetime.fromisoformat(_label(entry, "updated")).astimezone(UTC),
        country=country,
        language=None,  # the feed mixes languages; detection happens during enrichment
        author_hash=pseudonymize_author(Store.APP_STORE, author_uri) if author_uri else None,
        app_version=_optional_label(entry, "im:version"),
        helpful_count=int(helpful_votes) if helpful_votes is not None else None,
        raw={key: value for key, value in entry.items() if key != "author"},
    )


def _label(node: Mapping[str, Any], key: str) -> str:
    return str(node[key]["label"])


def _optional_label(node: Mapping[str, Any], key: str) -> str | None:
    child = node.get(key)
    label = child.get("label") if isinstance(child, Mapping) else None
    return str(label) if label not in (None, "") else None
