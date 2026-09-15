"""Test doubles and builders shared across test modules."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from reviewradar.catalog import AppSpec, MarketSpec
from reviewradar.domain import FetchedReview, Market, ReviewPage, Store

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)

LISTING_IDS = {Store.GOOGLE_PLAY: "com.duolingo", Store.APP_STORE: "570060128"}


def make_review(
    external_id: str,
    *,
    age: timedelta = timedelta(hours=1),
    body: str = "Great app for daily practice",
    rating: int = 5,
    country: str = "us",
) -> FetchedReview:
    return FetchedReview(
        external_id=external_id,
        rating=rating,
        body=body,
        reviewed_at=NOW - age,
        country=country,
    )


def page(*reviews: FetchedReview, hit_provider_limit: bool = False) -> ReviewPage:
    return ReviewPage(reviews=reviews, hit_provider_limit=hit_provider_limit)


def duolingo(*stores: Store, markets: Sequence[tuple[str, str]] = (("us", "en"),)) -> AppSpec:
    return AppSpec(
        slug="duolingo",
        name="Duolingo",
        listings={store: LISTING_IDS[store] for store in stores},
        markets=tuple(MarketSpec(country=c, language=lang) for c, lang in markets),
    )


class FakeSource:
    """In-memory :class:`ReviewSource` serving scripted pages per partition.

    Records which pages were served so tests can assert on pagination behaviour, and
    can raise an error after serving a partition's pages to simulate a mid-run outage.
    """

    def __init__(
        self,
        store: Store,
        pages: Mapping[str, Sequence[ReviewPage]] | None = None,
        *,
        by_language: bool = True,
        errors: Mapping[str, Exception] | None = None,
    ) -> None:
        self.store = store
        self.served: list[tuple[str, int]] = []
        self._pages = dict(pages or {})
        self._by_language = by_language
        self._errors = dict(errors or {})

    def partition_key(self, market: Market) -> str:
        return str(market) if self._by_language else market.country

    async def fetch_pages(self, external_app_id: str, market: Market) -> AsyncGenerator[ReviewPage]:
        key = self.partition_key(market)
        for index, scripted_page in enumerate(self._pages.get(key, ())):
            self.served.append((key, index))
            yield scripted_page
        if key in self._errors:
            raise self._errors[key]
