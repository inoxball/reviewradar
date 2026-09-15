import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from reviewradar.domain import Market, ReviewPage
from reviewradar.ingestion.retry import RetryPolicy
from reviewradar.ingestion.sources.app_store import MAX_PAGES, AppStoreSource, parse_feed

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "app_store_feed.json"
FEED_URL_PATTERN = (
    r"https://itunes\.apple\.com/us/rss/customerreviews"
    r"/page=(?P<page>\d+)/id=570060128/sortby=mostrecent/json"
)
FAST_RETRY = RetryPolicy(
    max_attempts=3, initial_wait_seconds=0, max_wait_seconds=0, jitter_seconds=0
)
EMPTY_FEED: dict[str, Any] = {"feed": {"author": {}}}


def load_feed() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


async def collect_pages(client: httpx.AsyncClient) -> list[ReviewPage]:
    source = AppStoreSource(client, retry_policy=FAST_RETRY)
    return [page async for page in source.fetch_pages("570060128", Market("us", "en"))]


class TestParseFeed:
    def test_maps_entries_to_reviews_and_skips_malformed_ones(self) -> None:
        reviews = parse_feed(load_feed(), country="us")

        assert [review.external_id for review in reviews] == ["90000000001", "90000000002"]
        first = reviews[0]
        assert first.rating == 2
        assert first.title == "Streak lost after update"
        assert first.app_version == "7.139.0"
        assert first.helpful_count == 3
        assert first.country == "us"
        assert first.language is None

    def test_converts_timestamps_to_utc(self) -> None:
        review = parse_feed(load_feed(), country="us")[0]

        assert review.reviewed_at == datetime(2026, 9, 12, 15, 33, 38, tzinfo=UTC)

    def test_empty_title_becomes_none(self) -> None:
        assert parse_feed(load_feed(), country="us")[1].title is None

    def test_never_keeps_author_identity(self) -> None:
        review = parse_feed(load_feed(), country="us")[0]

        assert "author" not in review.raw
        assert review.author_hash is not None
        assert "sample-user-1" not in json.dumps(review.raw)

    def test_accepts_single_entry_serialized_as_object(self) -> None:
        payload = load_feed()
        payload["feed"]["entry"] = payload["feed"]["entry"][0]

        assert len(parse_feed(payload, country="us")) == 1

    def test_feed_without_entries_yields_nothing(self) -> None:
        assert parse_feed(EMPTY_FEED, country="us") == []


class TestFetchPages:
    @respx.mock
    async def test_stops_when_feed_runs_out(self) -> None:
        feed = load_feed()
        respx.get(url__regex=FEED_URL_PATTERN).mock(
            side_effect=lambda request, page: httpx.Response(
                200, json=feed if page == "1" else EMPTY_FEED
            )
        )

        async with httpx.AsyncClient() as client:
            pages = await collect_pages(client)

        assert len(pages) == 1
        assert not pages[0].hit_provider_limit

    @respx.mock
    async def test_flags_provider_limit_on_the_last_available_page(self) -> None:
        route = respx.get(url__regex=FEED_URL_PATTERN).mock(
            return_value=httpx.Response(200, json=load_feed())
        )

        async with httpx.AsyncClient() as client:
            pages = await collect_pages(client)

        assert route.call_count == MAX_PAGES
        assert [page.hit_provider_limit for page in pages] == [False] * (MAX_PAGES - 1) + [True]

    @respx.mock
    async def test_retries_transient_failures(self) -> None:
        route = respx.get(url__regex=FEED_URL_PATTERN).mock(
            side_effect=[
                httpx.Response(503),
                httpx.ConnectError("connection reset"),
                httpx.Response(200, json=load_feed()),
                httpx.Response(200, json=EMPTY_FEED),
            ]
        )

        async with httpx.AsyncClient() as client:
            pages = await collect_pages(client)

        assert len(pages) == 1
        assert route.call_count == 4

    @respx.mock
    async def test_does_not_retry_client_errors(self) -> None:
        route = respx.get(url__regex=FEED_URL_PATTERN).mock(return_value=httpx.Response(404))

        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await collect_pages(client)

        assert route.call_count == 1
