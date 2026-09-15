import json
from datetime import UTC, datetime
from typing import Any

import pytest
from google_play_scraper.exceptions import NotFoundError

from reviewradar.domain import Market, ReviewPage
from reviewradar.ingestion.retry import RetryPolicy
from reviewradar.ingestion.sources.google_play import GooglePlaySource, parse_records

US_EN = Market("us", "en")
FAST_RETRY = RetryPolicy(
    max_attempts=3, initial_wait_seconds=0, max_wait_seconds=0, jitter_seconds=0
)
# google-play-scraper builds timestamps with `datetime.fromtimestamp`: naive local time.
EPOCH_SECONDS = 1_789_146_029
SCRAPER_TIMESTAMP = datetime.fromtimestamp(EPOCH_SECONDS)


def scraper_record(review_id: str = "gp-1", **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "reviewId": review_id,
        "userName": "Sample User",
        "userImage": "https://example.invalid/avatar.png",
        "content": "Great for Spanish practice",
        "score": 5,
        "thumbsUpCount": 1,
        "reviewCreatedVersion": "6.96.2",
        "at": SCRAPER_TIMESTAMP,
        "replyContent": None,
        "repliedAt": None,
        "appVersion": "6.96.2",
    }
    return record | overrides


class ContinuationToken:
    """Mimics the scraper's token: an inner ``token`` of ``None`` means no more pages."""

    def __init__(self, token: str | None) -> None:
        self.token = token


class ScriptedFetcher:
    """Stands in for the blocking scraper call, serving pages and injected failures."""

    def __init__(
        self, pages: list[list[dict[str, Any]]], failures: list[Exception] | None = None
    ) -> None:
        self.calls: list[str | None] = []
        self._pages = pages
        self._failures = list(failures or [])

    def __call__(
        self, app_id: str, market: Market, page_size: int, token: ContinuationToken | None
    ) -> tuple[list[dict[str, Any]], ContinuationToken]:
        self.calls.append(token.token if token else None)
        if self._failures:
            raise self._failures.pop(0)
        index = int(token.token) if token and token.token else 0
        has_next = index + 1 < len(self._pages)
        return self._pages[index], ContinuationToken(str(index + 1) if has_next else None)


async def collect_pages(fetcher: ScriptedFetcher) -> list[ReviewPage]:
    source = GooglePlaySource(retry_policy=FAST_RETRY, fetch_page=fetcher)
    return [page async for page in source.fetch_pages("com.duolingo", US_EN)]


class TestParseRecords:
    def test_maps_scraper_fields(self) -> None:
        (review,) = parse_records([scraper_record()], US_EN)

        assert review.external_id == "gp-1"
        assert review.rating == 5
        assert review.body == "Great for Spanish practice"
        assert review.app_version == "6.96.2"
        assert review.helpful_count == 1
        assert (review.country, review.language) == (None, "en")

    def test_converts_naive_local_timestamps_to_utc(self) -> None:
        (review,) = parse_records([scraper_record()], US_EN)

        assert review.reviewed_at == datetime.fromtimestamp(EPOCH_SECONDS, UTC)
        assert review.reviewed_at.tzinfo is UTC

    def test_maps_developer_reply(self) -> None:
        record = scraper_record(
            replyContent="Thanks for learning with us!", repliedAt=SCRAPER_TIMESTAMP
        )

        (review,) = parse_records([record], US_EN)

        assert review.developer_reply == "Thanks for learning with us!"
        assert review.developer_replied_at == datetime.fromtimestamp(EPOCH_SECONDS, UTC)

    def test_raw_payload_is_json_safe_and_free_of_personal_data(self) -> None:
        (review,) = parse_records([scraper_record()], US_EN)

        serialized = json.dumps(review.raw)
        assert "Sample User" not in serialized
        assert "avatar" not in serialized
        assert review.author_hash is not None

    def test_skips_malformed_records(self) -> None:
        reviews = parse_records([scraper_record("ok"), scraper_record("bad", score=None)], US_EN)

        assert [review.external_id for review in reviews] == ["ok"]


class TestFetchPages:
    async def test_follows_continuation_tokens_until_exhausted(self) -> None:
        fetcher = ScriptedFetcher(
            [[scraper_record("a")], [scraper_record("b")], [scraper_record("c")]]
        )

        pages = await collect_pages(fetcher)

        assert [page.reviews[0].external_id for page in pages] == ["a", "b", "c"]
        assert fetcher.calls == [None, "1", "2"]

    async def test_stops_on_empty_page(self) -> None:
        assert await collect_pages(ScriptedFetcher([[]])) == []

    async def test_retries_transient_failures(self) -> None:
        fetcher = ScriptedFetcher([[scraper_record()]], failures=[ConnectionResetError()])

        pages = await collect_pages(fetcher)

        assert len(pages) == 1
        assert len(fetcher.calls) == 2

    async def test_does_not_retry_unknown_app(self) -> None:
        fetcher = ScriptedFetcher([[scraper_record()]], failures=[NotFoundError("no such app")])

        with pytest.raises(NotFoundError):
            await collect_pages(fetcher)
        assert len(fetcher.calls) == 1

    def test_markets_differing_only_by_country_share_a_partition(self) -> None:
        source = GooglePlaySource(retry_policy=FAST_RETRY)

        assert source.partition_key(US_EN) == source.partition_key(Market("gb", "en")) == "en"
