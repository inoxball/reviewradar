"""Smoke tests against the real stores. Run explicitly with ``pytest -m live``."""

from datetime import UTC

import httpx
import pytest

from reviewradar.domain import Market
from reviewradar.ingestion.retry import RetryPolicy
from reviewradar.ingestion.sources import AppStoreSource, GooglePlaySource

pytestmark = pytest.mark.live

US_EN = Market("us", "en")
RETRY = RetryPolicy(max_attempts=2)


async def test_app_store_serves_duolingo_reviews() -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        pages = AppStoreSource(client, retry_policy=RETRY).fetch_pages("570060128", US_EN)
        first_page = await anext(pages)
        await pages.aclose()

    assert first_page.reviews
    assert all(review.reviewed_at.tzinfo is UTC for review in first_page.reviews)


async def test_google_play_serves_duolingo_reviews() -> None:
    pages = GooglePlaySource(retry_policy=RETRY, page_size=20).fetch_pages("com.duolingo", US_EN)
    first_page = await anext(pages)
    await pages.aclose()

    assert first_page.reviews
    assert all(review.reviewed_at.tzinfo is UTC for review in first_page.reviews)
