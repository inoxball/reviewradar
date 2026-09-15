"""Composition root: assembles a production ingestion service from settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from reviewradar import __version__
from reviewradar.config import Settings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.domain import Store
from reviewradar.ingestion.retry import RetryPolicy
from reviewradar.ingestion.service import IngestionService
from reviewradar.ingestion.sources import AppStoreSource, GooglePlaySource, ReviewSource

USER_AGENT = f"reviewradar/{__version__}"


@asynccontextmanager
async def build_ingestion_service(settings: Settings) -> AsyncIterator[IngestionService]:
    """Yield a ready service, owning the lifecycle of its engine and HTTP client."""
    config = settings.ingestion
    retry_policy = RetryPolicy(max_attempts=config.max_attempts)
    engine = create_engine(settings.database_url)
    try:
        async with httpx.AsyncClient(
            timeout=config.http_timeout_seconds,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            sources: dict[Store, ReviewSource] = {
                Store.APP_STORE: AppStoreSource(
                    client,
                    retry_policy=retry_policy,
                    page_delay_seconds=config.page_delay_seconds,
                ),
                Store.GOOGLE_PLAY: GooglePlaySource(
                    retry_policy=retry_policy,
                    page_size=config.google_play_page_size,
                    page_delay_seconds=config.page_delay_seconds,
                ),
            }
            yield IngestionService(create_session_factory(engine), sources, config)
    finally:
        await engine.dispose()
