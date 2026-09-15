"""Composition root: assembles a production search service from settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from reviewradar.config import Settings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.enrichment.wiring import create_embedder
from reviewradar.search.service import SearchService


@asynccontextmanager
async def build_search_service(settings: Settings) -> AsyncIterator[SearchService]:
    """Yield a ready service using the same embedding model that enrichment stored."""
    embedder = create_embedder(settings.enrichment)
    engine = create_engine(settings.database_url)
    try:
        yield SearchService(create_session_factory(engine), embedder, settings.search)
    finally:
        await engine.dispose()
