"""Composition root: assembles a production enrichment service from settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from reviewradar.config import EnrichmentSettings, Settings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.enrichment.embeddings import SentenceTransformerEmbedder
from reviewradar.enrichment.language import LangidDetector
from reviewradar.enrichment.service import EnrichmentService


def create_embedder(config: EnrichmentSettings) -> SentenceTransformerEmbedder:
    """Load the configured embedding model. Call this on the main thread.

    Importing torch/transformers for the first time inside a worker thread makes the
    interpreter crash at exit on Windows (access violation 0xC0000005), even after a
    successful run. Once initialized on the main thread, inference in worker threads is safe.
    """
    return SentenceTransformerEmbedder(
        config.embedding_model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
        document_prefix=config.document_prefix,
        query_prefix=config.query_prefix,
    )


@asynccontextmanager
async def build_enrichment_service(settings: Settings) -> AsyncIterator[EnrichmentService]:
    """Yield a ready service; the model loads before any concurrent work starts."""
    embedder = create_embedder(settings.enrichment)
    engine = create_engine(settings.database_url)
    try:
        yield EnrichmentService(
            create_session_factory(engine), LangidDetector(), embedder, settings.enrichment
        )
    finally:
        await engine.dispose()
