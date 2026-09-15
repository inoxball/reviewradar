"""Composition root: builds a topic service and the topic namer from settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import Settings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.topics.clustering import KMeansClusterer, import_clustering_runtime
from reviewradar.topics.naming import LocalTopicNamer
from reviewradar.topics.service import TopicService

NAME_TOKENS = 24
NAMING_BATCH_SIZE = 8


def create_topic_service(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> TopicService:
    """A topic service over stored embeddings; call on the main thread.

    It loads no model, but imports scikit-learn, see ``import_clustering_runtime``.
    """
    import_clustering_runtime()
    seed = settings.topics.seed
    return TopicService(
        session_factory,
        lambda topic_count: KMeansClusterer(topic_count, seed=seed),
        settings.topics,
        embedding_model=settings.enrichment.embedding_model,
        translation_model=settings.translation.model,
        target_language=settings.translation.target_language,
    )


def create_topic_namer(settings: Settings) -> LocalTopicNamer:
    """Load the local naming model in 4 bits on the GPU; call on the main thread.

    It needs the optional 'finetune' extra (bitsandbytes, peft) and imports torch.
    """
    from reviewradar.replies.generation import GenerationOptions, LocalReplyModel
    from reviewradar.replies.wiring import import_reply_runtime

    import_reply_runtime()
    options = GenerationOptions(max_new_tokens=NAME_TOKENS, batch_size=NAMING_BATCH_SIZE)
    model = LocalReplyModel(settings.topics.naming_model, options=options)
    return LocalTopicNamer(model, model_name=settings.topics.naming_model)


@asynccontextmanager
async def build_topic_service(settings: Settings) -> AsyncIterator[TopicService]:
    """Yield a topic service with its own engine."""
    engine = create_engine(settings.database_url)
    try:
        yield create_topic_service(settings, create_session_factory(engine))
    finally:
        await engine.dispose()
