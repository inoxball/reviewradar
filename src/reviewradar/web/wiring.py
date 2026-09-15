"""Composition root: builds the panel's services from settings."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.anomalies.service import AnomalyService
from reviewradar.catalog import Catalog
from reviewradar.config import Settings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.enrichment.wiring import create_embedder
from reviewradar.replies.drafts import ReplyDraftService
from reviewradar.search.service import SearchService
from reviewradar.topics.wiring import create_topic_service
from reviewradar.translation.service import TranslationService
from reviewradar.translation.translator import LazyTranslator, import_translation_runtime
from reviewradar.translation.wiring import create_translator
from reviewradar.web.app import PanelServices

logger = logging.getLogger(__name__)


@asynccontextmanager
async def build_panel_services(settings: Settings) -> AsyncIterator[PanelServices]:
    """Yield panel services from the server lifespan, which runs on the main thread.

    The embedding model loads here, up front. The translation and reply models are large
    and optional for most views, so they load on their first request; their libraries are
    imported here so that loading can safely happen in a worker thread.
    """
    catalog = Catalog.from_yaml(settings.catalog_path)
    embedder = create_embedder(settings.enrichment)
    import_translation_runtime()
    translator = LazyTranslator(
        lambda: create_translator(settings.translation), model_name=settings.translation.model
    )
    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        search = SearchService(session_factory, embedder, settings.search)
        yield PanelServices(
            catalog=catalog,
            session_factory=session_factory,
            search=search,
            translation=TranslationService(session_factory, translator, settings.translation),
            judgments_path=settings.web.judgments_path,
            topics=create_topic_service(settings, session_factory),
            anomalies=AnomalyService(session_factory, settings.anomalies),
            replies=_reply_drafts(settings, session_factory, catalog),
            replies_benchmark_path=settings.replies.benchmark_dir / "summary.json",
            translation_model_loaded=lambda: translator.loaded,
        )
    finally:
        await engine.dispose()


def _reply_drafts(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession], catalog: Catalog
) -> ReplyDraftService | None:
    """Reply drafting needs the optional fine-tuning libraries; without them it is disabled."""
    from reviewradar.replies.wiring import create_reply_draft_service, import_reply_runtime

    try:
        import_reply_runtime()
    except ImportError as exc:
        logger.warning("reply drafting disabled, install the 'finetune' extra: %s", exc)
        return None
    return create_reply_draft_service(settings, session_factory, catalog)
