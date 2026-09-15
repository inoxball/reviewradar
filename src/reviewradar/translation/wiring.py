"""Composition root: builds translators and translation services from settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from reviewradar.config import Settings, TranslationSettings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.translation.service import TranslationService
from reviewradar.translation.translator import M2M100Translator


def create_translator(config: TranslationSettings) -> M2M100Translator:
    """Load the configured translation model (call on the main thread; see the translator)."""
    return M2M100Translator(
        config.model,
        device=config.device,
        batch_size=config.batch_size,
        max_input_tokens=config.max_input_tokens,
        num_beams=config.num_beams,
    )


@asynccontextmanager
async def build_translation_service(settings: Settings) -> AsyncIterator[TranslationService]:
    """Yield a service for batch translation; the model loads before any work starts."""
    translator = create_translator(settings.translation)
    engine = create_engine(settings.database_url)
    try:
        yield TranslationService(create_session_factory(engine), translator, settings.translation)
    finally:
        await engine.dispose()
