from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import EnrichmentSettings, TranslationSettings
from reviewradar.domain import Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.translation.service import TranslationService
from tests.fakes import FakeEmbedder, FakeTranslator, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

APP = duolingo(Store.GOOGLE_PLAY)


async def test_translates_only_reviews_within_the_rating_limit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(
            listing_ids[Store.GOOGLE_PLAY],
            [
                make_review("critical", body="a energia acaba muito rápido", rating=1),
                make_review("happy", body="aplicativo muito bom", rating=5),
            ],
            seen_at=NOW,
        )
    await EnrichmentService(
        session_factory,
        FixedLanguageDetector("pt", confidence=0.99),
        FakeEmbedder(),
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)
    translator = FakeTranslator()
    service = TranslationService(
        session_factory, translator, TranslationSettings(), clock=lambda: NOW
    )

    critical_only = await service.translate_app(APP.slug, max_rating=3)
    everything_else = await service.translate_app(APP.slug)

    assert critical_only.reviews_translated == 1
    assert everything_else.reviews_translated == 1
    assert [text for _, texts in translator.calls for text in texts] == [
        "a energia acaba muito rápido",
        "aplicativo muito bom",
    ]
