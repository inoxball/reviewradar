from dataclasses import replace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import EnrichmentSettings, TranslationSettings
from reviewradar.db.models import Review
from reviewradar.domain import FetchedReview, Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.translation.service import TranslationOutcome, TranslationService
from tests.fakes import FakeEmbedder, FakeTranslator, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.GOOGLE_PLAY)


def in_language(external_id: str, language: str, body: str) -> FetchedReview:
    """A Google Play review whose language comes from the store hint."""
    return replace(make_review(external_id, body=body), country=None, language=language)


async def seed(session_factory: SessionFactory, *reviews: FetchedReview) -> None:
    """Store and enrich reviews; an unsure detector makes the store hint decide the language."""
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(
            listing_ids[Store.GOOGLE_PLAY], reviews, seen_at=NOW
        )
    await EnrichmentService(
        session_factory,
        FixedLanguageDetector("zz", confidence=0.1),
        FakeEmbedder(),
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)


async def review_ids(session_factory: SessionFactory) -> dict[str, int]:
    async with session_factory() as session:
        rows = await session.execute(select(Review.external_id, Review.id))
        return {external_id: review_id for external_id, review_id in rows}


def build_service(
    session_factory: SessionFactory, translator: FakeTranslator, **settings: object
) -> TranslationService:
    return TranslationService(
        session_factory, translator, TranslationSettings(**settings), clock=lambda: NOW
    )


@pytest.fixture
async def corpus(session_factory: SessionFactory) -> dict[str, int]:
    await seed(
        session_factory,
        in_language("pt", "pt", "muito bom"),
        in_language("de", "de", "sehr gut"),
        in_language("en", "en", "very good"),
        in_language("xx", "xx", "unsupported language"),
        in_language("emoji", "pt", "👍"),
    )
    return await review_ids(session_factory)


async def test_translates_pending_reviews_in_other_supported_languages(
    session_factory: SessionFactory, corpus: dict[str, int]
) -> None:
    translator = FakeTranslator()

    report = await build_service(session_factory, translator).translate_app("duolingo")

    assert (report.reviews_translated, report.reviews_unsupported) == (2, 1)
    assert sorted(translator.calls) == [("de", ["sehr gut"]), ("pt", ["muito bom"])]


async def test_rerun_does_not_translate_again(
    session_factory: SessionFactory, corpus: dict[str, int]
) -> None:
    translator = FakeTranslator()
    await build_service(session_factory, translator).translate_app("duolingo")
    translator.calls.clear()

    report = await build_service(session_factory, translator).translate_app("duolingo")

    assert report.reviews_translated == 0
    assert translator.calls == []


async def test_edited_text_invalidates_the_translation(
    session_factory: SessionFactory, corpus: dict[str, int]
) -> None:
    translator = FakeTranslator()
    await build_service(session_factory, translator).translate_app("duolingo")
    translator.calls.clear()

    await seed(session_factory, in_language("pt", "pt", "agora trava sempre"))
    report = await build_service(session_factory, translator).translate_app("duolingo")

    assert report.reviews_translated == 1
    assert translator.calls == [("pt", ["agora trava sempre"])]


async def test_on_demand_translation_reports_outcomes_and_caches(
    session_factory: SessionFactory, corpus: dict[str, int]
) -> None:
    translator = FakeTranslator()
    service = build_service(session_factory, translator)
    requested = [corpus["pt"], corpus["en"], corpus["xx"], corpus["emoji"], 999_999]

    first = await service.translate_reviews(requested)
    second = await service.translate_reviews(requested)

    assert [result.outcome for result in first] == [
        TranslationOutcome.TRANSLATED,
        TranslationOutcome.NOT_NEEDED,
        TranslationOutcome.UNAVAILABLE,
        TranslationOutcome.UNAVAILABLE,
        TranslationOutcome.UNAVAILABLE,
    ]
    assert first[0].text == "[pt→en] muito bom"
    assert second[0].outcome is TranslationOutcome.CACHED
    assert second[0].text == first[0].text
    assert len(translator.calls) == 1


async def test_on_demand_translations_are_reused_by_batch_runs(
    session_factory: SessionFactory, corpus: dict[str, int]
) -> None:
    translator = FakeTranslator()
    service = build_service(session_factory, translator)
    await service.translate_reviews([corpus["pt"]])

    report = await service.translate_app("duolingo")

    assert report.reviews_translated == 1
    assert translator.calls[-1] == ("de", ["sehr gut"])
