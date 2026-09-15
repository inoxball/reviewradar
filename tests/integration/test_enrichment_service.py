from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import EnrichmentSettings
from reviewradar.db.models import Review, ReviewEmbedding, ReviewEnrichment
from reviewradar.domain import FetchedReview, LanguageSource, Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from tests.fakes import FakeEmbedder, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.GOOGLE_PLAY, Store.APP_STORE, markets=[("us", "en"), ("br", "pt")])


def build_service(
    session_factory: SessionFactory,
    *,
    embedder: FakeEmbedder | None = None,
    detector: FixedLanguageDetector | None = None,
    enrichment_version: int = 1,
    **settings: Any,
) -> EnrichmentService:
    return EnrichmentService(
        session_factory,
        detector or FixedLanguageDetector("en", confidence=0.99),
        embedder or FakeEmbedder(),
        EnrichmentSettings(**({"min_language_chars": 1} | settings)),
        enrichment_version=enrichment_version,
        clock=lambda: NOW,
    )


async def seed(
    session_factory: SessionFactory, *reviews: FetchedReview, store: Store = Store.GOOGLE_PLAY
) -> None:
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(listing_ids[store], reviews, seen_at=NOW)


async def enrichment_rows(session_factory: SessionFactory) -> dict[str, ReviewEnrichment]:
    async with session_factory() as session:
        rows = await session.execute(
            select(Review.external_id, ReviewEnrichment).join(
                ReviewEnrichment, ReviewEnrichment.review_id == Review.id
            )
        )
        return {external_id: enrichment for external_id, enrichment in rows}


async def embedding_rows(
    session_factory: SessionFactory, model: str
) -> dict[str, NDArray[np.float32]]:
    async with session_factory() as session:
        rows = await session.execute(
            select(Review.external_id, ReviewEmbedding.embedding)
            .join(ReviewEmbedding, ReviewEmbedding.review_id == Review.id)
            .where(ReviewEmbedding.model == model)
        )
        return {external_id: vector for external_id, vector in rows}


async def test_enriches_reviews_with_text_language_and_normalized_embeddings(
    session_factory: SessionFactory,
) -> None:
    embedder = FakeEmbedder()
    await seed(
        session_factory,
        make_review("a", body="Crashes on launch"),
        make_review("b", body="Love the speaking drills"),
    )

    report = await build_service(session_factory, embedder=embedder).enrich_app(APP)

    assert (report.reviews_processed, report.reviews_embedded) == (2, 2)
    assert report.language_sources == {LanguageSource.DETECTED: 2}
    enrichment = (await enrichment_rows(session_factory))["a"]
    assert enrichment.model_text == "Crashes on launch"
    assert (enrichment.language, enrichment.language_source) == ("en", LanguageSource.DETECTED)
    vectors = await embedding_rows(session_factory, embedder.model_name)
    assert vectors["a"].dtype == np.float32
    np.testing.assert_allclose(vectors["a"], embedder.embed_query("Crashes on launch"), rtol=1e-6)
    assert float(np.linalg.norm(vectors["b"])) == pytest.approx(1.0, abs=1e-6)


async def test_rerun_is_a_no_op(session_factory: SessionFactory) -> None:
    await seed(session_factory, make_review("a"))
    service = build_service(session_factory)

    await service.enrich_app(APP)
    report = await service.enrich_app(APP)

    assert report.reviews_processed == 0


async def test_edited_text_is_re_enriched_and_re_embedded(session_factory: SessionFactory) -> None:
    embedder = FakeEmbedder()
    await seed(session_factory, make_review("a", body="Great course"), make_review("b"))
    await build_service(session_factory, embedder=embedder).enrich_app(APP)

    await seed(
        session_factory, make_review("a", body="Now it crashes constantly"), make_review("b")
    )
    report = await build_service(session_factory, embedder=embedder).enrich_app(APP)

    assert (report.reviews_processed, report.reviews_embedded) == (1, 1)
    vectors = await embedding_rows(session_factory, embedder.model_name)
    np.testing.assert_allclose(
        vectors["a"], embedder.embed_query("Now it crashes constantly"), rtol=1e-6
    )


async def test_rating_only_edit_re_enriches_without_re_embedding(
    session_factory: SessionFactory,
) -> None:
    embedder = FakeEmbedder()
    await seed(session_factory, make_review("a", body="Great course", rating=5))
    await build_service(session_factory, embedder=embedder).enrich_app(APP)

    await seed(session_factory, make_review("a", body="Great course", rating=1))
    report = await build_service(session_factory, embedder=embedder).enrich_app(APP)

    assert (report.reviews_processed, report.reviews_embedded) == (1, 0)
    assert embedder.documents == ["Great course"]


async def test_pipeline_version_bump_recomputes_enrichment_but_reuses_embeddings(
    session_factory: SessionFactory,
) -> None:
    embedder = FakeEmbedder()
    await seed(session_factory, make_review("a"))
    await build_service(session_factory, embedder=embedder, enrichment_version=1).enrich_app(APP)

    report = await build_service(
        session_factory, embedder=embedder, enrichment_version=2
    ).enrich_app(APP)

    assert (report.reviews_processed, report.reviews_embedded) == (1, 0)
    assert (await enrichment_rows(session_factory))["a"].enrichment_version == 2


async def test_review_without_words_is_enriched_once_and_never_embedded(
    session_factory: SessionFactory,
) -> None:
    await seed(session_factory, make_review("emoji", body="👍👍"))

    first = await build_service(session_factory).enrich_app(APP)
    second = await build_service(session_factory).enrich_app(APP)

    assert (first.reviews_processed, first.reviews_embedded, first.reviews_without_text) == (
        1,
        0,
        1,
    )
    assert second.reviews_processed == 0
    assert (await enrichment_rows(session_factory))["emoji"].model_text == ""


async def test_review_edited_to_empty_text_loses_its_embedding(
    session_factory: SessionFactory,
) -> None:
    embedder = FakeEmbedder()
    await seed(session_factory, make_review("a", body="Great course"))
    await build_service(session_factory, embedder=embedder).enrich_app(APP)

    await seed(session_factory, make_review("a", body="👍"))
    await build_service(session_factory, embedder=embedder).enrich_app(APP)

    assert await embedding_rows(session_factory, embedder.model_name) == {}


async def test_personal_data_never_reaches_the_embedder(session_factory: SessionFactory) -> None:
    embedder = FakeEmbedder()
    await seed(session_factory, make_review("a", body="Refund me, write to jane.doe@example.com"))

    await build_service(session_factory, embedder=embedder).enrich_app(APP)

    assert embedder.documents == ["Refund me, write to [email]"]


async def test_language_evidence_is_resolved_per_store(session_factory: SessionFactory) -> None:
    await seed(session_factory, replace(make_review("play"), country=None, language="de"))
    await seed(
        session_factory,
        make_review("apple-br", country="br"),
        make_review("apple-ca", country="ca"),
        store=Store.APP_STORE,
    )
    uncertain = FixedLanguageDetector("zu", confidence=0.2)

    await build_service(session_factory, detector=uncertain).enrich_app(APP)

    rows = await enrichment_rows(session_factory)
    resolved = {
        external_id: (row.language, row.language_source, row.language_candidate)
        for external_id, row in rows.items()
    }
    assert resolved == {
        "play": ("de", LanguageSource.STORE_HINT, "zu"),
        "apple-br": ("pt", LanguageSource.MARKET_DEFAULT, "zu"),
        "apple-ca": (None, LanguageSource.UNKNOWN, "zu"),
    }


async def test_new_embedding_model_backfills_and_keeps_previous_vectors(
    session_factory: SessionFactory,
) -> None:
    await seed(session_factory, make_review("a"), make_review("b"))
    await build_service(session_factory, embedder=FakeEmbedder("fake/v1")).enrich_app(APP)

    report = await build_service(
        session_factory, embedder=FakeEmbedder("fake/v2", dimension=8)
    ).enrich_app(APP)

    assert (report.reviews_processed, report.reviews_embedded) == (2, 2)
    assert len(await embedding_rows(session_factory, "fake/v1")) == 2
    v2 = await embedding_rows(session_factory, "fake/v2")
    assert {vector.shape for vector in v2.values()} == {(8,)}


async def test_paginates_in_batches_and_honours_limit(session_factory: SessionFactory) -> None:
    await seed(session_factory, *(make_review(f"r{index}") for index in range(5)))

    limited = await build_service(session_factory, review_batch_size=2).enrich_app(APP, limit=3)
    remainder = await build_service(session_factory, review_batch_size=2).enrich_app(APP)

    assert (limited.reviews_processed, remainder.reviews_processed) == (3, 2)
