from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import EnrichmentSettings, SearchSettings
from reviewradar.db.base import Base
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.domain import FetchedReview, Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.search.retrievers import UnsupportedDatabaseError
from reviewradar.search.service import SearchService
from reviewradar.search.types import SearchFilters, SearchMode, SearchResult
from tests.fakes import FakeEmbedder, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.GOOGLE_PLAY, Store.APP_STORE)
# A wide hashing space keeps the fake embedder free of token collisions in these tests.
EMBEDDING_DIMENSION = 1024


async def seed(
    session_factory: SessionFactory,
    embedder: FakeEmbedder,
    *reviews: FetchedReview,
    store: Store = Store.GOOGLE_PLAY,
) -> None:
    """Store reviews and enrich them through the real enrichment pipeline."""
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(listing_ids[store], reviews, seen_at=NOW)
    await EnrichmentService(
        session_factory,
        FixedLanguageDetector("en", confidence=0.99),
        embedder,
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)


def build_search(
    session_factory: SessionFactory, embedder: FakeEmbedder, **settings: Any
) -> SearchService:
    return SearchService(session_factory, embedder, SearchSettings(**settings))


def external_ids(results: list[SearchResult]) -> list[str]:
    return [result.external_id for result in results]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=EMBEDDING_DIMENSION)


@pytest.fixture
async def ads_corpus(postgres_session_factory: SessionFactory, embedder: FakeEmbedder) -> None:
    await seed(
        postgres_session_factory,
        embedder,
        make_review("both", body="too many ads in every lesson"),
        make_review("partial", body="many ads here"),
        make_review("unrelated", body="love the owl mascot"),
    )


@pytest.mark.usefixtures("ads_corpus")
class TestModes:
    async def test_lexical_search_requires_every_query_term(
        self, postgres_session_factory: SessionFactory, embedder: FakeEmbedder
    ) -> None:
        results = await build_search(postgres_session_factory, embedder).search(
            "duolingo", "too many ads", mode=SearchMode.LEXICAL
        )

        assert external_ids(results) == ["both"]

    async def test_semantic_search_ranks_partial_matches_by_similarity(
        self, postgres_session_factory: SessionFactory, embedder: FakeEmbedder
    ) -> None:
        results = await build_search(postgres_session_factory, embedder).search(
            "duolingo", "too many ads", mode=SearchMode.SEMANTIC, limit=3
        )

        assert external_ids(results) == ["both", "partial", "unrelated"]
        assert results[0].score > results[1].score > results[2].score

    async def test_hybrid_search_puts_agreement_first_and_keeps_semantic_recall(
        self, postgres_session_factory: SessionFactory, embedder: FakeEmbedder
    ) -> None:
        results = await build_search(postgres_session_factory, embedder).search(
            "duolingo", "too many ads", mode=SearchMode.HYBRID, limit=2
        )

        assert external_ids(results) == ["both", "partial"]

    async def test_results_carry_review_details(
        self, postgres_session_factory: SessionFactory, embedder: FakeEmbedder
    ) -> None:
        (result,) = await build_search(postgres_session_factory, embedder).search(
            "duolingo", "owl mascot", mode=SearchMode.LEXICAL
        )

        assert (result.store, result.rating, result.language) == (Store.GOOGLE_PLAY, 5, "en")
        assert result.text == "love the owl mascot"
        assert result.reviewed_at.tzinfo is not None


async def test_filters_apply_inside_every_retriever(
    postgres_session_factory: SessionFactory, embedder: FakeEmbedder
) -> None:
    await seed(
        postgres_session_factory, embedder, make_review("play", body="too many ads", rating=5)
    )
    await seed(
        postgres_session_factory,
        embedder,
        make_review("apple", body="too many ads", rating=1),
        store=Store.APP_STORE,
    )
    service = build_search(postgres_session_factory, embedder)

    by_store = await service.search(
        "duolingo",
        "ads",
        mode=SearchMode.SEMANTIC,
        filters=SearchFilters(stores=frozenset({Store.APP_STORE})),
    )
    by_rating = await service.search(
        "duolingo", "ads", mode=SearchMode.LEXICAL, filters=SearchFilters(max_rating=2)
    )
    by_language = await service.search(
        "duolingo",
        "ads",
        mode=SearchMode.HYBRID,
        filters=SearchFilters(languages=frozenset({"de"})),
    )

    assert external_ids(by_store) == ["apple"]
    assert external_ids(by_rating) == ["apple"]
    assert by_language == []


async def test_semantic_search_only_uses_vectors_of_its_own_model(
    postgres_session_factory: SessionFactory,
) -> None:
    await seed(
        postgres_session_factory,
        FakeEmbedder("fake/indexed", dimension=EMBEDDING_DIMENSION),
        make_review("a", body="too many ads"),
    )
    service = build_search(
        postgres_session_factory, FakeEmbedder("fake/other", dimension=EMBEDDING_DIMENSION)
    )

    assert await service.search("duolingo", "ads", mode=SearchMode.SEMANTIC) == []
    assert external_ids(await service.search("duolingo", "ads", mode=SearchMode.LEXICAL)) == ["a"]


async def test_search_is_scoped_to_the_requested_app(
    postgres_session_factory: SessionFactory, embedder: FakeEmbedder
) -> None:
    await seed(postgres_session_factory, embedder, make_review("a", body="too many ads"))

    assert await build_search(postgres_session_factory, embedder).search("babbel", "ads") == []


async def test_rejects_empty_queries(
    postgres_session_factory: SessionFactory, embedder: FakeEmbedder
) -> None:
    with pytest.raises(ValueError, match="empty"):
        await build_search(postgres_session_factory, embedder).search("duolingo", "   ")


async def test_search_requires_postgres(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'search.db').as_posix()}")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        service = build_search(create_session_factory(engine), FakeEmbedder())

        with pytest.raises(UnsupportedDatabaseError, match="PostgreSQL"):
            await service.search("duolingo", "ads")
    finally:
        await engine.dispose()
