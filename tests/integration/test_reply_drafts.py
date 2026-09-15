from collections.abc import Sequence

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import Catalog
from reviewradar.config import EnrichmentSettings
from reviewradar.domain import Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.replies.drafts import (
    DraftStatus,
    ReplyDraftRepository,
    ReplyDraftService,
    ReplyGeneratorSet,
    UnknownGeneratorError,
)
from reviewradar.replies.generation import ReplyTask
from tests.fakes import FakeEmbedder, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.GOOGLE_PLAY).model_copy(update={"support_contact": "support.duolingo.com"})
EMBEDDER = FakeEmbedder(dimension=32)


class EchoGenerator:
    """Answers every review with the same template and counts its calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def generate(self, tasks: Sequence[ReplyTask]) -> list[str]:
        self.calls += 1
        return [
            f"Sorry about the {task.request.rating}-star experience. Contact {{support_contact}}."
            for task in tasks
        ]


@pytest.fixture
async def review_ids(session_factory: SessionFactory) -> dict[str, int]:
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(
            listing_ids[Store.GOOGLE_PLAY],
            [
                make_review("crash", body="The app crashes on every lesson", rating=1),
                make_review("praise", body="Great for daily practice", rating=5),
            ],
            seen_at=NOW,
        )
    await EnrichmentService(
        session_factory,
        FixedLanguageDetector("en", confidence=0.99),
        EMBEDDER,
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)
    async with session_factory() as session:
        candidates = await ReplyDraftRepository(session).candidates(
            range(1, 100), embedding_model=EMBEDDER.model_name
        )
    return {
        "crash" if candidate.rating == 1 else "praise": candidate.review_id
        for candidate in candidates.values()
    }


def build_service(
    session_factory: SessionFactory,
    generator: EchoGenerator,
    *,
    version: str = "v1",
    embedding_model: str = EMBEDDER.model_name,
) -> ReplyDraftService:
    async def load() -> ReplyGeneratorSet:
        return ReplyGeneratorSet(
            {generator.name: generator},
            {generator.name: version},
            requires_embedding=frozenset({"retrieval"}),
        )

    return ReplyDraftService(
        session_factory,
        Catalog(apps=(APP,)),
        load,
        available=[generator.name],
        embedding_model=embedding_model,
        detect_language=lambda _: "en",
        clock=lambda: NOW,
    )


async def test_drafts_render_placeholders_and_are_stored(
    session_factory: SessionFactory, review_ids: dict[str, int]
) -> None:
    generator = EchoGenerator("fine-tuned")
    service = build_service(session_factory, generator)

    (result,) = await service.draft([review_ids["crash"]], "fine-tuned")

    assert result.status is DraftStatus.DRAFTED
    assert result.text == "Sorry about the 1-star experience. Contact support.duolingo.com."
    assert result.template == "Sorry about the 1-star experience. Contact {support_contact}."
    assert result.checks is not None
    assert result.checks.passed
    assert service.loaded
    async with session_factory() as session:
        stored = await ReplyDraftRepository(session).stored(
            [review_ids["crash"]], generator="fine-tuned"
        )
    assert stored[review_ids["crash"]].version == "v1"


async def test_reuses_drafts_until_the_generator_version_changes(
    session_factory: SessionFactory, review_ids: dict[str, int]
) -> None:
    generator = EchoGenerator("fine-tuned")
    ids = [review_ids["crash"], review_ids["praise"]]

    first = await build_service(session_factory, generator).draft(ids, "fine-tuned")
    second = await build_service(session_factory, generator).draft(ids, "fine-tuned")
    retrained = await build_service(session_factory, generator, version="v2").draft(
        ids, "fine-tuned"
    )

    assert [result.status for result in first] == [DraftStatus.DRAFTED] * 2
    assert [result.status for result in second] == [DraftStatus.CACHED] * 2
    assert [result.status for result in retrained] == [DraftStatus.DRAFTED] * 2
    assert generator.calls == 2


async def test_unknown_reviews_and_missing_embeddings_are_unavailable(
    session_factory: SessionFactory, review_ids: dict[str, int]
) -> None:
    generator = EchoGenerator("retrieval")
    service = build_service(session_factory, generator, embedding_model="other/model")

    results = await service.draft([review_ids["crash"], 999_999], "retrieval")

    assert [result.status for result in results] == [DraftStatus.UNAVAILABLE] * 2
    assert generator.calls == 0


async def test_rejects_generators_that_are_not_available(
    session_factory: SessionFactory, review_ids: dict[str, int]
) -> None:
    service = build_service(session_factory, EchoGenerator("zero-shot"))

    with pytest.raises(UnknownGeneratorError, match="not available"):
        await service.draft([review_ids["crash"]], "fine-tuned")
