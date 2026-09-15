from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import EnrichmentSettings, TopicSettings
from reviewradar.db.models import ReviewTopic, TopicRun
from reviewradar.domain import Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.topics.clustering import KMeansClusterer
from reviewradar.topics.naming import TopicDescription
from reviewradar.topics.repository import TopicRepository
from reviewradar.topics.service import (
    InsufficientReviewsError,
    NoTopicRunError,
    TopicScope,
    TopicService,
)
from tests.fakes import FakeEmbedder, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.GOOGLE_PLAY)
# A wide hashing space keeps the fake embedder free of token collisions.
EMBEDDER = FakeEmbedder(dimension=1024)

ENERGY = [
    "energy runs out after one lesson",
    "no energy left to finish the lesson",
    "the energy system blocks my lesson",
    "energy refill takes hours before a lesson",
]
ADS = [
    "too many ads after every video",
    "an ads video interrupts everything",
    "the ads video is so annoying",
    "forced to watch ads video again",
]


async def seed(session_factory: SessionFactory) -> None:
    reviews = [
        *(make_review(f"energy-{index}", body=text, rating=1) for index, text in enumerate(ENERGY)),
        *(make_review(f"ads-{index}", body=text, rating=3) for index, text in enumerate(ADS)),
        make_review("praise", body="love the owl mascot and streaks", rating=5),
    ]
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(
            listing_ids[Store.GOOGLE_PLAY], reviews, seen_at=NOW
        )
    await EnrichmentService(
        session_factory,
        FixedLanguageDetector("en", confidence=0.99),
        EMBEDDER,
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)


def build_service(session_factory: SessionFactory, **settings: Any) -> TopicService:
    return TopicService(
        session_factory,
        lambda topic_count: KMeansClusterer(topic_count, seed=0),
        TopicSettings(**({"min_reviews": 4, "min_words": 3, "topic_count": 2} | settings)),
        embedding_model=EMBEDDER.model_name,
        translation_model="fake/translator",
        target_language="en",
        clock=lambda: NOW,
    )


@pytest.fixture
async def seeded(session_factory: SessionFactory) -> SessionFactory:
    await seed(session_factory)
    return session_factory


async def test_discovers_labelled_topics_with_statistics(seeded: SessionFactory) -> None:
    report = await build_service(seeded).discover(APP)

    assert report.review_count == 8  # the 5★ review is outside the default ≤3★ scope
    energy, ads = sorted(report.topics, key=lambda topic: topic.average_rating)
    assert "energy" in energy.label
    assert "ads" in ads.label
    assert (energy.size, energy.average_rating, energy.negative_share) == (4, 1.0, 1.0)
    assert (ads.size, ads.average_rating, ads.negative_share) == (4, 3.0, 0.0)


async def test_stores_the_run_topics_and_assignments(seeded: SessionFactory) -> None:
    report = await build_service(seeded).discover(APP)

    async with seeded() as session:
        repository = TopicRepository(session)
        run = await repository.latest_run("duolingo")
        topics = await repository.topics(report.run_id)
        representatives = await repository.representatives(report.run_id, per_topic=2)
        daily = await repository.daily_counts(report.run_id)
        assignments = await session.scalar(select(func.count()).select_from(ReviewTopic))

    assert run is not None
    assert run.id == report.run_id
    assert run.parameters | {"n_init": None} == {
        "n_clusters": 2,
        "seed": 0,
        "n_init": None,
        "max_rating": 3,
        "min_words": 3,
    }
    assert [topic.size for topic in topics] == [4, 4]
    assert assignments == 8
    assert all(len(ids) == 2 for ids in representatives.values())
    assert all(sum(days.values()) == 4 for days in daily.values())


async def test_scope_can_include_every_rating(seeded: SessionFactory) -> None:
    report = await build_service(seeded).discover(
        APP, scope=TopicScope(max_rating=None, min_words=3)
    )

    assert report.review_count == 9


async def test_refuses_to_cluster_too_few_reviews(seeded: SessionFactory) -> None:
    with pytest.raises(InsufficientReviewsError, match="at least 50"):
        await build_service(seeded, min_reviews=50).discover(APP)


async def test_keeps_only_the_most_recent_runs(seeded: SessionFactory) -> None:
    service = build_service(seeded, runs_to_keep=2)

    for _ in range(3):
        await service.discover(APP)

    async with seeded() as session:
        runs = await session.scalar(select(func.count()).select_from(TopicRun))
        assignments = await session.scalar(select(func.count()).select_from(ReviewTopic))
    assert runs == 2
    assert assignments == 16


async def test_topic_count_can_be_chosen_per_run(seeded: SessionFactory) -> None:
    report = await build_service(seeded).discover(APP, topic_count=3)

    assert len(report.topics) == 3
    assert sum(topic.size for topic in report.topics) == 8


class FakeNamer:
    model_name = "fake/namer"

    def __init__(self) -> None:
        self.seen: list[TopicDescription] = []

    def name(self, app_name: str, topics: Sequence[TopicDescription]) -> list[str | None]:
        self.seen += topics
        return ["Energy runs out mid-lesson" if "energy" in t.keywords else None for t in topics]


async def test_relabels_the_latest_run_with_keywords(seeded: SessionFactory) -> None:
    service = build_service(seeded)
    discovered = await service.discover(APP)

    report = await service.relabel(APP)

    assert report.run_id == discovered.run_id
    assert report.model is None
    assert {topic.label for topic in report.topics} == {topic.label for topic in discovered.topics}
    async with seeded() as session:
        run = await TopicRepository(session).latest_run(APP.slug)
    assert run is not None
    assert run.parameters["naming"]["model"] is None
    assert run.parameters["n_clusters"] == 2  # clustering parameters are kept


async def test_relabels_with_model_names_and_keeps_keywords_as_fallback(
    seeded: SessionFactory,
) -> None:
    service = build_service(seeded)
    await service.discover(APP)
    namer = FakeNamer()

    report = await service.relabel(APP, namer)

    assert report.fallbacks == 1
    labels = {topic.label for topic in report.topics}
    assert "Energy runs out mid-lesson" in labels
    assert any(" · " in label for label in labels)
    assert all(1 <= len(topic.examples) <= 4 for topic in namer.seen)
    async with seeded() as session:
        repository = TopicRepository(session)
        run = await repository.latest_run(APP.slug)
        assert run is not None
        stored = await repository.topics(run.id)
    assert {topic.label for topic in stored} == labels
    assert run.parameters["naming"] == {
        "model": "fake/namer",
        "labelled_at": NOW.isoformat(),
        "fallbacks": 1,
    }


async def test_relabelling_needs_a_topic_run(seeded: SessionFactory) -> None:
    with pytest.raises(NoTopicRunError):
        await build_service(seeded).relabel(APP)
