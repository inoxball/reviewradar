from dataclasses import replace
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.anomalies.repository import AnomalyRepository, source_key
from reviewradar.anomalies.service import AnomalyService
from reviewradar.config import AnomalySettings
from reviewradar.db.models import Review
from reviewradar.domain import FetchedReview, Store
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.topics.repository import NewTopic, TopicRepository
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.APP_STORE, markets=(("de", "de"),))
LABELS = ("energy", "icon", "ads", "streak", "price")
DAYS = 10
SPIKE_DAY = 5
SURGE = 30


def review(external_id: str, days_ago: int, version: str = "7.137.0") -> FetchedReview:
    base = make_review(
        external_id,
        body="Die App ist nicht gut",
        rating=1,
        country="de",
        age=timedelta(days=days_ago),
    )
    return replace(base, app_version=version)


async def seed_run(session_factory: SessionFactory) -> int:
    """Five topics with three reviews a day, and one day when the icon topic surges."""
    reviews: list[FetchedReview] = []
    members: dict[str, list[str]] = {label: [] for label in LABELS}
    for day in range(DAYS):
        for label in LABELS:
            for index in range(3):
                reviews.append(review(f"{label}-{day}-{index}", day))
                members[label].append(f"{label}-{day}-{index}")
    for index in range(SURGE):
        reviews.append(review(f"surge-{index}", SPIKE_DAY, version="7.138.0"))
        members["icon"].append(f"surge-{index}")

    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(
            listing_ids[Store.APP_STORE], reviews, seen_at=NOW
        )
        rows = await session.execute(select(Review.external_id, Review.id))
        ids = dict(rows.tuples().all())
        topics = TopicRepository(session)
        app_id = await topics.app_id(APP.slug)
        assert app_id is not None
        return await topics.save_run(
            app_id=app_id,
            embedding_model="fake",
            algorithm="kmeans",
            parameters={},
            review_count=len(reviews),
            topics=[
                NewTopic(
                    label=label,
                    keywords=[label],
                    size=len(external_ids),
                    average_rating=1.0,
                    negative_share=1.0,
                    languages={"de": len(external_ids)},
                    members=[(ids[external_id], 0.1) for external_id in external_ids],
                )
                for label, external_ids in members.items()
            ],
            created_at=NOW,
        )


async def test_finds_the_surge_in_the_latest_topic_run(session_factory: SessionFactory) -> None:
    run_id = await seed_run(session_factory)

    report = await AnomalyService(session_factory, AnomalySettings()).detect(APP.slug)

    assert report is not None
    assert report.run.id == run_id
    (spike,) = report.detection.spikes
    assert report.topics[spike.topic_id].label == "icon"
    assert spike.source == "app_store:de"
    assert spike.day == (NOW - timedelta(days=SPIKE_DAY)).date()
    assert (spike.count, spike.total) == (3 + SURGE, len(LABELS) * 3 + SURGE)
    assert spike.top_version == "7.138.0"
    (coverage,) = report.detection.coverage
    assert (coverage.usable_days, coverage.tested) == (DAYS - 2, True)


async def test_reports_nothing_before_topic_discovery(session_factory: SessionFactory) -> None:
    async with session_factory.begin() as session:
        await CatalogRepository(session).sync_app(APP)

    assert await AnomalyService(session_factory, AnomalySettings()).detect(APP.slug) is None


async def test_review_texts_come_from_enrichment(session_factory: SessionFactory) -> None:
    await seed_run(session_factory)

    async with session_factory() as session:
        texts = await AnomalyRepository(session).review_texts([1, 2])

    assert texts == {}  # the seeded reviews are not enriched


def test_sources_follow_how_each_store_partitions_reviews() -> None:
    assert source_key(Store.APP_STORE, "de", "en") == "app_store:de"
    assert source_key(Store.GOOGLE_PLAY, None, "pt") == "google_play:pt"
    assert source_key(Store.GOOGLE_PLAY, None, None) == "google_play:unknown"


async def test_backtest_needs_earlier_coverage_before_it_alerts(
    session_factory: SessionFactory,
) -> None:
    await seed_run(session_factory)

    strict = await AnomalyService(session_factory, AnomalySettings()).backtest(APP.slug)
    relaxed = await AnomalyService(session_factory, AnomalySettings(min_baseline_days=3)).backtest(
        APP.slug
    )

    assert strict is not None
    assert relaxed is not None
    assert len(strict.backtest.days) == DAYS - 2
    (missed,) = strict.outcomes
    assert missed.reason is not None
    assert "only 3 earlier days" in missed.reason
    (caught,) = relaxed.outcomes
    assert caught.delay_days == 0
    (incident,) = relaxed.backtest.incidents
    assert relaxed.topics[incident.topic_id].label == "icon"
    assert incident.release == "7.138.0"


async def test_praise_is_left_out_even_when_the_run_covers_every_rating(
    session_factory: SessionFactory,
) -> None:
    run_id = await seed_run(session_factory)
    async with session_factory.begin() as session:
        await session.execute(update(Review).values(rating=5))

    async with session_factory() as session:
        repository = AnomalyRepository(session)
        critical = await repository.observations(run_id, max_rating=3)
        everything = await repository.observations(run_id, max_rating=None)

    assert critical == []
    assert len(everything) == len(LABELS) * 3 * DAYS + SURGE
