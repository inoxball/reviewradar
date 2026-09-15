from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import IngestionSettings
from reviewradar.db.models import IngestionCursor, IngestionRun, Review
from reviewradar.domain import RunStatus, Store
from reviewradar.ingestion.repository import UpsertStats
from reviewradar.ingestion.service import IngestionService
from tests.support import NOW, FakeSource, duolingo, make_review, page

SessionFactory = async_sessionmaker[AsyncSession]


def build_service(
    session_factory: SessionFactory, *sources: FakeSource, **settings: Any
) -> IngestionService:
    return IngestionService(
        session_factory,
        {source.store: source for source in sources},
        IngestionSettings(page_delay_seconds=0, **settings),
        clock=lambda: NOW,
    )


async def stored_reviews(session_factory: SessionFactory) -> list[Review]:
    async with session_factory() as session:
        return list(await session.scalars(select(Review).order_by(Review.external_id)))


async def cursor_for(session_factory: SessionFactory, partition_key: str) -> datetime | None:
    async with session_factory() as session:
        newest: datetime | None = await session.scalar(
            select(IngestionCursor.newest_reviewed_at).where(
                IngestionCursor.partition_key == partition_key
            )
        )
        return newest


async def recorded_runs(session_factory: SessionFactory) -> list[IngestionRun]:
    async with session_factory() as session:
        return list(await session.scalars(select(IngestionRun).order_by(IngestionRun.id)))


async def test_first_run_persists_reviews_and_records_cursor(
    session_factory: SessionFactory,
) -> None:
    source = FakeSource(
        Store.GOOGLE_PLAY,
        {
            "us-en": [
                page(
                    make_review("a", age=timedelta(hours=1)),
                    make_review("b", age=timedelta(hours=2)),
                )
            ]
        },
    )

    report = await build_service(session_factory, source).ingest_app(duolingo(Store.GOOGLE_PLAY))

    assert report.totals == UpsertStats(inserted=2)
    assert [review.external_id for review in await stored_reviews(session_factory)] == ["a", "b"]
    assert await cursor_for(session_factory, "us-en") == NOW - timedelta(hours=1)
    (run,) = await recorded_runs(session_factory)
    assert run.status is RunStatus.SUCCEEDED
    assert (run.fetched, run.inserted, run.coverage_complete) == (2, 2, True)


async def test_rerun_is_idempotent_and_picks_up_edits(session_factory: SessionFactory) -> None:
    app = duolingo(Store.GOOGLE_PLAY)
    original = page(make_review("a"), make_review("b"))
    edited = page(make_review("a", body="Now it crashes on launch", rating=1), make_review("b"))

    await build_service(
        session_factory, FakeSource(Store.GOOGLE_PLAY, {"us-en": [original]})
    ).ingest_app(app)
    report = await build_service(
        session_factory, FakeSource(Store.GOOGLE_PLAY, {"us-en": [edited]})
    ).ingest_app(app)

    assert report.totals == UpsertStats(updated=1, unchanged=1)
    first, second = await stored_reviews(session_factory)
    assert (first.body, first.rating) == ("Now it crashes on launch", 1)
    assert second.body == "Great app for daily practice"


async def test_stops_paginating_once_a_page_crosses_the_cutoff(
    session_factory: SessionFactory,
) -> None:
    source = FakeSource(
        Store.GOOGLE_PLAY,
        {
            "us-en": [
                page(
                    make_review("d1", age=timedelta(days=1)),
                    make_review("d10", age=timedelta(days=10)),
                ),
                page(
                    make_review("d29", age=timedelta(days=29)),
                    make_review("d31", age=timedelta(days=31)),
                ),
                page(make_review("d40", age=timedelta(days=40))),
            ]
        },
    )

    await build_service(session_factory, source, default_lookback_days=30).ingest_app(
        duolingo(Store.GOOGLE_PLAY)
    )

    assert source.served == [("us-en", 0), ("us-en", 1)]
    assert [r.external_id for r in await stored_reviews(session_factory)] == ["d1", "d10", "d29"]


async def test_incremental_run_resumes_from_cursor_and_catches_late_arrivals(
    session_factory: SessionFactory,
) -> None:
    app = duolingo(Store.GOOGLE_PLAY)
    await build_service(
        session_factory, FakeSource(Store.GOOGLE_PLAY, {"us-en": [page(make_review("fresh"))]})
    ).ingest_app(app)

    # Published after moderation, timestamped well before the newest review already seen.
    late = make_review("late", age=timedelta(hours=50))
    await build_service(
        session_factory,
        FakeSource(Store.GOOGLE_PLAY, {"us-en": [page(make_review("fresh"), late)]}),
        incremental_overlap_hours=72,
        overlap_hours_by_store={},
    ).ingest_app(app)

    _, second_run = await recorded_runs(session_factory)
    assert second_run.cutoff == NOW - timedelta(hours=1) - timedelta(hours=72)
    assert second_run.inserted == 1
    assert "late" in {r.external_id for r in await stored_reviews(session_factory)}


async def test_store_specific_overlap_overrides_the_default(
    session_factory: SessionFactory,
) -> None:
    app = duolingo(Store.GOOGLE_PLAY)
    await build_service(
        session_factory, FakeSource(Store.GOOGLE_PLAY, {"us-en": [page(make_review("fresh"))]})
    ).ingest_app(app)

    await build_service(
        session_factory,
        FakeSource(Store.GOOGLE_PLAY),
        incremental_overlap_hours=72,
        overlap_hours_by_store={Store.GOOGLE_PLAY: 336},
    ).ingest_app(app)

    _, second_run = await recorded_runs(session_factory)
    assert second_run.cutoff == NOW - timedelta(hours=1) - timedelta(hours=336)


async def test_failed_partition_is_isolated_and_does_not_advance_its_cursor(
    session_factory: SessionFactory,
) -> None:
    play = FakeSource(
        Store.GOOGLE_PLAY,
        {"us-en": [page(make_review("gp-1"))]},
        errors={"us-en": ConnectionError("store unavailable")},
    )
    app_store = FakeSource(Store.APP_STORE, {"us": [page(make_review("as-1"))]}, by_language=False)

    report = await build_service(session_factory, play, app_store).ingest_app(
        duolingo(Store.GOOGLE_PLAY, Store.APP_STORE)
    )

    (failure,) = report.failed
    assert failure.partition_key == "us-en"
    assert failure.error == "ConnectionError: store unavailable"
    # Pages written before the failure are kept; the next run re-covers the same window.
    assert {r.external_id for r in await stored_reviews(session_factory)} == {"gp-1", "as-1"}
    assert await cursor_for(session_factory, "us-en") is None
    assert await cursor_for(session_factory, "us") is not None


async def test_markets_sharing_a_partition_are_fetched_once(
    session_factory: SessionFactory,
) -> None:
    source = FakeSource(Store.APP_STORE, {"us": [page(make_review("a"))]}, by_language=False)

    report = await build_service(session_factory, source).ingest_app(
        duolingo(Store.APP_STORE, markets=[("us", "en"), ("us", "es")])
    )

    assert [result.partition_key for result in report.results] == ["us"]
    assert source.served == [("us", 0)]


async def test_overlapping_partitions_store_each_review_once(
    session_factory: SessionFactory,
) -> None:
    shared = page(make_review("shared"))
    source = FakeSource(Store.GOOGLE_PLAY, {"us-en": [shared], "gb-en": [shared]})

    report = await build_service(session_factory, source).ingest_app(
        duolingo(Store.GOOGLE_PLAY, markets=[("us", "en"), ("gb", "en")])
    )

    assert report.totals == UpsertStats(inserted=1, unchanged=1)
    assert len(await stored_reviews(session_factory)) == 1


async def test_provider_limit_marks_coverage_partial(session_factory: SessionFactory) -> None:
    source = FakeSource(
        Store.APP_STORE,
        {"us": [page(make_review("a"), hit_provider_limit=True)]},
        by_language=False,
    )

    report = await build_service(session_factory, source).ingest_app(duolingo(Store.APP_STORE))

    (result,) = report.results
    assert result.status is RunStatus.SUCCEEDED
    assert not result.coverage_complete
    (run,) = await recorded_runs(session_factory)
    assert run.coverage_complete is False


async def test_per_market_cap_keeps_newest_reviews(session_factory: SessionFactory) -> None:
    source = FakeSource(
        Store.GOOGLE_PLAY,
        {
            "us-en": [
                page(*(make_review(f"r{hour}", age=timedelta(hours=hour)) for hour in (1, 2, 3))),
                page(make_review("r4", age=timedelta(hours=4))),
            ]
        },
    )

    report = await build_service(session_factory, source, max_reviews_per_market=2).ingest_app(
        duolingo(Store.GOOGLE_PLAY)
    )

    assert [r.external_id for r in await stored_reviews(session_factory)] == ["r1", "r2"]
    assert source.served == [("us-en", 0)]
    assert not report.results[0].coverage_complete
