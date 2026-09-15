"""Orchestrates incremental, idempotent review ingestion across stores and markets.

For each app the service plans one job per (listing, partition) and runs the jobs
concurrently under a semaphore. Each job:

1. resolves its time window from an explicit ``since``, the saved cursor, or a
   default lookback;
2. streams pages newest first and upserts in-window reviews page by page, so
   progress survives a mid-run failure;
3. stops at the first page reaching past the window, at the per-market cap, or when
   the provider's history ends;
4. advances the cursor **only** on success. A failed job is simply retried over the
   same window next time, which the idempotent upserts make safe.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import AppSpec
from reviewradar.config import IngestionSettings
from reviewradar.domain import FetchedReview, Market, RunStatus, Store, utc_now
from reviewradar.ingestion.repository import (
    CatalogRepository,
    IngestionStateRepository,
    ReviewRepository,
    UpsertStats,
)
from reviewradar.ingestion.sources.base import ReviewSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PartitionJob:
    """One unit of work: a listing's review stream for one partition."""

    listing_id: int
    external_app_id: str
    source: ReviewSource
    market: Market
    partition_key: str

    def __str__(self) -> str:
        return f"store={self.source.store} partition={self.partition_key}"


@dataclass(frozen=True, slots=True)
class PartitionResult:
    """Outcome of one partition job."""

    store: Store
    partition_key: str
    status: RunStatus
    fetched: int
    stats: UpsertStats
    coverage_complete: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Outcome of ingesting one app across all of its partitions."""

    app_slug: str
    results: tuple[PartitionResult, ...]

    @property
    def totals(self) -> UpsertStats:
        return sum((result.stats for result in self.results), start=UpsertStats())

    @property
    def failed(self) -> tuple[PartitionResult, ...]:
        return tuple(result for result in self.results if result.status is RunStatus.FAILED)


@dataclass(slots=True)
class _Progress:
    """Mutable accumulator for a single partition job."""

    fetched: int = 0
    stats: UpsertStats = field(default_factory=UpsertStats)
    newest_reviewed_at: datetime | None = None
    coverage_complete: bool = True

    def record(self, reviews: Sequence[FetchedReview], stats: UpsertStats) -> None:
        self.fetched += len(reviews)
        self.stats += stats
        newest = max(review.reviewed_at for review in reviews)
        if self.newest_reviewed_at is None or newest > self.newest_reviewed_at:
            self.newest_reviewed_at = newest


class IngestionService:
    """Collects reviews for catalog apps into the database."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        sources: Mapping[Store, ReviewSource],
        settings: IngestionSettings,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._sources = sources
        self._settings = settings
        self._clock = clock

    async def ingest_app(
        self,
        app: AppSpec,
        *,
        stores: Collection[Store] | None = None,
        since: datetime | None = None,
    ) -> IngestionReport:
        """Fetch new and edited reviews for every listing and market of ``app``.

        Partitions fail independently: one store being unavailable never discards
        another store's results. Inspect :attr:`IngestionReport.failed` for errors.
        """
        if since is not None and since.tzinfo is None:
            raise ValueError("since must be timezone-aware")

        async with self._session_factory.begin() as session:
            listing_ids = await CatalogRepository(session).sync_app(app)

        jobs = self._plan_jobs(app, listing_ids, stores)
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_fetches)

        async def run_bounded(job: PartitionJob) -> PartitionResult:
            async with semaphore:
                return await self._ingest_partition(job, since)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_bounded(job)) for job in jobs]
        return IngestionReport(app_slug=app.slug, results=tuple(task.result() for task in tasks))

    def _plan_jobs(
        self,
        app: AppSpec,
        listing_ids: Mapping[Store, int],
        stores: Collection[Store] | None,
    ) -> list[PartitionJob]:
        jobs: list[PartitionJob] = []
        for store, external_app_id in app.listings.items():
            if stores is not None and store not in stores:
                continue
            source = self._sources.get(store)
            if source is None:
                logger.warning(
                    "no source configured for store=%s; skipping app=%s", store, app.slug
                )
                continue

            planned_keys: set[str] = set()
            for market in app.iter_markets():
                key = source.partition_key(market)
                if key in planned_keys:
                    continue
                planned_keys.add(key)
                jobs.append(
                    PartitionJob(
                        listing_id=listing_ids[store],
                        external_app_id=external_app_id,
                        source=source,
                        market=market,
                        partition_key=key,
                    )
                )
        return jobs

    async def _ingest_partition(self, job: PartitionJob, since: datetime | None) -> PartitionResult:
        async with self._session_factory.begin() as session:
            state = IngestionStateRepository(session)
            cursor = await state.get_cursor(job.listing_id, job.partition_key)
            cutoff = self._resolve_cutoff(job.source.store, since=since, cursor=cursor)
            run_id = await state.start_run(
                job.listing_id, job.partition_key, cutoff=cutoff, started_at=self._clock()
            )

        logger.info("ingestion started %s cutoff=%s", job, cutoff.isoformat())
        progress = _Progress()
        error: str | None = None
        try:
            await self._consume_pages(job, cutoff, progress)
        except Exception as exc:  # isolate partitions; the failure is persisted on the run
            logger.exception("ingestion failed %s after %d reviews", job, progress.fetched)
            error = f"{type(exc).__name__}: {exc}"

        succeeded = error is None
        status = RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED
        async with self._session_factory.begin() as session:
            state = IngestionStateRepository(session)
            if succeeded and progress.newest_reviewed_at is not None:
                await state.advance_cursor(
                    job.listing_id,
                    job.partition_key,
                    progress.newest_reviewed_at,
                    updated_at=self._clock(),
                )
            await state.finish_run(
                run_id,
                status=status,
                fetched=progress.fetched,
                stats=progress.stats,
                coverage_complete=progress.coverage_complete if succeeded else None,
                error=error,
                finished_at=self._clock(),
            )

        if succeeded:
            self._log_success(job, progress)
        return PartitionResult(
            store=job.source.store,
            partition_key=job.partition_key,
            status=status,
            fetched=progress.fetched,
            stats=progress.stats,
            coverage_complete=succeeded and progress.coverage_complete,
            error=error,
        )

    async def _consume_pages(
        self, job: PartitionJob, cutoff: datetime, progress: _Progress
    ) -> None:
        limit = self._settings.max_reviews_per_market
        pages = job.source.fetch_pages(job.external_app_id, job.market)
        try:
            async for page in pages:
                in_window = [review for review in page.reviews if review.reviewed_at >= cutoff]
                accepted = in_window[: limit - progress.fetched]
                if accepted:
                    await self._store(job.listing_id, accepted, progress)

                if len(accepted) < len(in_window):  # per-market cap reached
                    progress.coverage_complete = False
                    return
                if len(in_window) < len(page.reviews):  # page reaches past the window
                    return
                if page.hit_provider_limit:
                    progress.coverage_complete = False
        finally:
            await pages.aclose()

    async def _store(
        self, listing_id: int, reviews: Sequence[FetchedReview], progress: _Progress
    ) -> None:
        async with self._session_factory.begin() as session:
            stats = await ReviewRepository(session).upsert_batch(
                listing_id, reviews, seen_at=self._clock()
            )
        progress.record(reviews, stats)

    def _resolve_cutoff(
        self, store: Store, *, since: datetime | None, cursor: datetime | None
    ) -> datetime:
        """Lower bound of the review window for a run.

        An explicit ``since`` forces a backfill. Otherwise the run resumes from the cursor
        minus an overlap that absorbs store moderation delays; the first run for a
        partition falls back to the default lookback.
        """
        if since is not None:
            return since
        if cursor is not None:
            return cursor - self._settings.overlap_for(store)
        return self._clock() - timedelta(days=self._settings.default_lookback_days)

    @staticmethod
    def _log_success(job: PartitionJob, progress: _Progress) -> None:
        logger.info(
            "ingestion finished %s fetched=%d inserted=%d updated=%d unchanged=%d",
            job,
            progress.fetched,
            progress.stats.inserted,
            progress.stats.updated,
            progress.stats.unchanged,
        )
        if not progress.coverage_complete:
            logger.warning(
                "coverage gap %s: older in-window reviews were unreachable; "
                "run ingestion more often or raise max_reviews_per_market",
                job,
            )
