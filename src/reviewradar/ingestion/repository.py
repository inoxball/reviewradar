"""Persistence for catalog entries, reviews and ingestion bookkeeping.

Repositories operate on a caller-provided ``AsyncSession`` and never commit:
transaction boundaries belong to the service, which keeps units of work explicit.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from reviewradar.catalog import AppSpec
from reviewradar.db.models import App, AppListing, IngestionCursor, IngestionRun, Review
from reviewradar.db.upsert import conflict_aware_insert
from reviewradar.domain import FetchedReview, RunStatus, Store

_INSERT_CHUNK_SIZE = 500


@dataclass(frozen=True, slots=True)
class UpsertStats:
    """Outcome counts of writing a batch of reviews."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.unchanged

    def __add__(self, other: UpsertStats) -> UpsertStats:
        return UpsertStats(
            inserted=self.inserted + other.inserted,
            updated=self.updated + other.updated,
            unchanged=self.unchanged + other.unchanged,
        )


@dataclass(frozen=True, slots=True)
class MarketSummary:
    """Aggregate view of stored reviews for one store and country."""

    store: Store
    country: str | None
    review_count: int
    average_rating: float
    oldest_reviewed_at: datetime
    newest_reviewed_at: datetime


class CatalogRepository:
    """Mirrors the declarative app catalog into the database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def sync_app(self, spec: AppSpec) -> dict[Store, int]:
        """Create or update an app and its listings; return listing ids keyed by store.

        Listings dropped from the catalog are kept, so their review history survives.
        """
        app = await self._session.scalar(
            select(App).where(App.slug == spec.slug).options(selectinload(App.listings))
        )
        if app is None:
            app = App(slug=spec.slug, name=spec.name, listings=[])
            self._session.add(app)
        app.name = spec.name

        listings_by_store = {listing.store: listing for listing in app.listings}
        for store, external_id in spec.listings.items():
            if (listing := listings_by_store.get(store)) is None:
                app.listings.append(AppListing(store=store, external_id=external_id))
            else:
                listing.external_id = external_id

        await self._session.flush()
        return {listing.store: listing.id for listing in app.listings}


class ReviewRepository:
    """Reads and writes reviews."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_batch(
        self, listing_id: int, reviews: Sequence[FetchedReview], *, seen_at: datetime
    ) -> UpsertStats:
        """Insert new reviews, update edited ones and refresh ``last_seen_at`` on the rest.

        Safe under concurrent writers to the same listing (overlapping partitions, or two
        ingestion runs racing): inserts skip rows that already exist, and those rows are
        counted as unchanged.
        """
        latest = {review.external_id: review for review in reviews}  # pages may repeat a review
        if not latest:
            return UpsertStats()

        existing = await self._existing_hashes(listing_id, latest.keys())
        new: list[FetchedReview] = []
        changed: list[tuple[int, FetchedReview]] = []
        unchanged_ids: list[int] = []
        for external_id, review in latest.items():
            match existing.get(external_id):
                case None:
                    new.append(review)
                case (review_id, stored_hash) if stored_hash == review.content_hash:
                    unchanged_ids.append(review_id)
                case (review_id, _):
                    changed.append((review_id, review))

        inserted = await self._insert_new(listing_id, new, seen_at)
        if changed:
            await self._session.execute(
                update(Review),
                [
                    {"id": review_id, "last_seen_at": seen_at, **_mutable_columns(review)}
                    for review_id, review in changed
                ],
            )
        if unchanged_ids:
            await self._session.execute(
                update(Review).where(Review.id.in_(unchanged_ids)).values(last_seen_at=seen_at)
            )

        return UpsertStats(
            inserted=inserted,
            updated=len(changed),
            unchanged=len(unchanged_ids) + len(new) - inserted,
        )

    async def summarize_app(self, app_slug: str) -> list[MarketSummary]:
        """Review counts, average rating and date range per store and country."""
        review_count = func.count(Review.id)
        statement = (
            select(
                AppListing.store,
                Review.country,
                review_count,
                func.avg(Review.rating),
                func.min(Review.reviewed_at),
                func.max(Review.reviewed_at),
            )
            .select_from(Review)
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .where(App.slug == app_slug)
            .group_by(AppListing.store, Review.country)
            .order_by(AppListing.store, review_count.desc())
        )
        rows = await self._session.execute(statement)
        return [
            MarketSummary(
                store=store,
                country=country,
                review_count=count,
                average_rating=float(average),
                oldest_reviewed_at=oldest,
                newest_reviewed_at=newest,
            )
            for store, country, count, average, oldest, newest in rows
        ]

    async def _existing_hashes(
        self, listing_id: int, external_ids: Collection[str]
    ) -> dict[str, tuple[int, str]]:
        rows = await self._session.execute(
            select(Review.external_id, Review.id, Review.content_hash).where(
                Review.listing_id == listing_id, Review.external_id.in_(list(external_ids))
            )
        )
        return {external_id: (review_id, digest) for external_id, review_id, digest in rows}

    async def _insert_new(
        self, listing_id: int, reviews: Sequence[FetchedReview], seen_at: datetime
    ) -> int:
        """Insert rows, ignoring unique-key conflicts; return how many were actually inserted."""
        if not reviews:
            return 0

        inserted = 0
        for chunk in _chunked(reviews, _INSERT_CHUNK_SIZE):
            statement = (
                conflict_aware_insert(self._session, Review)
                .values([_new_row(listing_id, review, seen_at) for review in chunk])
                .on_conflict_do_nothing(index_elements=["listing_id", "external_id"])
                .returning(Review.id)
            )
            result = await self._session.execute(statement)
            inserted += len(result.all())
        return inserted


class IngestionStateRepository:
    """Cursors and run records that make ingestion incremental and observable."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_cursor(self, listing_id: int, partition_key: str) -> datetime | None:
        newest_reviewed_at: datetime | None = await self._session.scalar(
            select(IngestionCursor.newest_reviewed_at).where(
                IngestionCursor.listing_id == listing_id,
                IngestionCursor.partition_key == partition_key,
            )
        )
        return newest_reviewed_at

    async def advance_cursor(
        self,
        listing_id: int,
        partition_key: str,
        newest_reviewed_at: datetime,
        *,
        updated_at: datetime,
    ) -> None:
        """Move the cursor forward; never backwards (e.g. after a narrow backfill)."""
        cursor = await self._session.get(IngestionCursor, (listing_id, partition_key))
        if cursor is None:
            self._session.add(
                IngestionCursor(
                    listing_id=listing_id,
                    partition_key=partition_key,
                    newest_reviewed_at=newest_reviewed_at,
                    updated_at=updated_at,
                )
            )
        elif newest_reviewed_at > cursor.newest_reviewed_at:
            cursor.newest_reviewed_at = newest_reviewed_at
            cursor.updated_at = updated_at

    async def start_run(
        self, listing_id: int, partition_key: str, *, cutoff: datetime, started_at: datetime
    ) -> int:
        run = IngestionRun(
            listing_id=listing_id,
            partition_key=partition_key,
            status=RunStatus.RUNNING,
            cutoff=cutoff,
            started_at=started_at,
            fetched=0,
            inserted=0,
            updated=0,
            unchanged=0,
        )
        self._session.add(run)
        await self._session.flush()
        return run.id

    async def finish_run(
        self,
        run_id: int,
        *,
        status: RunStatus,
        fetched: int,
        stats: UpsertStats,
        coverage_complete: bool | None,
        error: str | None,
        finished_at: datetime,
    ) -> None:
        await self._session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                status=status,
                finished_at=finished_at,
                fetched=fetched,
                inserted=stats.inserted,
                updated=stats.updated,
                unchanged=stats.unchanged,
                coverage_complete=coverage_complete,
                error=error,
            )
        )


def _mutable_columns(review: FetchedReview) -> dict[str, Any]:
    """Columns a store may change after first publication: edits, replies, votes."""
    return {
        "rating": review.rating,
        "title": review.title,
        "body": review.body,
        "app_version": review.app_version,
        "helpful_count": review.helpful_count,
        "developer_reply": review.developer_reply,
        "developer_replied_at": review.developer_replied_at,
        "reviewed_at": review.reviewed_at,
        "content_hash": review.content_hash,
        "raw": dict(review.raw),
    }


def _new_row(listing_id: int, review: FetchedReview, seen_at: datetime) -> dict[str, Any]:
    return {
        "listing_id": listing_id,
        "external_id": review.external_id,
        "country": review.country,
        "language": review.language,
        "author_hash": review.author_hash,
        "first_seen_at": seen_at,
        "last_seen_at": seen_at,
        **_mutable_columns(review),
    }


def _chunked[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
