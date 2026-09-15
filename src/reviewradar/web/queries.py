"""Read models for the panel: aggregate and browsing queries that no pipeline needs.

Writes stay inside their pipelines; these queries only read, and are scoped to one app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import ColumnElement, Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.db.migrate import current_revision, head_revision
from reviewradar.db.models import (
    App,
    AppListing,
    IngestionCursor,
    IngestionRun,
    Review,
    ReviewEmbedding,
    ReviewEnrichment,
    ReviewTranslation,
)
from reviewradar.domain import LanguageSource, RunStatus, Store
from reviewradar.search.types import SearchFilters


class ReviewSort(StrEnum):
    NEWEST = "newest"
    OLDEST = "oldest"
    LOWEST_RATING = "lowest_rating"
    HIGHEST_RATING = "highest_rating"


@dataclass(frozen=True, slots=True)
class RatingCount:
    store: Store
    rating: int
    review_count: int


@dataclass(frozen=True, slots=True)
class PipelineCounts:
    reviews: int
    enriched: int
    embedded: int
    translated: int
    language_sources: dict[LanguageSource, int]


@dataclass(frozen=True, slots=True)
class IngestionRunView:
    id: int
    store: Store
    partition_key: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    fetched: int
    inserted: int
    updated: int
    unchanged: int
    coverage_complete: bool | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CursorView:
    store: Store
    partition_key: str
    newest_reviewed_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ReviewView:
    review_id: int
    store: Store
    external_id: str
    rating: int
    country: str | None
    language: str | None
    app_version: str | None
    reviewed_at: datetime
    text: str


@dataclass(frozen=True, slots=True)
class DatabaseInfo:
    backend: str
    server_version: str
    pgvector: str | None
    revision: str | None
    head_revision: str

    @property
    def migrations_current(self) -> bool:
        return self.revision == self.head_revision


class PanelQueries:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def database_info(self) -> DatabaseInfo:
        """Backend, versions, and whether the schema is at the newest bundled migration."""
        backend = self._session.get_bind().dialect.name
        revision = await self._session.run_sync(
            lambda session: current_revision(session.connection())
        )
        pgvector: str | None = None
        if backend == "postgresql":
            version = await self._session.scalar(text("SHOW server_version"))
            pgvector = await self._session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
        else:
            version = await self._session.scalar(text("SELECT sqlite_version()"))
        return DatabaseInfo(backend, str(version), pgvector, revision, head_revision())

    async def rating_distribution(self, app_slug: str) -> list[RatingCount]:
        review_count = func.count(Review.id)
        statement = _in_app(
            select(AppListing.store, Review.rating, review_count).select_from(Review), app_slug
        ).group_by(AppListing.store, Review.rating)
        rows = await self._session.execute(statement.order_by(AppListing.store, Review.rating))
        return [RatingCount(store, rating, count) for store, rating, count in rows]

    async def pipeline_counts(
        self,
        app_slug: str,
        *,
        embedding_model: str,
        translation_model: str,
        target_language: str,
    ) -> PipelineCounts:
        """How far each pipeline stage has progressed for an app."""
        reviews = await self._count(select(func.count(Review.id)).select_from(Review), app_slug)
        embedded = await self._count(
            select(func.count(ReviewEmbedding.review_id))
            .select_from(ReviewEmbedding)
            .join(Review, Review.id == ReviewEmbedding.review_id)
            .where(ReviewEmbedding.model == embedding_model),
            app_slug,
        )
        translated = await self._count(
            select(func.count(ReviewTranslation.review_id))
            .select_from(ReviewTranslation)
            .join(Review, Review.id == ReviewTranslation.review_id)
            .where(
                ReviewTranslation.model == translation_model,
                ReviewTranslation.target_language == target_language,
            ),
            app_slug,
        )
        source_rows = await self._session.execute(
            _in_app(
                select(ReviewEnrichment.language_source, func.count(ReviewEnrichment.review_id))
                .select_from(ReviewEnrichment)
                .join(Review, Review.id == ReviewEnrichment.review_id),
                app_slug,
            ).group_by(ReviewEnrichment.language_source)
        )
        sources = {source: count for source, count in source_rows}
        return PipelineCounts(
            reviews=reviews,
            enriched=sum(sources.values()),
            embedded=embedded,
            translated=translated,
            language_sources=sources,
        )

    async def recent_runs(self, app_slug: str, *, limit: int) -> list[IngestionRunView]:
        statement = (
            select(
                IngestionRun.id,
                AppListing.store,
                IngestionRun.partition_key,
                IngestionRun.status,
                IngestionRun.started_at,
                IngestionRun.finished_at,
                IngestionRun.fetched,
                IngestionRun.inserted,
                IngestionRun.updated,
                IngestionRun.unchanged,
                IngestionRun.coverage_complete,
                IngestionRun.error,
            )
            .join(AppListing, IngestionRun.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .where(App.slug == app_slug)
            .order_by(IngestionRun.started_at.desc(), IngestionRun.id.desc())
            .limit(limit)
        )
        return [IngestionRunView(*row) for row in await self._session.execute(statement)]

    async def cursors(self, app_slug: str) -> list[CursorView]:
        statement = (
            select(
                AppListing.store,
                IngestionCursor.partition_key,
                IngestionCursor.newest_reviewed_at,
                IngestionCursor.updated_at,
            )
            .join(AppListing, IngestionCursor.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .where(App.slug == app_slug)
            .order_by(AppListing.store, IngestionCursor.partition_key)
        )
        return [CursorView(*row) for row in await self._session.execute(statement)]

    async def browse_reviews(
        self,
        app_slug: str,
        *,
        filters: SearchFilters,
        sort: ReviewSort,
        limit: int,
        offset: int,
    ) -> tuple[int, list[ReviewView]]:
        """One page of reviews plus the total matching count.

        Shows the redacted model text once a review is enriched, and the stored body before.
        """
        language = _REVIEW_LANGUAGE
        statement = _review_rows(app_slug)

        if filters.stores:
            statement = statement.where(AppListing.store.in_(sorted(filters.stores)))
        if filters.languages:
            statement = statement.where(language.in_(sorted(filters.languages)))
        if filters.min_rating is not None:
            statement = statement.where(Review.rating >= filters.min_rating)
        if filters.max_rating is not None:
            statement = statement.where(Review.rating <= filters.max_rating)
        if filters.since is not None:
            statement = statement.where(Review.reviewed_at >= filters.since)

        total = await self._session.scalar(select(func.count()).select_from(statement.subquery()))
        orderings: dict[ReviewSort, tuple[ColumnElement[Any], ...]] = {
            ReviewSort.NEWEST: (Review.reviewed_at.desc(), Review.id.desc()),
            ReviewSort.OLDEST: (Review.reviewed_at.asc(), Review.id.asc()),
            ReviewSort.LOWEST_RATING: (Review.rating.asc(), Review.reviewed_at.desc()),
            ReviewSort.HIGHEST_RATING: (Review.rating.desc(), Review.reviewed_at.desc()),
        }
        ordering = orderings[sort]
        rows = await self._session.execute(
            statement.order_by(*ordering).limit(limit).offset(offset)
        )
        return int(total or 0), [ReviewView(*row) for row in rows]

    async def reviews_by_ids(self, app_slug: str, review_ids: list[int]) -> dict[int, ReviewView]:
        """Reviews of one app keyed by id, shown the way ``browse_reviews`` shows them."""
        if not review_ids:
            return {}
        rows = await self._session.execute(_review_rows(app_slug).where(Review.id.in_(review_ids)))
        return {row.review_id: row for row in (ReviewView(*row) for row in rows)}

    async def _count(self, statement: Select[Any], app_slug: str) -> int:
        return int(await self._session.scalar(_in_app(statement, app_slug)) or 0)


_REVIEW_TEXT = func.coalesce(ReviewEnrichment.model_text, Review.body)
"""The redacted model text once a review is enriched, the stored body before."""
_REVIEW_LANGUAGE = func.coalesce(ReviewEnrichment.language, Review.language)


def _review_rows(app_slug: str) -> Select[Any]:
    """One app's reviews in the columns of :class:`ReviewView`."""
    return _in_app(
        select(
            Review.id,
            AppListing.store,
            Review.external_id,
            Review.rating,
            Review.country,
            _REVIEW_LANGUAGE,
            Review.app_version,
            Review.reviewed_at,
            _REVIEW_TEXT,
        ).select_from(Review),
        app_slug,
    ).outerjoin(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)


def _in_app(statement: Select[Any], app_slug: str) -> Select[Any]:
    """Scope a statement that already selects from ``reviews`` to one app."""
    return (
        statement.join(AppListing, Review.listing_id == AppListing.id)
        .join(App, AppListing.app_id == App.id)
        .where(App.slug == app_slug)
    )
