"""PostgreSQL retrievers: full-text search and pgvector cosine search.

Both retrievers are exact and apply filters inside the SQL, so a filtered query always
returns the true top-k. At the scale of one app's reviews a scan costs milliseconds; the
path to approximate indexes (and why filters make them tricky) is in the engineering notes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import ColumnClause, Select, cast, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.db.models import App, AppListing, Review, ReviewEmbedding, ReviewEnrichment
from reviewradar.search.types import RankedReview, SearchFilters

TEXT_SEARCH_CONFIG: ColumnClause[str] = literal_column("'simple'::regconfig")
"""No stemming and no stop words. Reviews span dozens of languages, and a per-language
configuration would make tokenization depend on language detection being right."""

TEXT_SEARCH_DOCUMENT = func.to_tsvector(TEXT_SEARCH_CONFIG, ReviewEnrichment.model_text)
"""Must stay identical to the GIN index expression in migration ``0004``."""


class UnsupportedDatabaseError(RuntimeError):
    """Raised when search runs against a backend without full-text and vector support."""


def ensure_postgres(session: AsyncSession) -> None:
    dialect = session.get_bind().dialect.name
    if dialect != "postgresql":
        raise UnsupportedDatabaseError(
            f"search requires PostgreSQL with pgvector; the configured database is {dialect!r}"
        )


class LexicalRetriever:
    """Postgres full-text search over the redacted model text, ranked by cover density."""

    async def retrieve(
        self,
        session: AsyncSession,
        app_slug: str,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[RankedReview]:
        text_query = func.websearch_to_tsquery(TEXT_SEARCH_CONFIG, query)
        rank = func.ts_rank_cd(TEXT_SEARCH_DOCUMENT, text_query)
        statement = (
            _candidates(select(Review.id, rank), app_slug, filters)
            .where(TEXT_SEARCH_DOCUMENT.op("@@")(text_query))
            .order_by(rank.desc(), Review.id)
            .limit(limit)
        )
        rows = await session.execute(statement)
        return [RankedReview(review_id=review_id, score=float(score)) for review_id, score in rows]


class SemanticRetriever:
    """Exact cosine search over one embedding model's vectors with pgvector."""

    def __init__(self, model: str) -> None:
        self._model = model

    async def retrieve(
        self,
        session: AsyncSession,
        app_slug: str,
        query_vector: NDArray[np.float32],
        filters: SearchFilters,
        limit: int,
    ) -> list[RankedReview]:
        dimension = int(query_vector.shape[0])
        distance = cast(ReviewEmbedding.embedding, VECTOR(dimension)).cosine_distance(query_vector)
        statement = (
            _candidates(select(Review.id, distance), app_slug, filters)
            .join(ReviewEmbedding, ReviewEmbedding.review_id == Review.id)
            .where(ReviewEmbedding.model == self._model)
            .order_by(distance, Review.id)
            .limit(limit)
        )
        rows = await session.execute(statement)
        return [
            RankedReview(review_id=review_id, score=1.0 - float(cosine_distance))
            for review_id, cosine_distance in rows
        ]


def _candidates(statement: Select[Any], app_slug: str, filters: SearchFilters) -> Select[Any]:
    """Scope a statement over reviews to one app, its enrichment, and the filters."""
    statement = (
        statement.select_from(Review)
        .join(AppListing, Review.listing_id == AppListing.id)
        .join(App, AppListing.app_id == App.id)
        .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
        .where(App.slug == app_slug)
    )
    if filters.stores:
        statement = statement.where(AppListing.store.in_(sorted(filters.stores)))
    if filters.languages:
        statement = statement.where(ReviewEnrichment.language.in_(sorted(filters.languages)))
    if filters.min_rating is not None:
        statement = statement.where(Review.rating >= filters.min_rating)
    if filters.max_rating is not None:
        statement = statement.where(Review.rating <= filters.max_rating)
    if filters.since is not None:
        statement = statement.where(Review.reviewed_at >= filters.since)
    return statement
