"""Persistence for enrichment results: model text, language and embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.db.models import App, AppListing, Review, ReviewEmbedding, ReviewEnrichment
from reviewradar.db.upsert import conflict_aware_insert
from reviewradar.domain import LanguageSource, Store


@dataclass(frozen=True, slots=True)
class PendingReview:
    """A review whose enrichment is missing or outdated, or whose embedding is stale."""

    review_id: int
    content_hash: str
    title: str | None
    body: str
    country: str | None
    language_hint: str | None
    embedded_text_hash: str | None
    """Fingerprint of the text behind the stored embedding for the current model, if any."""


@dataclass(frozen=True, slots=True)
class EnrichedReview:
    """Derived fields computed for one review revision by one pipeline version.

    Field names mirror ``review_enrichments`` columns.
    """

    review_id: int
    content_hash: str
    enrichment_version: int
    model_text: str
    text_hash: str
    language: str | None
    language_confidence: float | None
    language_candidate: str | None
    language_source: LanguageSource


@dataclass(frozen=True, slots=True)
class LanguageShare:
    """Number of enriched reviews per store and resolved language."""

    store: Store
    language: str | None
    review_count: int


class EnrichmentRepository:
    """Finds enrichment work and stores its results."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def pending_reviews(
        self,
        app_slug: str,
        *,
        model: str,
        enrichment_version: int,
        after_id: int,
        limit: int,
    ) -> list[PendingReview]:
        """Reviews with id above ``after_id`` needing enrichment or a ``model`` embedding.

        Enrichment is stale when missing, computed for an older review revision, or by a
        different pipeline version. An embedding is stale when missing or computed from
        text other than the current model text. Texts without words are never embedded,
        so they do not reappear once enriched.
        """
        enrichment_stale = or_(
            ReviewEnrichment.review_id.is_(None),
            ReviewEnrichment.content_hash != Review.content_hash,
            ReviewEnrichment.enrichment_version != enrichment_version,
        )
        embedding_stale = or_(
            ReviewEmbedding.review_id.is_(None),
            ReviewEmbedding.text_hash != ReviewEnrichment.text_hash,
        )
        statement = (
            select(
                Review.id,
                Review.content_hash,
                Review.title,
                Review.body,
                Review.country,
                Review.language,
                ReviewEmbedding.text_hash,
            )
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .outerjoin(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .outerjoin(
                ReviewEmbedding,
                and_(ReviewEmbedding.review_id == Review.id, ReviewEmbedding.model == model),
            )
            .where(
                App.slug == app_slug,
                Review.id > after_id,
                or_(enrichment_stale, and_(ReviewEnrichment.model_text != "", embedding_stale)),
            )
            .order_by(Review.id)
            .limit(limit)
        )
        rows = await self._session.execute(statement)
        return [
            PendingReview(
                review_id=review_id,
                content_hash=content_hash,
                title=title,
                body=body,
                country=country,
                language_hint=language,
                embedded_text_hash=embedded_text_hash,
            )
            for review_id, content_hash, title, body, country, language, embedded_text_hash in rows
        ]

    async def upsert_enrichments(
        self, enriched: Sequence[EnrichedReview], *, enriched_at: datetime
    ) -> None:
        if not enriched:
            return
        rows = [asdict(review) | {"enriched_at": enriched_at} for review in enriched]
        insert = conflict_aware_insert(self._session, ReviewEnrichment).values(rows)
        await self._session.execute(
            insert.on_conflict_do_update(
                index_elements=["review_id"],
                set_={
                    column: insert.excluded[column] for column in rows[0] if column != "review_id"
                },
            )
        )

    async def upsert_embeddings(
        self,
        model: str,
        embeddings: Sequence[tuple[EnrichedReview, NDArray[np.float32]]],
        *,
        created_at: datetime,
    ) -> None:
        if not embeddings:
            return
        insert = conflict_aware_insert(self._session, ReviewEmbedding).values(
            [
                {
                    "review_id": review.review_id,
                    "model": model,
                    "text_hash": review.text_hash,
                    "embedding": vector,
                    "created_at": created_at,
                }
                for review, vector in embeddings
            ]
        )
        await self._session.execute(
            insert.on_conflict_do_update(
                index_elements=["review_id", "model"],
                set_={
                    column: insert.excluded[column]
                    for column in ("text_hash", "embedding", "created_at")
                },
            )
        )

    async def delete_embeddings(self, review_ids: Sequence[int]) -> None:
        """Drop the vectors (under every model) of reviews that lost their embeddable text."""
        if review_ids:
            await self._session.execute(
                delete(ReviewEmbedding).where(ReviewEmbedding.review_id.in_(review_ids))
            )

    async def language_mix(self, app_slug: str) -> list[LanguageShare]:
        """Enriched review counts per store and resolved language, most common first."""
        review_count = func.count(ReviewEnrichment.review_id)
        statement = (
            select(AppListing.store, ReviewEnrichment.language, review_count)
            .select_from(ReviewEnrichment)
            .join(Review, ReviewEnrichment.review_id == Review.id)
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .where(App.slug == app_slug)
            .group_by(AppListing.store, ReviewEnrichment.language)
            .order_by(AppListing.store, review_count.desc())
        )
        rows = await self._session.execute(statement)
        return [
            LanguageShare(store=store, language=language, review_count=count)
            for store, language, count in rows
        ]
