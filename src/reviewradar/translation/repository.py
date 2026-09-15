"""Persistence for review translations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, Row, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.db.models import App, AppListing, Review, ReviewEnrichment, ReviewTranslation
from reviewradar.db.upsert import conflict_aware_insert


@dataclass(frozen=True, slots=True)
class TranslationCandidate:
    """The redacted model text of an enriched review, with its resolved language."""

    review_id: int
    text: str
    text_hash: str
    source_language: str | None


@dataclass(frozen=True, slots=True)
class StoredTranslation:
    review_id: int
    source_language: str
    text: str


class TranslationRepository:
    """Finds translation work and stores its results."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def pending(
        self,
        app_slug: str,
        *,
        model: str,
        target_language: str,
        after_id: int,
        limit: int,
        max_rating: int | None = None,
    ) -> list[TranslationCandidate]:
        """Enriched reviews in another known language whose translation is missing or stale.

        A translation is stale when the model text or the resolved source language changed
        since it was produced. ``max_rating`` restricts the work to critical reviews, the
        scope topic discovery labels from English text.
        """
        statement = (
            select(
                ReviewEnrichment.review_id,
                ReviewEnrichment.model_text,
                ReviewEnrichment.text_hash,
                ReviewEnrichment.language,
            )
            .join(Review, Review.id == ReviewEnrichment.review_id)
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .outerjoin(ReviewTranslation, _translation_join(model, target_language))
            .where(
                App.slug == app_slug,
                ReviewEnrichment.review_id > after_id,
                ReviewEnrichment.model_text != "",
                ReviewEnrichment.language.is_not(None),
                ReviewEnrichment.language != target_language,
                or_(
                    ReviewTranslation.review_id.is_(None),
                    ReviewTranslation.text_hash != ReviewEnrichment.text_hash,
                    ReviewTranslation.source_language != ReviewEnrichment.language,
                ),
            )
            .order_by(ReviewEnrichment.review_id)
            .limit(limit)
        )
        if max_rating is not None:
            statement = statement.where(Review.rating <= max_rating)
        return [_candidate(row) for row in await self._session.execute(statement)]

    async def candidates(self, review_ids: Sequence[int]) -> dict[int, TranslationCandidate]:
        """Enrichment of the given reviews; reviews that were never enriched are absent."""
        if not review_ids:
            return {}
        rows = await self._session.execute(
            select(
                ReviewEnrichment.review_id,
                ReviewEnrichment.model_text,
                ReviewEnrichment.text_hash,
                ReviewEnrichment.language,
            ).where(ReviewEnrichment.review_id.in_(list(review_ids)))
        )
        return {candidate.review_id: candidate for candidate in map(_candidate, rows)}

    async def fresh(
        self, review_ids: Sequence[int], *, model: str, target_language: str
    ) -> dict[int, StoredTranslation]:
        """Stored translations that still match the reviews' current text and language."""
        if not review_ids:
            return {}
        rows = await self._session.execute(
            select(
                ReviewTranslation.review_id,
                ReviewTranslation.source_language,
                ReviewTranslation.translated_text,
            )
            .join(ReviewEnrichment, ReviewEnrichment.review_id == ReviewTranslation.review_id)
            .where(
                ReviewTranslation.review_id.in_(list(review_ids)),
                ReviewTranslation.model == model,
                ReviewTranslation.target_language == target_language,
                ReviewTranslation.text_hash == ReviewEnrichment.text_hash,
                ReviewTranslation.source_language == ReviewEnrichment.language,
            )
        )
        return {
            review_id: StoredTranslation(review_id=review_id, source_language=source, text=text)
            for review_id, source, text in rows
        }

    async def upsert(
        self,
        translations: Sequence[tuple[TranslationCandidate, str]],
        *,
        model: str,
        target_language: str,
        created_at: datetime,
    ) -> None:
        if not translations:
            return
        insert = conflict_aware_insert(self._session, ReviewTranslation).values(
            [
                {
                    "review_id": candidate.review_id,
                    "model": model,
                    "target_language": target_language,
                    "source_language": candidate.source_language,
                    "text_hash": candidate.text_hash,
                    "translated_text": text,
                    "created_at": created_at,
                }
                for candidate, text in translations
            ]
        )
        await self._session.execute(
            insert.on_conflict_do_update(
                index_elements=["review_id", "model", "target_language"],
                set_={
                    column: insert.excluded[column]
                    for column in ("source_language", "text_hash", "translated_text", "created_at")
                },
            )
        )


def _translation_join(model: str, target_language: str) -> ColumnElement[bool]:
    return and_(
        ReviewTranslation.review_id == ReviewEnrichment.review_id,
        ReviewTranslation.model == model,
        ReviewTranslation.target_language == target_language,
    )


def _candidate(row: Row[tuple[int, str, str, str | None]]) -> TranslationCandidate:
    review_id, text, text_hash, language = row
    return TranslationCandidate(
        review_id=review_id, text=text, text_hash=text_hash, source_language=language
    )
