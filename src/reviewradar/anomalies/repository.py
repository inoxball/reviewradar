"""Reads topic assignments as detector observations."""

from __future__ import annotations

from collections.abc import Collection

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.anomalies.detector import Observation
from reviewradar.db.models import AppListing, Review, ReviewEnrichment, ReviewTopic
from reviewradar.domain import Store


def source_key(store: Store, country: str | None, language: str | None) -> str:
    """Where a review was collected: App Store feeds per storefront, Google Play per language."""
    if store is Store.APP_STORE:
        return f"{store.value}:{country or 'unknown'}"
    return f"{store.value}:{language or 'unknown'}"


class AnomalyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def observations(self, run_id: int, *, max_rating: int | None) -> list[Observation]:
        """Reviews of a topic run with their source, UTC day and app version.

        ``max_rating`` keeps detection on critical reviews even when a topic run covers every
        rating: otherwise shifts in praise ("good", "love it") would be tested as incidents.
        """
        statement = (
            select(
                ReviewTopic.review_id,
                ReviewTopic.topic_id,
                ReviewTopic.distance,
                AppListing.store,
                Review.country,
                func.coalesce(ReviewEnrichment.language, Review.language),
                Review.reviewed_at,
                Review.app_version,
            )
            .select_from(ReviewTopic)
            .join(Review, Review.id == ReviewTopic.review_id)
            .join(AppListing, Review.listing_id == AppListing.id)
            .outerjoin(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .where(ReviewTopic.run_id == run_id)
        )
        if max_rating is not None:
            statement = statement.where(Review.rating <= max_rating)
        rows = await self._session.execute(statement)
        observations: list[Observation] = []
        for review_id, topic_id, distance, store, country, language, reviewed_at, version in rows:
            observations.append(
                Observation(
                    review_id=review_id,
                    topic_id=topic_id,
                    source=source_key(store, country, language),
                    day=reviewed_at.date(),
                    app_version=version,
                    distance=distance,
                )
            )
        return observations

    async def review_texts(self, review_ids: Collection[int]) -> dict[int, str]:
        """The redacted text of reviews, for printing examples."""
        if not review_ids:
            return {}
        rows = await self._session.execute(
            select(ReviewEnrichment.review_id, ReviewEnrichment.model_text).where(
                ReviewEnrichment.review_id.in_(sorted(review_ids))
            )
        )
        return {review_id: text for review_id, text in rows}
