"""Persistence for topic runs, topics and review assignments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.db.models import (
    App,
    AppListing,
    Review,
    ReviewEmbedding,
    ReviewEnrichment,
    ReviewTopic,
    ReviewTranslation,
    Topic,
    TopicRun,
)

_ASSIGNMENT_CHUNK_SIZE = 1000


@dataclass(frozen=True, slots=True)
class TopicCandidate:
    """An enriched, embedded review eligible for topic discovery."""

    review_id: int
    rating: int
    language: str | None
    english_text: str
    """The stored translation when one exists, otherwise the redacted model text."""
    translated: bool
    embedding: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TopicMemberText:
    """A review of a stored topic, with what is needed to label the topic again."""

    topic_id: int
    review_id: int
    rating: int
    language: str | None
    text: str
    """The redacted review text, in its own language."""
    translation: str | None
    distance: float


@dataclass(frozen=True, slots=True)
class NewTopic:
    label: str
    keywords: list[str]
    size: int
    average_rating: float
    negative_share: float
    languages: dict[str, int]
    members: list[tuple[int, float]]
    """``(review_id, distance)`` pairs, closest to the centroid first."""


@dataclass(frozen=True, slots=True)
class TopicRunView:
    id: int
    embedding_model: str
    algorithm: str
    parameters: dict[str, Any]
    review_count: int
    topic_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TopicView:
    id: int
    run_id: int
    label: str
    keywords: list[str]
    size: int
    average_rating: float
    negative_share: float
    languages: dict[str, int]


class TopicRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def app_id(self, app_slug: str) -> int | None:
        app_id: int | None = await self._session.scalar(select(App.id).where(App.slug == app_slug))
        return app_id

    async def candidates(
        self,
        app_slug: str,
        *,
        embedding_model: str,
        translation_model: str,
        target_language: str,
        max_rating: int | None,
    ) -> list[TopicCandidate]:
        """Reviews with a current embedding, and a current translation where one exists."""
        statement = (
            select(
                Review.id,
                Review.rating,
                ReviewEnrichment.language,
                ReviewEnrichment.model_text,
                ReviewTranslation.translated_text,
                ReviewEmbedding.embedding,
            )
            .select_from(Review)
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .join(
                ReviewEmbedding,
                and_(
                    ReviewEmbedding.review_id == Review.id,
                    ReviewEmbedding.model == embedding_model,
                    ReviewEmbedding.text_hash == ReviewEnrichment.text_hash,
                ),
            )
            .outerjoin(
                ReviewTranslation,
                _current_translation(translation_model, target_language),
            )
            .where(App.slug == app_slug)
            .order_by(Review.id)
        )
        if max_rating is not None:
            statement = statement.where(Review.rating <= max_rating)
        rows = await self._session.execute(statement)
        return [
            TopicCandidate(
                review_id=review_id,
                rating=rating,
                language=language,
                english_text=translation or model_text,
                translated=translation is not None,
                embedding=embedding,
            )
            for review_id, rating, language, model_text, translation, embedding in rows
        ]

    async def member_texts(
        self, run_id: int, *, translation_model: str, target_language: str
    ) -> list[TopicMemberText]:
        """Every review of a run with its text and current translation, closest first per topic."""
        rows = await self._session.execute(
            select(
                ReviewTopic.topic_id,
                ReviewTopic.review_id,
                Review.rating,
                ReviewEnrichment.language,
                ReviewEnrichment.model_text,
                ReviewTranslation.translated_text,
                ReviewTopic.distance,
            )
            .select_from(ReviewTopic)
            .join(Review, Review.id == ReviewTopic.review_id)
            .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .outerjoin(
                ReviewTranslation,
                _current_translation(translation_model, target_language),
            )
            .where(ReviewTopic.run_id == run_id)
            .order_by(ReviewTopic.topic_id, ReviewTopic.distance, ReviewTopic.review_id)
        )
        return [TopicMemberText(*row) for row in rows]

    async def save_run(
        self,
        *,
        app_id: int,
        embedding_model: str,
        algorithm: str,
        parameters: dict[str, Any],
        review_count: int,
        topics: Sequence[NewTopic],
        created_at: datetime,
    ) -> int:
        run = TopicRun(
            app_id=app_id,
            embedding_model=embedding_model,
            algorithm=algorithm,
            parameters=parameters,
            review_count=review_count,
            topic_count=len(topics),
            created_at=created_at,
        )
        self._session.add(run)
        await self._session.flush()

        rows: list[dict[str, Any]] = []
        for new_topic in topics:
            topic = Topic(
                run_id=run.id,
                label=new_topic.label,
                keywords=new_topic.keywords,
                size=new_topic.size,
                average_rating=new_topic.average_rating,
                negative_share=new_topic.negative_share,
                languages=new_topic.languages,
            )
            self._session.add(topic)
            await self._session.flush()
            rows.extend(
                {
                    "run_id": run.id,
                    "review_id": review_id,
                    "topic_id": topic.id,
                    "distance": distance,
                }
                for review_id, distance in new_topic.members
            )
        for start in range(0, len(rows), _ASSIGNMENT_CHUNK_SIZE):
            await self._session.execute(
                insert(ReviewTopic), rows[start : start + _ASSIGNMENT_CHUNK_SIZE]
            )
        return run.id

    async def relabel(
        self,
        run_id: int,
        labels: Mapping[int, tuple[str, list[str]]],
        *,
        naming: dict[str, Any],
    ) -> None:
        """Replace the label and keywords of a run's topics and record how they were named."""
        for topic_id, (label, keywords) in labels.items():
            await self._session.execute(
                update(Topic)
                .where(Topic.id == topic_id, Topic.run_id == run_id)
                .values(label=label, keywords=keywords)
            )
        parameters = await self._session.scalar(
            select(TopicRun.parameters).where(TopicRun.id == run_id)
        )
        await self._session.execute(
            update(TopicRun)
            .where(TopicRun.id == run_id)
            .values(parameters={**(parameters or {}), "naming": naming})
        )

    async def prune_runs(self, app_id: int, *, keep: int) -> int:
        """Delete all but the ``keep`` most recent runs of an app; return how many were removed."""
        recent = (
            select(TopicRun.id)
            .where(TopicRun.app_id == app_id)
            .order_by(TopicRun.created_at.desc(), TopicRun.id.desc())
            .limit(keep)
        )
        stale = list(
            await self._session.scalars(
                select(TopicRun.id).where(TopicRun.app_id == app_id, TopicRun.id.not_in(recent))
            )
        )
        if stale:
            await self._session.execute(delete(TopicRun).where(TopicRun.id.in_(stale)))
        return len(stale)

    async def latest_run(self, app_slug: str) -> TopicRunView | None:
        row = (
            await self._session.execute(
                select(
                    TopicRun.id,
                    TopicRun.embedding_model,
                    TopicRun.algorithm,
                    TopicRun.parameters,
                    TopicRun.review_count,
                    TopicRun.topic_count,
                    TopicRun.created_at,
                )
                .join(App, TopicRun.app_id == App.id)
                .where(App.slug == app_slug)
                .order_by(TopicRun.created_at.desc(), TopicRun.id.desc())
                .limit(1)
            )
        ).first()
        return TopicRunView(*row) if row else None

    async def topics(self, run_id: int) -> list[TopicView]:
        rows = await self._session.execute(
            _topic_columns().where(Topic.run_id == run_id).order_by(Topic.size.desc(), Topic.id)
        )
        return [TopicView(*row) for row in rows]

    async def topic_in_app(self, topic_id: int, app_slug: str) -> TopicView | None:
        row = (
            await self._session.execute(
                _topic_columns()
                .join(TopicRun, Topic.run_id == TopicRun.id)
                .join(App, TopicRun.app_id == App.id)
                .where(Topic.id == topic_id, App.slug == app_slug)
            )
        ).first()
        return TopicView(*row) if row else None

    async def representatives(self, run_id: int, *, per_topic: int) -> dict[int, list[int]]:
        """The review ids closest to each topic's centroid."""
        ranked = (
            select(
                ReviewTopic.topic_id,
                ReviewTopic.review_id,
                func.row_number()
                .over(
                    partition_by=ReviewTopic.topic_id,
                    order_by=(ReviewTopic.distance, ReviewTopic.review_id),
                )
                .label("position"),
            )
            .where(ReviewTopic.run_id == run_id)
            .subquery()
        )
        rows = await self._session.execute(
            select(ranked.c.topic_id, ranked.c.review_id)
            .where(ranked.c.position <= per_topic)
            .order_by(ranked.c.topic_id, ranked.c.position)
        )
        representatives: defaultdict[int, list[int]] = defaultdict(list)
        for topic_id, review_id in rows:
            representatives[topic_id].append(review_id)
        return dict(representatives)

    async def daily_counts(self, run_id: int) -> dict[int, dict[date, int]]:
        """Reviews per UTC day for every topic of a run."""
        rows = await self._session.execute(
            select(ReviewTopic.topic_id, Review.reviewed_at)
            .join(Review, Review.id == ReviewTopic.review_id)
            .where(ReviewTopic.run_id == run_id)
        )
        counts: defaultdict[int, defaultdict[date, int]] = defaultdict(lambda: defaultdict(int))
        for topic_id, reviewed_at in rows:
            counts[topic_id][reviewed_at.date()] += 1
        return {topic_id: dict(days) for topic_id, days in counts.items()}

    async def topic_review_ids(
        self, topic_id: int, *, limit: int, offset: int
    ) -> tuple[int, list[int]]:
        """One page of a topic's review ids, closest to the centroid first, and the total."""
        total = await self._session.scalar(
            select(func.count()).select_from(ReviewTopic).where(ReviewTopic.topic_id == topic_id)
        )
        review_ids = await self._session.scalars(
            select(ReviewTopic.review_id)
            .where(ReviewTopic.topic_id == topic_id)
            .order_by(ReviewTopic.distance, ReviewTopic.review_id)
            .limit(limit)
            .offset(offset)
        )
        return int(total or 0), list(review_ids)


def _current_translation(translation_model: str, target_language: str) -> Any:
    """Join condition for a review's translation that still matches its current text."""
    return and_(
        ReviewTranslation.review_id == Review.id,
        ReviewTranslation.model == translation_model,
        ReviewTranslation.target_language == target_language,
        ReviewTranslation.text_hash == ReviewEnrichment.text_hash,
    )


def _topic_columns() -> Any:
    return select(
        Topic.id,
        Topic.run_id,
        Topic.label,
        Topic.keywords,
        Topic.size,
        Topic.average_rating,
        Topic.negative_share,
        Topic.languages,
    )
