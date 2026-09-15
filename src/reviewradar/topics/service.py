"""Discovers topics in an app's reviews and stores them as a run.

Discovery clusters current review embeddings, labels every cluster with its most
distinctive English keywords, and records per-topic statistics. Keywords come only from
English text (translations, or reviews written in English): untranslated reviews would put
Portuguese or Spanish words into labels. Each run is a snapshot; only the most recent runs
are kept.

Relabelling recomputes the latest run's keywords and labels without clustering again,
optionally with names written by a local language model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import AppSpec
from reviewradar.config import TopicSettings
from reviewradar.domain import utc_now
from reviewradar.topics.clustering import Clusterer
from reviewradar.topics.keywords import keyword_label, topic_keywords
from reviewradar.topics.naming import TopicDescription, TopicNamer, resolve_labels
from reviewradar.topics.repository import (
    NewTopic,
    TopicCandidate,
    TopicMemberText,
    TopicRepository,
    TopicView,
)

logger = logging.getLogger(__name__)

NEGATIVE_RATING = 2


class InsufficientReviewsError(RuntimeError):
    """Raised when too few eligible reviews exist to discover meaningful topics."""


class NoTopicRunError(LookupError):
    """Raised when an app has no topic run to relabel."""


@dataclass(frozen=True, slots=True)
class TopicScope:
    """Which reviews take part in discovery."""

    max_rating: int | None = 3
    min_words: int = 4

    def __post_init__(self) -> None:
        if self.max_rating is not None and not 1 <= self.max_rating <= 5:
            raise ValueError(f"max_rating must be within 1..5, got {self.max_rating}")
        if self.min_words < 1:
            raise ValueError(f"min_words must be positive, got {self.min_words}")


@dataclass(frozen=True, slots=True)
class TopicSummary:
    label: str
    size: int
    average_rating: float
    negative_share: float


@dataclass(frozen=True, slots=True)
class TopicRunReport:
    app_slug: str
    run_id: int
    review_count: int
    topics: tuple[TopicSummary, ...]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RelabelReport:
    app_slug: str
    run_id: int
    model: str | None
    """The naming model, or None for keyword labels."""
    topics: tuple[TopicSummary, ...]
    fallbacks: int
    """Topics that kept their keyword label because the model's answer was unusable."""
    elapsed_seconds: float


class TopicService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        clusterer_factory: Callable[[int], Clusterer],
        settings: TopicSettings,
        *,
        embedding_model: str,
        translation_model: str,
        target_language: str,
        clock: Callable[[], datetime] = utc_now,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session_factory = session_factory
        self._clusterer_factory = clusterer_factory
        self._settings = settings
        self._embedding_model = embedding_model
        self._translation_model = translation_model
        self._target_language = target_language
        self._clock = clock
        self._timer = timer

    @property
    def default_scope(self) -> TopicScope:
        return TopicScope(max_rating=self._settings.max_rating, min_words=self._settings.min_words)

    async def discover(
        self,
        app: AppSpec,
        *,
        scope: TopicScope | None = None,
        topic_count: int | None = None,
    ) -> TopicRunReport:
        """Cluster the app's eligible reviews into topics and store the run.

        ``scope`` and ``topic_count`` default to the configured settings.
        """
        started = self._timer()
        scope = scope or self.default_scope
        async with self._session_factory() as session:
            repository = TopicRepository(session)
            app_id = await repository.app_id(app.slug)
            candidates = await repository.candidates(
                app.slug,
                embedding_model=self._embedding_model,
                translation_model=self._translation_model,
                target_language=self._target_language,
                max_rating=scope.max_rating,
            )

        eligible = [
            candidate
            for candidate in candidates
            if len(candidate.english_text.split()) >= scope.min_words
        ]
        if app_id is None or len(eligible) < self._settings.min_reviews:
            raise InsufficientReviewsError(
                f"topic discovery needs at least {self._settings.min_reviews} eligible reviews "
                f"for '{app.slug}', found {len(eligible)}; "
                "ingest and enrich first, or widen the scope"
            )

        clusterer = self._clusterer_factory(topic_count or self._settings.topic_count)
        topics = await asyncio.to_thread(
            self._build_topics, clusterer, eligible, _app_stop_words(app)
        )
        async with self._session_factory.begin() as session:
            repository = TopicRepository(session)
            run_id = await repository.save_run(
                app_id=app_id,
                embedding_model=self._embedding_model,
                algorithm=clusterer.name,
                parameters={
                    **clusterer.parameters(),
                    "max_rating": scope.max_rating,
                    "min_words": scope.min_words,
                },
                review_count=len(eligible),
                topics=topics,
                created_at=self._clock(),
            )
            pruned = await repository.prune_runs(app_id, keep=self._settings.runs_to_keep)
        logger.info(
            "topic run app=%s run=%d reviews=%d topics=%d pruned=%d",
            app.slug,
            run_id,
            len(eligible),
            len(topics),
            pruned,
        )

        return TopicRunReport(
            app_slug=app.slug,
            run_id=run_id,
            review_count=len(eligible),
            topics=tuple(
                TopicSummary(topic.label, topic.size, topic.average_rating, topic.negative_share)
                for topic in topics
            ),
            elapsed_seconds=self._timer() - started,
        )

    async def relabel(self, app: AppSpec, namer: TopicNamer | None = None) -> RelabelReport:
        """Recompute the latest run's keywords and labels, without clustering again.

        The same reviews stay in the same topics. Labels are the strongest English keywords,
        or, with a ``namer``, names written by a model that reads each topic's keywords and
        most typical reviews.
        """
        started = self._timer()
        async with self._session_factory() as session:
            repository = TopicRepository(session)
            run = await repository.latest_run(app.slug)
            if run is None:
                raise NoTopicRunError(
                    f"'{app.slug}' has no topic run; run `reviewradar topics {app.slug}` first"
                )
            topics = await repository.topics(run.id)
            members = await repository.member_texts(
                run.id,
                translation_model=self._translation_model,
                target_language=self._target_language,
            )

        by_topic: defaultdict[int, list[TopicMemberText]] = defaultdict(list)
        for member in members:
            by_topic[member.topic_id].append(member)
        keywords, names = await asyncio.to_thread(self._label_topics, app, namer, topics, by_topic)
        labels = resolve_labels(names, keywords)
        fallbacks = sum(name is None for name in names) if namer is not None else 0

        async with self._session_factory.begin() as session:
            await TopicRepository(session).relabel(
                run.id,
                {
                    topic.id: (label, terms)
                    for topic, label, terms in zip(topics, labels, keywords, strict=True)
                },
                naming={
                    "model": namer.model_name if namer is not None else None,
                    "labelled_at": self._clock().isoformat(),
                    "fallbacks": fallbacks,
                },
            )
        logger.info(
            "relabelled topics app=%s run=%d topics=%d model=%s fallbacks=%d",
            app.slug,
            run.id,
            len(topics),
            namer.model_name if namer is not None else "keywords",
            fallbacks,
        )
        return RelabelReport(
            app_slug=app.slug,
            run_id=run.id,
            model=namer.model_name if namer is not None else None,
            topics=tuple(
                TopicSummary(label, topic.size, topic.average_rating, topic.negative_share)
                for topic, label in zip(topics, labels, strict=True)
            ),
            fallbacks=fallbacks,
            elapsed_seconds=self._timer() - started,
        )

    def _build_topics(
        self,
        clusterer: Clusterer,
        candidates: Sequence[TopicCandidate],
        stop_words: set[str],
    ) -> list[NewTopic]:
        """Model-bound part of discovery; runs in a worker thread."""
        vectors = np.stack([candidate.embedding for candidate in candidates]).astype(np.float32)
        clustering = clusterer.fit(vectors)
        clusters = [
            _closest_first(np.flatnonzero(clustering.labels == cluster), clustering.distances)
            for cluster in sorted(set(clustering.labels.tolist()))
        ]
        keywords = self._keywords(
            [
                [
                    candidates[index].english_text
                    for index in members
                    if candidates[index].translated
                    or candidates[index].language == self._target_language
                ]
                for members in clusters
            ],
            stop_words,
        )

        topics: list[NewTopic] = []
        for members, terms in zip(clusters, keywords, strict=True):
            ratings = [candidates[index].rating for index in members]
            languages = Counter(candidates[index].language or "unknown" for index in members)
            topics.append(
                NewTopic(
                    label=keyword_label(terms),
                    keywords=terms,
                    size=len(members),
                    average_rating=sum(ratings) / len(ratings),
                    negative_share=sum(rating <= NEGATIVE_RATING for rating in ratings)
                    / len(ratings),
                    languages=dict(languages.most_common()),
                    members=[
                        (candidates[index].review_id, float(clustering.distances[index]))
                        for index in members
                    ],
                )
            )
        topics.sort(key=lambda topic: topic.size, reverse=True)
        return topics

    def _label_topics(
        self,
        app: AppSpec,
        namer: TopicNamer | None,
        topics: Sequence[TopicView],
        members: dict[int, list[TopicMemberText]],
    ) -> tuple[list[list[str]], list[str | None]]:
        """Keywords, and model names when a namer is given; runs in a worker thread."""
        keywords = self._keywords(
            [
                [
                    member.translation or member.text
                    for member in members[topic.id]
                    if member.translation or member.language == self._target_language
                ]
                for topic in topics
            ],
            _app_stop_words(app),
        )
        if namer is None:
            return keywords, [None] * len(topics)
        descriptions = [
            TopicDescription(
                keywords=terms,
                average_rating=topic.average_rating,
                examples=[
                    member.translation or member.text
                    for member in members[topic.id][: self._settings.name_examples]
                ],
            )
            for topic, terms in zip(topics, keywords, strict=True)
        ]
        return keywords, namer.name(app.name, descriptions)

    def _keywords(self, english_texts: list[list[str]], stop_words: set[str]) -> list[list[str]]:
        return topic_keywords(
            english_texts,
            top_n=self._settings.keywords_per_topic,
            extra_stop_words=stop_words,
        )


def _closest_first(members: NDArray[np.intp], distances: NDArray[np.float32]) -> list[int]:
    """Member indices ordered by distance to their centroid."""
    order = np.argsort(distances[members], kind="stable")
    return [int(index) for index in members[order]]


def _app_stop_words(app: AppSpec) -> set[str]:
    """The app's own name says nothing about what a review of it is about."""
    return {word.lower() for word in app.name.split()} | {app.slug.lower()}
