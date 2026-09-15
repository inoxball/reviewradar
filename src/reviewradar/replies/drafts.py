"""Reply drafts for stored reviews: generated on demand, stored, and reused while valid.

A draft is stored per review and generator together with the generator's version (base
model plus adapter, or guidelines fingerprint). Asking again returns the stored draft while
the version matches; retraining the adapter or editing the guidelines produces new drafts.

Models load lazily on the first request. Generation runs in a worker thread behind one
lock, because a single GPU model serves every caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import Catalog, UnknownAppError
from reviewradar.db.models import (
    App,
    AppListing,
    ReplyDraft,
    Review,
    ReviewEmbedding,
    ReviewEnrichment,
)
from reviewradar.db.upsert import conflict_aware_insert
from reviewradar.domain import Store, utc_now
from reviewradar.replies.evaluation import ReplyChecks, check_reply
from reviewradar.replies.generation import ReplyGenerator, ReplyTask
from reviewradar.replies.prompting import ReplyRequest, render_reply


class UnknownGeneratorError(ValueError):
    """Raised when a draft is requested from a generator that is not available."""


class DraftStatus(StrEnum):
    DRAFTED = "drafted"  # generated now and stored
    CACHED = "cached"  # a stored draft from the current generator version
    UNAVAILABLE = "unavailable"  # unknown or unenriched review, or no embedding for retrieval


@dataclass(frozen=True, slots=True)
class DraftCandidate:
    """An enriched review as reply generators see it."""

    review_id: int
    app: str
    store: Store
    rating: int
    language: str | None
    review: str
    embedding: NDArray[np.float32] | None


@dataclass(frozen=True, slots=True)
class StoredDraft:
    review_id: int
    version: str
    text: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DraftResult:
    review_id: int
    generator: str
    status: DraftStatus
    text: str | None = None
    """The reply rendered for the review's app, ready to post."""
    template: str | None = None
    """The reply as generated, with placeholders."""
    checks: ReplyChecks | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReplyGeneratorSet:
    """Loaded generators, the version of each, and which ones need a review embedding."""

    generators: Mapping[str, ReplyGenerator]
    versions: Mapping[str, str]
    requires_embedding: frozenset[str] = field(default_factory=frozenset)


GeneratorsFactory = Callable[[], Awaitable[ReplyGeneratorSet]]


class ReplyDraftRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def candidates(
        self, review_ids: Collection[int], *, embedding_model: str
    ) -> dict[int, DraftCandidate]:
        """Enriched reviews by id, with their current embedding when one exists."""
        if not review_ids:
            return {}
        statement = (
            select(
                Review.id,
                App.slug,
                AppListing.store,
                Review.rating,
                ReviewEnrichment.language,
                ReviewEnrichment.model_text,
                ReviewEmbedding.embedding,
            )
            .select_from(Review)
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .outerjoin(
                ReviewEmbedding,
                and_(
                    ReviewEmbedding.review_id == Review.id,
                    ReviewEmbedding.model == embedding_model,
                    ReviewEmbedding.text_hash == ReviewEnrichment.text_hash,
                ),
            )
            .where(Review.id.in_(sorted(review_ids)))
        )
        rows = await self._session.execute(statement)
        return {
            review_id: DraftCandidate(
                review_id=review_id,
                app=slug,
                store=store,
                rating=rating,
                language=language,
                review=text,
                embedding=None if vector is None else np.asarray(vector, dtype=np.float32),
            )
            for review_id, slug, store, rating, language, text, vector in rows
        }

    async def stored(
        self, review_ids: Collection[int], *, generator: str
    ) -> dict[int, StoredDraft]:
        if not review_ids:
            return {}
        rows = await self._session.execute(
            select(
                ReplyDraft.review_id, ReplyDraft.version, ReplyDraft.text, ReplyDraft.created_at
            ).where(ReplyDraft.review_id.in_(sorted(review_ids)), ReplyDraft.generator == generator)
        )
        return {row.review_id: StoredDraft(*row) for row in rows}

    async def upsert(
        self,
        drafts: Sequence[tuple[int, str]],
        *,
        generator: str,
        version: str,
        created_at: datetime,
    ) -> None:
        if not drafts:
            return
        insert = conflict_aware_insert(self._session, ReplyDraft).values(
            [
                {
                    "review_id": review_id,
                    "generator": generator,
                    "version": version,
                    "text": text,
                    "created_at": created_at,
                }
                for review_id, text in drafts
            ]
        )
        await self._session.execute(
            insert.on_conflict_do_update(
                index_elements=["review_id", "generator"],
                set_={
                    column: insert.excluded[column] for column in ("version", "text", "created_at")
                },
            )
        )


class ReplyDraftService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        catalog: Catalog,
        generators_factory: GeneratorsFactory,
        *,
        available: Sequence[str],
        embedding_model: str,
        detect_language: Callable[[str], str | None],
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._catalog = catalog
        self._generators_factory = generators_factory
        self._available = tuple(available)
        self._embedding_model = embedding_model
        self._detect_language = detect_language
        self._clock = clock
        self._load_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._generator_set: ReplyGeneratorSet | None = None

    @property
    def available_generators(self) -> tuple[str, ...]:
        return self._available

    @property
    def loaded(self) -> bool:
        return self._generator_set is not None

    async def draft(self, review_ids: Sequence[int], generator: str) -> list[DraftResult]:
        """Drafts for specific reviews, generating and storing any that are missing or stale."""
        if generator not in self._available:
            raise UnknownGeneratorError(
                f"generator '{generator}' is not available; choose from {list(self._available)}"
            )
        ids = list(dict.fromkeys(review_ids))
        generators = await self._generators()
        version = generators.versions[generator]
        async with self._session_factory() as session:
            repository = ReplyDraftRepository(session)
            candidates = await repository.candidates(ids, embedding_model=self._embedding_model)
            stored = await repository.stored(ids, generator=generator)

        results: dict[int, DraftResult] = {}
        to_generate: list[DraftCandidate] = []
        for review_id in ids:
            candidate = candidates.get(review_id)
            if candidate is None or not self._usable(candidate, generator, generators):
                results[review_id] = DraftResult(review_id, generator, DraftStatus.UNAVAILABLE)
            elif (draft := stored.get(review_id)) is not None and draft.version == version:
                results[review_id] = self._result(
                    candidate, generator, DraftStatus.CACHED, draft.text, draft.created_at
                )
            else:
                to_generate.append(candidate)

        if to_generate:
            tasks = [
                ReplyTask(
                    ReplyRequest(
                        candidate.store, candidate.rating, candidate.language, candidate.review
                    ),
                    candidate.embedding,
                )
                for candidate in to_generate
            ]
            async with self._model_lock:
                texts = await asyncio.to_thread(generators.generators[generator].generate, tasks)
            created_at = self._clock()
            async with self._session_factory.begin() as session:
                await ReplyDraftRepository(session).upsert(
                    [
                        (candidate.review_id, text)
                        for candidate, text in zip(to_generate, texts, strict=True)
                    ],
                    generator=generator,
                    version=version,
                    created_at=created_at,
                )
            for candidate, text in zip(to_generate, texts, strict=True):
                results[candidate.review_id] = self._result(
                    candidate, generator, DraftStatus.DRAFTED, text, created_at
                )
        return [results[review_id] for review_id in ids]

    async def _generators(self) -> ReplyGeneratorSet:
        async with self._load_lock:
            if self._generator_set is None:
                self._generator_set = await self._generators_factory()
            return self._generator_set

    def _usable(
        self, candidate: DraftCandidate, generator: str, generators: ReplyGeneratorSet
    ) -> bool:
        if not candidate.review.strip():
            return False
        if generator in generators.requires_embedding and candidate.embedding is None:
            return False
        try:
            self._catalog.get(candidate.app)
        except UnknownAppError:
            return False
        return True

    def _result(
        self,
        candidate: DraftCandidate,
        generator: str,
        status: DraftStatus,
        template: str,
        created_at: datetime,
    ) -> DraftResult:
        app = self._catalog.get(candidate.app)
        return DraftResult(
            review_id=candidate.review_id,
            generator=generator,
            status=status,
            text=render_reply(template, app_name=app.name, support_contact=app.support_contact),
            template=template,
            checks=check_reply(
                template,
                review=candidate.review,
                language=candidate.language,
                app_name=app.name,
                support_contact=app.support_contact,
                foreign_brands=[
                    other.name for other in self._catalog.apps if other.slug != app.slug
                ],
                detect_language=self._detect_language,
            ),
            created_at=created_at,
        )
