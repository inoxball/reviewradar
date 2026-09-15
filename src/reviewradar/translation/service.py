"""Translates reviews into the target language, in bulk or on demand, and stores the results.

Translations are keyed by review, model and target language, and stay valid while the
review's model text and resolved language are unchanged. Model calls run in a worker
thread, one at a time, because a single GPU model is shared by every caller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import TranslationSettings
from reviewradar.domain import utc_now
from reviewradar.translation.repository import TranslationCandidate, TranslationRepository
from reviewradar.translation.translator import Translator

logger = logging.getLogger(__name__)


class TranslationOutcome(StrEnum):
    TRANSLATED = "translated"  # produced now and stored
    CACHED = "cached"  # a stored translation is still valid
    NOT_NEEDED = "not_needed"  # the review is already in the target language
    UNAVAILABLE = "unavailable"  # not enriched, no text, unknown or unsupported language


@dataclass(frozen=True, slots=True)
class ReviewTranslationResult:
    review_id: int
    outcome: TranslationOutcome
    source_language: str | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationReport:
    app_slug: str
    model: str
    reviews_translated: int
    reviews_unsupported: int
    elapsed_seconds: float

    @property
    def reviews_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.reviews_translated / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class _BatchResult:
    translated: list[tuple[TranslationCandidate, str]]
    unsupported: list[TranslationCandidate]


class TranslationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        translator: Translator,
        settings: TranslationSettings,
        *,
        clock: Callable[[], datetime] = utc_now,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session_factory = session_factory
        self._translator = translator
        self._settings = settings
        self._clock = clock
        self._timer = timer
        self._model_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._translator.model_name

    @property
    def target_language(self) -> str:
        return self._settings.target_language

    async def translate_reviews(self, review_ids: Sequence[int]) -> list[ReviewTranslationResult]:
        """Translations for specific reviews, producing and storing any that are missing."""
        ids = list(dict.fromkeys(review_ids))
        async with self._session_factory() as session:
            repository = TranslationRepository(session)
            candidates = await repository.candidates(ids)
            stored = await repository.fresh(
                ids, model=self.model_name, target_language=self.target_language
            )

        results: dict[int, ReviewTranslationResult] = {}
        to_translate: list[TranslationCandidate] = []
        for review_id in ids:
            candidate = candidates.get(review_id)
            if candidate is None or not candidate.text or candidate.source_language is None:
                results[review_id] = ReviewTranslationResult(
                    review_id, TranslationOutcome.UNAVAILABLE
                )
            elif candidate.source_language == self.target_language:
                results[review_id] = ReviewTranslationResult(
                    review_id, TranslationOutcome.NOT_NEEDED, candidate.source_language
                )
            elif (translation := stored.get(review_id)) is not None:
                results[review_id] = ReviewTranslationResult(
                    review_id,
                    TranslationOutcome.CACHED,
                    translation.source_language,
                    translation.text,
                )
            else:
                to_translate.append(candidate)

        if to_translate:
            batch = await self._translate(to_translate)
            await self._save(batch)
            for candidate, text in batch.translated:
                results[candidate.review_id] = ReviewTranslationResult(
                    candidate.review_id,
                    TranslationOutcome.TRANSLATED,
                    candidate.source_language,
                    text,
                )
            for candidate in batch.unsupported:
                results[candidate.review_id] = ReviewTranslationResult(
                    candidate.review_id, TranslationOutcome.UNAVAILABLE, candidate.source_language
                )
        return [results[review_id] for review_id in ids]

    async def translate_app(
        self, app_slug: str, *, limit: int | None = None, max_rating: int | None = None
    ) -> TranslationReport:
        """Translate every pending review of an app, one committed batch at a time.

        ``max_rating`` limits the work to reviews rated at most that, for example the
        critical reviews topic discovery uses.
        """
        started = self._timer()
        processed = translated = unsupported = 0
        after_id = 0
        while limit is None or processed < limit:
            batch_limit = self._settings.review_batch_size
            if limit is not None:
                batch_limit = min(batch_limit, limit - processed)
            async with self._session_factory() as session:
                pending = await TranslationRepository(session).pending(
                    app_slug,
                    model=self.model_name,
                    target_language=self.target_language,
                    after_id=after_id,
                    limit=batch_limit,
                    max_rating=max_rating,
                )
            if not pending:
                break

            batch = await self._translate(pending)
            await self._save(batch)
            after_id = pending[-1].review_id
            processed += len(pending)
            translated += len(batch.translated)
            unsupported += len(batch.unsupported)
            logger.info(
                "translation progress app=%s processed=%d translated=%d",
                app_slug,
                processed,
                translated,
            )

        return TranslationReport(
            app_slug=app_slug,
            model=self.model_name,
            reviews_translated=translated,
            reviews_unsupported=unsupported,
            elapsed_seconds=self._timer() - started,
        )

    async def _translate(self, candidates: Sequence[TranslationCandidate]) -> _BatchResult:
        async with self._model_lock:
            return await asyncio.to_thread(self._translate_blocking, candidates)

    def _translate_blocking(self, candidates: Sequence[TranslationCandidate]) -> _BatchResult:
        by_language: defaultdict[str, list[TranslationCandidate]] = defaultdict(list)
        unsupported: list[TranslationCandidate] = []
        for candidate in candidates:
            language = candidate.source_language
            if language is not None and self._translator.supports(language):
                by_language[language].append(candidate)
            else:
                unsupported.append(candidate)

        translated: list[tuple[TranslationCandidate, str]] = []
        for language, group in by_language.items():
            texts = self._translator.translate(
                [candidate.text for candidate in group],
                source_language=language,
                target_language=self.target_language,
            )
            translated.extend(zip(group, texts, strict=True))
        return _BatchResult(translated=translated, unsupported=unsupported)

    async def _save(self, batch: _BatchResult) -> None:
        if not batch.translated:
            return
        async with self._session_factory.begin() as session:
            await TranslationRepository(session).upsert(
                batch.translated,
                model=self.model_name,
                target_language=self.target_language,
                created_at=self._clock(),
            )
