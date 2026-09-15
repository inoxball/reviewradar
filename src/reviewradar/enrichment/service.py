"""Enriches stored reviews with redacted model text, language and embeddings.

Work discovery is driven by fingerprints stored next to the results:

* enrichment is recomputed when the review revision (``content_hash``) or the pipeline
  version (:data:`ENRICHMENT_VERSION`) changes;
* an embedding is recomputed only when the model text itself changes, so rating edits,
  developer replies and pipeline upgrades that keep the text intact cost no model time.

Batches are keyset-paginated by review id and committed one at a time; model-bound work
runs in a worker thread so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import AppSpec
from reviewradar.config import EnrichmentSettings
from reviewradar.domain import LanguageSource, utc_now
from reviewradar.enrichment.embeddings import Embedder
from reviewradar.enrichment.language import LanguageDetector, resolve_language
from reviewradar.enrichment.repository import EnrichedReview, EnrichmentRepository, PendingReview
from reviewradar.enrichment.text import prepare_model_text, text_fingerprint

logger = logging.getLogger(__name__)

ENRICHMENT_VERSION = 1
"""Bump whenever text preparation or language resolution changes.

Stored enrichments from other versions are recomputed on the next run; embeddings are
only recomputed for reviews whose model text actually changed.
"""


@dataclass(frozen=True, slots=True)
class EnrichmentReport:
    """Outcome of enriching one app."""

    app_slug: str
    model: str
    reviews_processed: int
    reviews_embedded: int
    reviews_without_text: int
    language_sources: Mapping[LanguageSource, int]
    elapsed_seconds: float

    @property
    def reviews_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.reviews_processed / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class _BatchResult:
    enriched: list[EnrichedReview]
    embedded: list[tuple[EnrichedReview, NDArray[np.float32]]]


class EnrichmentService:
    """Brings enrichment and embeddings up to date with the stored reviews."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        detector: LanguageDetector,
        embedder: Embedder,
        settings: EnrichmentSettings,
        *,
        enrichment_version: int = ENRICHMENT_VERSION,
        clock: Callable[[], datetime] = utc_now,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session_factory = session_factory
        self._detector = detector
        self._embedder = embedder
        self._settings = settings
        self._enrichment_version = enrichment_version
        self._clock = clock
        self._timer = timer

    async def enrich_app(self, app: AppSpec, *, limit: int | None = None) -> EnrichmentReport:
        """Enrich every review of ``app`` that is new or changed since the last run.

        ``limit`` caps the number of reviews processed in this call, which is useful for
        quick experiments on a large backlog.
        """
        started = self._timer()
        market_defaults = app.default_language_by_country()
        processed = embedded = without_text = 0
        sources: Counter[LanguageSource] = Counter()
        after_id = 0

        while limit is None or processed < limit:
            batch_limit = self._settings.review_batch_size
            if limit is not None:
                batch_limit = min(batch_limit, limit - processed)

            async with self._session_factory() as session:
                pending = await EnrichmentRepository(session).pending_reviews(
                    app.slug,
                    model=self._embedder.model_name,
                    enrichment_version=self._enrichment_version,
                    after_id=after_id,
                    limit=batch_limit,
                )
            if not pending:
                break

            result = await asyncio.to_thread(self._process_batch, pending, market_defaults)
            await self._save(result)

            after_id = pending[-1].review_id
            processed += len(pending)
            embedded += len(result.embedded)
            without_text += sum(1 for review in result.enriched if not review.model_text)
            sources.update(review.language_source for review in result.enriched)
            logger.info(
                "enrichment progress app=%s processed=%d embedded=%d", app.slug, processed, embedded
            )

        return EnrichmentReport(
            app_slug=app.slug,
            model=self._embedder.model_name,
            reviews_processed=processed,
            reviews_embedded=embedded,
            reviews_without_text=without_text,
            language_sources=dict(sources),
            elapsed_seconds=self._timer() - started,
        )

    def _process_batch(
        self, pending: Sequence[PendingReview], market_defaults: Mapping[str, str]
    ) -> _BatchResult:
        """Model-bound part of a batch; runs in a worker thread."""
        enriched = [self._enrich(review, market_defaults) for review in pending]
        to_embed = [
            result
            for review, result in zip(pending, enriched, strict=True)
            if result.model_text and result.text_hash != review.embedded_text_hash
        ]
        if not to_embed:
            return _BatchResult(enriched=enriched, embedded=[])

        vectors = self._embedder.embed_documents([review.model_text for review in to_embed])
        return _BatchResult(enriched=enriched, embedded=list(zip(to_embed, vectors, strict=True)))

    def _enrich(self, review: PendingReview, market_defaults: Mapping[str, str]) -> EnrichedReview:
        text = prepare_model_text(review.title, review.body)
        guess = resolve_language(
            text,
            self._detector,
            hint=review.language_hint,
            market_default=market_defaults.get(review.country) if review.country else None,
            min_confidence=self._settings.min_language_confidence,
            min_corroborated_confidence=self._settings.min_corroborated_language_confidence,
            min_chars=self._settings.min_language_chars,
        )
        return EnrichedReview(
            review_id=review.review_id,
            content_hash=review.content_hash,
            enrichment_version=self._enrichment_version,
            model_text=text,
            text_hash=text_fingerprint(text),
            language=guess.language,
            language_confidence=guess.confidence,
            language_candidate=guess.candidate,
            language_source=guess.source,
        )

    async def _save(self, result: _BatchResult) -> None:
        now = self._clock()
        async with self._session_factory.begin() as session:
            repository = EnrichmentRepository(session)
            await repository.upsert_enrichments(result.enriched, enriched_at=now)
            await repository.upsert_embeddings(
                self._embedder.model_name, result.embedded, created_at=now
            )
            await repository.delete_embeddings(
                [review.review_id for review in result.enriched if not review.model_text]
            )
