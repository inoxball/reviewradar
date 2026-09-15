"""Answers review search queries in lexical, semantic or hybrid mode."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.config import SearchSettings
from reviewradar.db.models import AppListing, Review, ReviewEnrichment
from reviewradar.enrichment.embeddings import Embedder
from reviewradar.search.fusion import reciprocal_rank_fusion
from reviewradar.search.retrievers import LexicalRetriever, SemanticRetriever, ensure_postgres
from reviewradar.search.types import RankedReview, SearchFilters, SearchMode, SearchResult


class SearchService:
    """Retrieves reviews for a free-text query.

    Hybrid mode retrieves a candidate pool from each retriever and fuses the rankings with
    reciprocal rank fusion. Lexical search contributes exact terms (product names, error
    codes); semantic search contributes paraphrases and other languages.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        settings: SearchSettings,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._settings = settings
        self._lexical = LexicalRetriever()
        self._semantic = SemanticRetriever(embedder.model_name)

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    async def search(
        self,
        app_slug: str,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError(f"limit must be positive, got {limit}")
        filters = filters or SearchFilters()
        pool = max(limit, self._settings.candidate_pool_size)

        async with self._session_factory() as session:
            ensure_postgres(session)
            rankings: list[list[RankedReview]] = []
            if mode in (SearchMode.LEXICAL, SearchMode.HYBRID):
                rankings.append(
                    await self._lexical.retrieve(session, app_slug, query, filters, pool)
                )
            if mode in (SearchMode.SEMANTIC, SearchMode.HYBRID):
                query_vector = await asyncio.to_thread(self._embedder.embed_query, query)
                rankings.append(
                    await self._semantic.retrieve(session, app_slug, query_vector, filters, pool)
                )

            if len(rankings) == 1:
                ranked = rankings[0][:limit]
            else:
                ranked = reciprocal_rank_fusion(
                    [[hit.review_id for hit in ranking] for ranking in rankings],
                    k=self._settings.rrf_k,
                )[:limit]
            return await self._hydrate(session, ranked)

    async def _hydrate(
        self, session: AsyncSession, ranked: Sequence[RankedReview]
    ) -> list[SearchResult]:
        if not ranked:
            return []
        rows = await session.execute(
            select(
                Review.id,
                AppListing.store,
                Review.external_id,
                Review.rating,
                Review.country,
                ReviewEnrichment.language,
                Review.app_version,
                Review.reviewed_at,
                ReviewEnrichment.model_text,
            )
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .where(Review.id.in_([hit.review_id for hit in ranked]))
        )
        details = {row[0]: row for row in rows}
        return [
            SearchResult(
                review_id=hit.review_id,
                store=details[hit.review_id][1],
                external_id=details[hit.review_id][2],
                rating=details[hit.review_id][3],
                country=details[hit.review_id][4],
                language=details[hit.review_id][5],
                app_version=details[hit.review_id][6],
                reviewed_at=details[hit.review_id][7],
                text=details[hit.review_id][8],
                score=hit.score,
            )
            for hit in ranked
        ]
