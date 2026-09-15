"""Reciprocal rank fusion: merges rankings whose scores are not comparable."""

from __future__ import annotations

from collections.abc import Sequence

from reviewradar.search.types import RankedReview

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = DEFAULT_RRF_K
) -> list[RankedReview]:
    """Fuse ranked lists of review ids into one ranking.

    Every list contributes ``1 / (k + rank)`` for each id it contains (ranks start at 1).
    Ranks are fused instead of scores because full-text ranks and cosine similarities live
    on unrelated scales. ``k = 60`` (Cormack et al., 2009) damps the dominance of any
    single list's top positions. Ties break on the best individual rank, then on id, so
    the output is deterministic.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, review_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[review_id] = scores.get(review_id, 0.0) + 1.0 / (k + rank)
            best_rank[review_id] = min(rank, best_rank.get(review_id, rank))

    ordered = sorted(
        scores, key=lambda review_id: (-scores[review_id], best_rank[review_id], review_id)
    )
    return [RankedReview(review_id=review_id, score=scores[review_id]) for review_id in ordered]
