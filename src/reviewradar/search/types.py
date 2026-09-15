"""Search request and result types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from reviewradar.domain import Store


class SearchMode(StrEnum):
    """Which retrievers answer a query."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Constraints applied inside every retriever's query, so filtered results are exact."""

    stores: frozenset[Store] = frozenset()
    languages: frozenset[str] = frozenset()
    min_rating: int | None = None
    max_rating: int | None = None
    since: datetime | None = None

    def __post_init__(self) -> None:
        for bound in (self.min_rating, self.max_rating):
            if bound is not None and not 1 <= bound <= 5:
                raise ValueError(f"rating bounds must be within 1..5, got {bound}")
        if self.since is not None and self.since.tzinfo is None:
            raise ValueError("since must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RankedReview:
    """A review id with a retriever- or fusion-specific score (higher is better)."""

    review_id: int
    score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A review returned to the caller, with the fields needed to display and judge it."""

    review_id: int
    store: Store
    external_id: str
    rating: int
    country: str | None
    language: str | None
    app_version: str | None
    reviewed_at: datetime
    text: str
    score: float
