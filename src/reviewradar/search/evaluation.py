"""Offline retrieval evaluation against graded relevance judgments.

Judgments identify reviews by ``(store, external_id)``, which is stable across databases
and re-ingestion, unlike surrogate keys. Grades: 2 = directly about the query's need,
1 = partially relevant or a passing mention, 0 = not relevant.

Judgments are built by *pooling*: the top results of every system under test are judged,
and anything never judged counts as not relevant. That favours the pooled systems, so each
report includes the judged fraction of the top k to make the bias visible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reviewradar.domain import Store
from reviewradar.search.service import SearchService
from reviewradar.search.types import SearchMode, SearchResult

ReviewKey = tuple[Store, str]


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    text: str = Field(min_length=1)


class QuerySet(BaseModel):
    """Queries to evaluate, before any judgments exist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app: str
    queries: tuple[EvaluationQuery, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _query_ids_are_unique(self) -> Self:
        _require_unique_ids(query.id for query in self.queries)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> QuerySet:
        with path.open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


class Judgment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    store: Store
    id: str
    grade: int = Field(ge=0, le=2)
    text: str | None = Field(
        default=None, description="Snapshot of the review text when judged, for auditing."
    )


class JudgedQuery(EvaluationQuery):
    judgments: tuple[Judgment, ...] = ()

    @model_validator(mode="after")
    def _reviews_are_judged_once(self) -> Self:
        keys = [(judgment.store, judgment.id) for judgment in self.judgments]
        if len(keys) != len(set(keys)):
            raise ValueError(f"query {self.id!r} judges the same review more than once")
        return self

    def grades(self) -> dict[ReviewKey, int]:
        return {(judgment.store, judgment.id): judgment.grade for judgment in self.judgments}


class JudgmentSet(BaseModel):
    """A versioned file of queries and their graded relevance judgments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app: str
    annotator: str
    notes: str = ""
    queries: tuple[JudgedQuery, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _query_ids_are_unique(self) -> Self:
        _require_unique_ids(query.id for query in self.queries)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> JudgmentSet:
        with path.open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


@dataclass(frozen=True, slots=True)
class PooledQuery:
    """The union of every mode's top results for one query, ready for annotation."""

    query: EvaluationQuery
    candidates: tuple[SearchResult, ...]


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    query_id: str
    ndcg: float
    precision: float
    reciprocal_rank: float
    judged_fraction: float


@dataclass(frozen=True, slots=True)
class ModeReport:
    """Metrics of one search mode, averaged over all queries."""

    mode: SearchMode
    k: int
    queries: tuple[QueryMetrics, ...]

    @property
    def ndcg(self) -> float:
        return fmean(metrics.ndcg for metrics in self.queries)

    @property
    def precision(self) -> float:
        return fmean(metrics.precision for metrics in self.queries)

    @property
    def mrr(self) -> float:
        return fmean(metrics.reciprocal_rank for metrics in self.queries)

    @property
    def judged_fraction(self) -> float:
        return fmean(metrics.judged_fraction for metrics in self.queries)


async def pool_candidates(
    service: SearchService,
    query_set: QuerySet,
    *,
    modes: Sequence[SearchMode],
    depth: int,
) -> list[PooledQuery]:
    """Collect the top ``depth`` results of every mode per query.

    Candidates are ordered by store and id, not by rank, so the annotator cannot tell
    which system ranked a review highly.
    """
    pooled: list[PooledQuery] = []
    for query in query_set.queries:
        union: dict[ReviewKey, SearchResult] = {}
        for mode in modes:
            results = await service.search(query_set.app, query.text, mode=mode, limit=depth)
            for result in results:
                union.setdefault((result.store, result.external_id), result)
        pooled.append(
            PooledQuery(query=query, candidates=tuple(union[key] for key in sorted(union)))
        )
    return pooled


async def evaluate_modes(
    service: SearchService,
    judgment_set: JudgmentSet,
    *,
    modes: Sequence[SearchMode],
    k: int,
) -> list[ModeReport]:
    """Run every query in every mode and score the top ``k`` against the judgments."""
    reports: list[ModeReport] = []
    for mode in modes:
        metrics: list[QueryMetrics] = []
        for query in judgment_set.queries:
            results = await service.search(judgment_set.app, query.text, mode=mode, limit=k)
            ranked = [(result.store, result.external_id) for result in results]
            metrics.append(evaluate_ranking(query.id, ranked, query.grades(), k=k))
        reports.append(ModeReport(mode=mode, k=k, queries=tuple(metrics)))
    return reports


def evaluate_ranking(
    query_id: str, ranked: Sequence[ReviewKey], grades: Mapping[ReviewKey, int], *, k: int
) -> QueryMetrics:
    """Score one ranked list against graded judgments; unjudged results count as 0."""
    top = list(ranked[:k])
    gains = [grades.get(key, 0) for key in top]
    ideal_dcg = _dcg(sorted(grades.values(), reverse=True)[:k])
    first_relevant = next((rank for rank, gain in enumerate(gains, start=1) if gain > 0), None)
    return QueryMetrics(
        query_id=query_id,
        ndcg=_dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0,
        precision=sum(gain > 0 for gain in gains) / k,
        reciprocal_rank=1.0 / first_relevant if first_relevant else 0.0,
        judged_fraction=sum(key in grades for key in top) / k,
    )


def _dcg(gains: Sequence[int]) -> float:
    """Discounted cumulative gain with exponential gains, rewarding grade 2 over grade 1."""
    return sum((2.0**gain - 1.0) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _require_unique_ids(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for query_id in ids:
        if query_id in seen:
            raise ValueError(f"duplicate query id {query_id!r}")
        seen.add(query_id)
