"""Spike detection on the share of a day's critical reviews that belong to one topic.

Raw daily counts are misleading for review data. Stores expose reviews unevenly: the App
Store feed stops at 500 reviews per storefront, so a busy storefront covers four days and a
quiet one three weeks; an ingestion run that starts mid-day sees half a day. Counting reviews
per topic would flag every change in coverage as an incident.

The detector therefore works per **source** (a store and market) and compares shares:

- Only a source's *usable* days are tested: days strictly between its first and last
  observed day, because the edges are partial.
- For each topic and usable day, the observed share of that day's critical reviews is tested
  against the topic's share on the source's other usable days, with an exact binomial test.
- A spike needs enough reviews, a large lift and a p-value below the family-wise threshold
  (Bonferroni over every topic-day tested), so one alarm in a hundred runs is a false one.

The baseline uses all other days, before and after: this is retrospective analysis of an
ingested window, which also catches incidents on the first usable day. An online monitor
would use only earlier days.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from reviewradar.config import AnomalySettings

BASELINE_PSEUDOCOUNT = 0.5
"""Added to a topic's baseline count, so a topic never seen before still has a small share."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One critical review assigned to a topic."""

    review_id: int
    topic_id: int
    source: str
    day: date
    app_version: str | None
    distance: float = 0.0


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    source: str
    first_day: date
    last_day: date
    usable_days: int
    tested: bool


@dataclass(frozen=True, slots=True)
class Spike:
    topic_id: int
    source: str
    day: date
    count: int
    """Reviews in the topic from this source on this day."""
    total: int
    """All critical reviews from this source on this day."""
    expected_share: float
    lift: float
    p_value: float
    top_version: str | None
    top_version_share: float
    review_ids: tuple[int, ...]
    """The spike's reviews, closest to the topic centre first."""

    @property
    def share(self) -> float:
        return self.count / self.total


@dataclass(frozen=True, slots=True)
class Detection:
    spikes: list[Spike]
    coverage: list[SourceCoverage]
    tests: int
    threshold: float
    settings: AnomalySettings = field(repr=False)


def binomial_tail(successes: int, trials: int, probability: float) -> float:
    """P(X ≥ successes) for X ~ Binomial(trials, probability), exact and stable in log space."""
    if successes <= 0:
        return 1.0
    if successes > trials or probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    log_p, log_q = math.log(probability), math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        + k * log_p
        + (trials - k) * log_q
        for k in range(successes, trials + 1)
    ]
    peak = max(log_terms)
    return min(1.0, math.exp(peak) * sum(math.exp(term - peak) for term in log_terms))


def usable_days(days: Iterable[date]) -> list[date]:
    """Observed days strictly between the first and the last: the edges are partial."""
    observed = sorted(set(days))
    return observed[1:-1]


def detect_spikes(observations: Sequence[Observation], settings: AnomalySettings) -> Detection:
    by_source: defaultdict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_source[observation.source].append(observation)

    coverage: list[SourceCoverage] = []
    candidates: list[tuple[str, date, int, int, int, float, list[Observation]]] = []
    for source in sorted(by_source):
        items = by_source[source]
        usable = set(usable_days(item.day for item in items))
        days = [item.day for item in items]
        tested = len(usable) >= settings.min_usable_days
        coverage.append(SourceCoverage(source, min(days), max(days), len(usable), tested))
        if not tested:
            continue

        in_window = [item for item in items if item.day in usable]
        totals = Counter(item.day for item in in_window)
        topic_totals = Counter(item.topic_id for item in in_window)
        grouped: defaultdict[tuple[int, date], list[Observation]] = defaultdict(list)
        for item in in_window:
            grouped[(item.topic_id, item.day)].append(item)
        grand_total = len(in_window)

        for (topic_id, day), members in grouped.items():
            count, total = len(members), totals[day]
            other_topic = topic_totals[topic_id] - count
            other_total = grand_total - total
            expected = (other_topic + BASELINE_PSEUDOCOUNT) / (
                other_total + 2 * BASELINE_PSEUDOCOUNT
            )
            candidates.append((source, day, topic_id, count, total, expected, members))

    tests = len(candidates)
    threshold = settings.family_alpha / tests if tests else 0.0
    spikes: list[Spike] = []
    for source, day, topic_id, count, total, expected, members in candidates:
        lift = (count / total) / expected
        if count < settings.min_count or lift < settings.min_lift:
            continue
        p_value = binomial_tail(count, total, expected)
        if p_value > threshold:
            continue
        versions = Counter(member.app_version for member in members)
        top_version, top_count = versions.most_common(1)[0]
        spikes.append(
            Spike(
                topic_id=topic_id,
                source=source,
                day=day,
                count=count,
                total=total,
                expected_share=expected,
                lift=lift,
                p_value=p_value,
                top_version=top_version,
                top_version_share=top_count / count,
                review_ids=tuple(
                    member.review_id
                    for member in sorted(members, key=lambda item: (item.distance, item.review_id))
                ),
            )
        )
    spikes.sort(key=lambda spike: (spike.p_value, -spike.count))
    return Detection(spikes, coverage, tests, threshold, settings)
