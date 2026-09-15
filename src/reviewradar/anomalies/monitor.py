"""Daily incident monitoring: each day is judged against earlier days only.

The retrospective detector (:mod:`reviewradar.anomalies.detector`) compares a day with every
other day of an ingested window. That suits analysis, but an alert job cannot use days that
have not happened yet. The monitor applies the same share test causally:

- A source's day is judged against its usable days in a trailing window, and only once that
  window holds enough of them.
- The false-alarm budget is per day (Bonferroni across that day's tests), so false alerts
  accumulate with days monitored, not with the length of the history.
- Days that raised an alert are left out of later baselines for their source, so a multi-day
  incident keeps alerting instead of becoming its own baseline.
- Alerts for the same topic and source on consecutive days, allowing a short gap, form one
  incident: a three-day outage pages once.
- An incident is blamed on a release only when most of its reviews are on one app version that
  was rare in the source's reviews just before it. The version everyone already runs is not a
  suspect, however dominant.

:func:`backtest` replays every day of the ingested history, and :func:`compare_with_spikes`
reports for each retrospective spike whether, and how early, the monitor would have alerted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from reviewradar.anomalies.detector import (
    BASELINE_PSEUDOCOUNT,
    Observation,
    Spike,
    binomial_tail,
    usable_days,
)
from reviewradar.config import AnomalySettings


@dataclass(frozen=True, slots=True)
class Alert:
    """A topic that dominated a source's critical reviews on one day, judged on earlier days."""

    topic_id: int
    source: str
    day: date
    count: int
    total: int
    expected_share: float
    lift: float
    p_value: float
    baseline_days: int
    review_ids: tuple[int, ...]
    """Closest to the topic centre first."""

    @property
    def share(self) -> float:
        return self.count / self.total


@dataclass(frozen=True, slots=True)
class DayResult:
    day: date
    sources: int
    """Sources with enough earlier coverage to judge this day."""
    tests: int
    threshold: float
    alerts: tuple[Alert, ...]


@dataclass(frozen=True, slots=True)
class Incident:
    topic_id: int
    source: str
    alerts: tuple[Alert, ...]
    """One per alerting day, in day order."""
    top_version: str | None
    top_version_share: float
    prior_version_share: float | None
    """The top version's share of the source's reviews in the days before; None without data."""
    release: str | None
    """The app version to blame, when the incident looks like a release regression."""

    @property
    def first_day(self) -> date:
        return self.alerts[0].day

    @property
    def last_day(self) -> date:
        return self.alerts[-1].day

    @property
    def review_count(self) -> int:
        return sum(alert.count for alert in self.alerts)

    @property
    def peak(self) -> Alert:
        return min(self.alerts, key=lambda alert: (alert.p_value, -alert.count))

    @property
    def review_ids(self) -> tuple[int, ...]:
        """Reviews of the most significant day first."""
        ordered = sorted(self.alerts, key=lambda alert: (alert.p_value, alert.day))
        return tuple(review_id for alert in ordered for review_id in alert.review_ids)


@dataclass(frozen=True, slots=True)
class Backtest:
    days: tuple[DayResult, ...]
    """Every usable day of any source, in order."""
    incidents: tuple[Incident, ...]
    usable_days: dict[str, tuple[date, ...]]
    settings: AnomalySettings = field(repr=False)

    @property
    def tests(self) -> int:
        return sum(result.tests for result in self.days)

    @property
    def judged_days(self) -> int:
        """Days on which at least one source could be judged."""
        return sum(1 for result in self.days if result.sources)

    @property
    def alert_days(self) -> int:
        return sum(1 for result in self.days if result.alerts)

    def earlier_days(self, source: str, day: date) -> int:
        """Usable days of a source inside the baseline window before ``day``."""
        start = day - timedelta(days=self.settings.baseline_days)
        return sum(1 for usable in self.usable_days.get(source, ()) if start <= usable < day)


@dataclass(frozen=True, slots=True)
class SpikeOutcome:
    """Whether the monitor would have alerted on a retrospective spike."""

    spike: Spike
    incident: Incident | None
    reason: str | None
    """Why it was missed."""

    @property
    def delay_days(self) -> int | None:
        """Days from the spike to the incident's first alert; negative if it alerted earlier."""
        return None if self.incident is None else (self.incident.first_day - self.spike.day).days


class _Source:
    """One source's reviews indexed by day, for repeated baseline sums."""

    def __init__(self, name: str, observations: Sequence[Observation]) -> None:
        self.name = name
        self.usable = tuple(usable_days(observation.day for observation in observations))
        self.usable_set = frozenset(self.usable)
        self.alerted: set[date] = set()
        self.topics_by_day: defaultdict[date, Counter[int]] = defaultdict(Counter)
        self.versions_by_day: defaultdict[date, Counter[str | None]] = defaultdict(Counter)
        self.members: defaultdict[tuple[int, date], list[Observation]] = defaultdict(list)
        for observation in observations:
            self.topics_by_day[observation.day][observation.topic_id] += 1
            self.versions_by_day[observation.day][observation.app_version] += 1
            self.members[(observation.topic_id, observation.day)].append(observation)

    def baseline(self, day: date, window: int) -> list[date]:
        start = day - timedelta(days=window)
        return [
            usable for usable in self.usable if start <= usable < day and usable not in self.alerted
        ]

    def version_share(self, version: str | None, start: date, end: date) -> float | None:
        """A version's share of the source's reviews on ``start <= day < end``."""
        counts: Counter[str | None] = Counter()
        for offset in range((end - start).days):
            counts.update(self.versions_by_day.get(start + timedelta(days=offset), Counter()))
        total = sum(counts.values())
        return counts[version] / total if total else None


def backtest(observations: Sequence[Observation], settings: AnomalySettings) -> Backtest:
    """Replay the monitor over every usable day, in order, as a daily job would have run."""
    by_source: defaultdict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_source[observation.source].append(observation)
    sources = [_Source(name, by_source[name]) for name in sorted(by_source)]

    days = sorted({day for source in sources for day in source.usable})
    results = tuple(_judge_day(sources, day, settings) for day in days)
    return Backtest(
        days=results,
        incidents=_incidents(results, {source.name: source for source in sources}, settings),
        usable_days={source.name: source.usable for source in sources},
        settings=settings,
    )


def compare_with_spikes(spikes: Sequence[Spike], result: Backtest) -> tuple[SpikeOutcome, ...]:
    """For each retrospective spike, the incident that covers its day, or why there is none."""
    outcomes: list[SpikeOutcome] = []
    for spike in spikes:
        incident = next(
            (
                incident
                for incident in result.incidents
                if incident.source == spike.source
                and incident.topic_id == spike.topic_id
                and incident.first_day <= spike.day <= incident.last_day
            ),
            None,
        )
        reason = None
        if incident is None:
            earlier = result.earlier_days(spike.source, spike.day)
            needed = result.settings.min_baseline_days
            reason = (
                f"only {earlier} earlier days of coverage, and the monitor needs {needed}"
                if earlier < needed
                else "not significant against earlier days alone"
            )
        outcomes.append(SpikeOutcome(spike, incident, reason))
    return tuple(outcomes)


def _judge_day(sources: Sequence[_Source], day: date, settings: AnomalySettings) -> DayResult:
    candidates: list[tuple[_Source, int, int, int, float, int]] = []
    judged = 0
    for source in sources:
        if day not in source.usable_set:
            continue
        baseline = source.baseline(day, settings.baseline_days)
        if len(baseline) < settings.min_baseline_days:
            continue
        judged += 1
        baseline_topics: Counter[int] = Counter()
        for baseline_day in baseline:
            baseline_topics.update(source.topics_by_day[baseline_day])
        baseline_total = sum(baseline_topics.values())
        today = source.topics_by_day[day]
        total = sum(today.values())
        for topic_id, count in today.items():
            expected = (baseline_topics[topic_id] + BASELINE_PSEUDOCOUNT) / (
                baseline_total + 2 * BASELINE_PSEUDOCOUNT
            )
            candidates.append((source, topic_id, count, total, expected, len(baseline)))

    tests = len(candidates)
    threshold = settings.daily_alpha / tests if tests else 0.0
    alerts: list[Alert] = []
    for source, topic_id, count, total, expected, baseline_days in candidates:
        lift = (count / total) / expected
        if count < settings.min_count or lift < settings.min_lift:
            continue
        p_value = binomial_tail(count, total, expected)
        if p_value > threshold:
            continue
        members = sorted(
            source.members[(topic_id, day)], key=lambda item: (item.distance, item.review_id)
        )
        alerts.append(
            Alert(
                topic_id=topic_id,
                source=source.name,
                day=day,
                count=count,
                total=total,
                expected_share=expected,
                lift=lift,
                p_value=p_value,
                baseline_days=baseline_days,
                review_ids=tuple(member.review_id for member in members),
            )
        )
        source.alerted.add(day)
    alerts.sort(key=lambda alert: (alert.p_value, -alert.count))
    return DayResult(day, judged, tests, threshold, tuple(alerts))


def _incidents(
    days: Sequence[DayResult], sources: dict[str, _Source], settings: AnomalySettings
) -> tuple[Incident, ...]:
    grouped: defaultdict[tuple[str, int], list[list[Alert]]] = defaultdict(list)
    for result in days:
        for alert in result.alerts:
            runs = grouped[(alert.source, alert.topic_id)]
            if runs and (alert.day - runs[-1][-1].day).days <= settings.incident_gap_days + 1:
                runs[-1].append(alert)
            else:
                runs.append([alert])

    incidents = [
        _incident(alerts, sources[source], settings)
        for (source, _), runs in grouped.items()
        for alerts in runs
    ]
    incidents.sort(key=lambda incident: (incident.first_day, incident.source, incident.topic_id))
    return tuple(incidents)


def _incident(alerts: list[Alert], source: _Source, settings: AnomalySettings) -> Incident:
    first = alerts[0]
    versions = Counter(
        member.app_version
        for alert in alerts
        for member in source.members[(first.topic_id, alert.day)]
    )
    top_version, top_count = versions.most_common(1)[0]
    top_share = top_count / sum(versions.values())
    prior_share = source.version_share(
        top_version, first.day - timedelta(days=settings.new_version_days), first.day
    )
    is_release = (
        top_version is not None
        and top_share >= settings.release_share
        and prior_share is not None
        and prior_share <= settings.max_prior_version_share
    )
    return Incident(
        topic_id=first.topic_id,
        source=source.name,
        alerts=tuple(alerts),
        top_version=top_version,
        top_version_share=top_share,
        prior_version_share=prior_share,
        release=top_version if is_release else None,
    )
