import math
from datetime import date, timedelta
from itertools import count

import pytest

from reviewradar.anomalies.detector import (
    Observation,
    binomial_tail,
    detect_spikes,
    usable_days,
)
from reviewradar.config import AnomalySettings

START = date(2026, 8, 20)
ids = count(1)


def reviews(
    source: str,
    day_offset: int,
    topic_id: int,
    number: int,
    *,
    version: str = "7.137.0",
) -> list[Observation]:
    return [
        Observation(
            next(ids), topic_id, source, START + timedelta(days=day_offset), version, index / 10
        )
        for index in range(number)
    ]


def steady_source(
    source: str, days: int, *, per_topic: int = 4, topics: int = 5
) -> list[Observation]:
    """A source whose critical reviews spread evenly over its topics every day."""
    return [
        observation
        for day in range(days)
        for topic_id in range(1, topics + 1)
        for observation in reviews(source, day, topic_id, per_topic)
    ]


@pytest.mark.parametrize(
    ("successes", "trials", "probability"),
    [(0, 10, 0.3), (3, 10, 0.3), (10, 10, 0.3), (7, 20, 0.05)],
)
def test_binomial_tail_matches_the_direct_sum(
    successes: int, trials: int, probability: float
) -> None:
    direct = sum(
        math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
        for k in range(successes, trials + 1)
    )

    assert binomial_tail(successes, trials, probability) == pytest.approx(direct, rel=1e-9)


def test_binomial_tail_stays_finite_for_extreme_spikes() -> None:
    p_value = binomial_tail(31, 41, 0.1)

    assert 0 < p_value < 1e-15


def test_usable_days_drop_the_partial_edges() -> None:
    days = [START + timedelta(days=offset) for offset in (0, 1, 1, 2, 3)]

    assert usable_days(days) == [START + timedelta(days=1), START + timedelta(days=2)]


def test_detects_a_topic_that_takes_over_one_day() -> None:
    observations = steady_source("app_store:de", 10)
    observations += reviews("app_store:de", 5, 3, 30, version="7.138.0")

    detection = detect_spikes(observations, AnomalySettings())

    (spike,) = detection.spikes
    assert (spike.source, spike.day, spike.topic_id) == (
        "app_store:de",
        START + timedelta(days=5),
        3,
    )
    assert (spike.count, spike.total) == (34, 50)
    assert spike.lift > 3
    assert spike.p_value < detection.threshold
    assert spike.top_version == "7.138.0"
    assert spike.top_version_share == pytest.approx(30 / 34)
    assert len(set(spike.review_ids)) == 34


def test_more_coverage_is_not_a_spike() -> None:
    quiet = steady_source("app_store:us", 10, per_topic=1)
    busy_day = [
        observation
        for topic_id in range(1, 6)
        for observation in reviews("app_store:us", 6, topic_id, 20)
    ]

    detection = detect_spikes(quiet + busy_day, AnomalySettings())

    assert detection.spikes == []


def test_ignores_small_spikes_and_sources_with_short_coverage() -> None:
    observations = steady_source("app_store:de", 10)
    observations += reviews("app_store:de", 5, 3, 3)
    short = steady_source("app_store:gb", 4) + reviews("app_store:gb", 2, 1, 40)

    detection = detect_spikes(observations + short, AnomalySettings())

    assert detection.spikes == []
    coverage = {entry.source: entry for entry in detection.coverage}
    assert coverage["app_store:gb"].tested is False
    assert coverage["app_store:gb"].usable_days == 2
    assert coverage["app_store:de"].tested is True


def test_threshold_splits_the_false_alarm_rate_across_tests() -> None:
    detection = detect_spikes(steady_source("app_store:de", 10), AnomalySettings(family_alpha=0.01))

    assert detection.tests == 8 * 5
    assert detection.threshold == pytest.approx(0.01 / 40)
