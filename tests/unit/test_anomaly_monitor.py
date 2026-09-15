from datetime import timedelta

from reviewradar.anomalies.detector import Observation, detect_spikes
from reviewradar.anomalies.monitor import backtest, compare_with_spikes
from reviewradar.config import AnomalySettings
from tests.unit.test_anomaly_detector import START, reviews, steady_source

SOURCE = "app_store:de"
SURGE = 30


def surge(day: int, *, topic_id: int = 2, version: str = "7.137.0") -> list[Observation]:
    return reviews(SOURCE, day, topic_id, SURGE, version=version)


def test_alerts_on_a_surge_judged_on_earlier_days() -> None:
    observations = steady_source(SOURCE, 12) + surge(8)

    result = backtest(observations, AnomalySettings())

    (incident,) = result.incidents
    assert (incident.source, incident.topic_id) == (SOURCE, 2)
    assert incident.first_day == incident.last_day == START + timedelta(days=8)
    assert (incident.peak.count, incident.peak.total) == (4 + SURGE, 20 + SURGE)
    assert incident.peak.baseline_days == 7
    assert result.alert_days == 1
    assert len(incident.review_ids) == 4 + SURGE


def test_days_without_enough_history_are_not_judged() -> None:
    observations = steady_source(SOURCE, 12) + surge(3)
    settings = AnomalySettings()

    result = backtest(observations, settings)
    (outcome,) = compare_with_spikes(detect_spikes(observations, settings).spikes, result)

    assert result.incidents == ()
    day_three = next(day for day in result.days if day.day == START + timedelta(days=3))
    assert day_three.sources == 0
    assert outcome.incident is None
    assert outcome.reason is not None
    assert "only 2 earlier days" in outcome.reason


def test_a_multi_day_incident_keeps_alerting_and_pages_once() -> None:
    observations = steady_source(SOURCE, 20) + surge(8) + surge(9) + surge(11) + surge(15)

    result = backtest(observations, AnomalySettings(incident_gap_days=1))

    first, second = result.incidents
    assert [alert.day.day - START.day for alert in first.alerts] == [8, 9, 11]
    assert second.first_day == START + timedelta(days=15)


def test_release_regressions_need_a_new_dominant_version() -> None:
    steady = steady_source(SOURCE, 12)  # every steady review is on 7.137.0

    new_version = backtest(steady + surge(8, version="7.138.0"), AnomalySettings())
    old_version = backtest(steady + surge(8), AnomalySettings())

    assert new_version.incidents[0].release == "7.138.0"
    assert new_version.incidents[0].top_version_share == SURGE / (SURGE + 4)
    assert old_version.incidents[0].release is None
    assert old_version.incidents[0].top_version == "7.137.0"


def test_threshold_is_split_across_one_days_tests() -> None:
    observations = steady_source(SOURCE, 12) + steady_source("google_play:de", 12)
    settings = AnomalySettings(daily_alpha=0.01)

    result = backtest(observations, settings)

    day = next(day for day in result.days if day.day == START + timedelta(days=8))
    assert (day.sources, day.tests) == (2, 10)
    assert day.threshold == 0.01 / 10
    assert result.incidents == ()


def test_caught_spikes_report_their_delay() -> None:
    observations = steady_source(SOURCE, 12) + surge(8)
    settings = AnomalySettings()

    result = backtest(observations, settings)
    (outcome,) = compare_with_spikes(detect_spikes(observations, settings).spikes, result)

    assert outcome.incident is result.incidents[0]
    assert outcome.delay_days == 0
    assert outcome.reason is None


def test_the_version_everyone_already_runs_is_not_blamed() -> None:
    before_release = [
        observation
        for day in range(4)
        for topic_id in range(1, 6)
        for observation in reviews(SOURCE, day, topic_id, 4, version="7.137.0")
    ]
    after_release = [
        observation
        for day in range(4, 12)
        for topic_id in range(1, 6)
        for observation in reviews(SOURCE, day, topic_id, 4, version="7.138.0")
    ]
    observations = before_release + after_release + surge(8, version="7.138.0")

    (incident,) = backtest(observations, AnomalySettings()).incidents

    assert incident.top_version == "7.138.0"
    assert incident.prior_version_share == 4 / 7
    assert incident.release is None
