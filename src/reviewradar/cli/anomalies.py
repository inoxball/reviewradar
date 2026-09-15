"""``reviewradar anomalies``."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

import typer

from reviewradar.anomalies.monitor import SpikeOutcome
from reviewradar.anomalies.repository import AnomalyRepository
from reviewradar.anomalies.service import AnomalyService, BacktestReport
from reviewradar.cli._common import AppSlug, fail, load_app, one_line, read
from reviewradar.config import Settings, get_settings
from reviewradar.db.session import create_engine, create_session_factory


def anomalies(
    app_slug: AppSlug,
    backtest: Annotated[
        bool,
        typer.Option(
            "--backtest",
            help="Replay the daily monitor instead: each day judged on earlier days only.",
        ),
    ] = False,
) -> None:
    """Find days when one topic suddenly dominated a source's critical reviews."""
    settings = get_settings()
    spec = load_app(settings, app_slug)
    if backtest:
        _print_backtest(settings, spec.slug)
        return

    report = asyncio.run(_with_service(settings, lambda service: service.detect(spec.slug)))
    if report is None:
        fail(f"no topic run for '{spec.slug}'; run `reviewradar topics {spec.slug}` first")

    detection = report.detection
    tested = [entry for entry in detection.coverage if entry.tested]
    skipped = [entry for entry in detection.coverage if not entry.tested]
    typer.echo(
        f"\n{spec.slug}: topic run {report.run.id}, {len(tested)} sources tested, "
        f"{detection.tests} topic-days, threshold p ≤ {detection.threshold:.1e}"
    )
    if skipped:
        short = ", ".join(f"{entry.source} ({entry.usable_days}d)" for entry in skipped)
        typer.echo(f"Too little coverage to test: {short}")
    if not detection.spikes:
        typer.echo("\nNo spikes.")
        return

    samples = settings.anomalies.samples
    example_ids = [
        review_id for spike in detection.spikes for review_id in spike.review_ids[:samples]
    ]
    texts = asyncio.run(
        read(settings, lambda session: AnomalyRepository(session).review_texts(example_ids))
    )
    for spike in detection.spikes:
        topic = report.topics[spike.topic_id]
        typer.echo(
            f"\n{spike.day} · {spike.source} · {topic.label}\n"
            f"  {spike.count} of {spike.total} critical reviews ({spike.share:.0%}, usually "
            f"{spike.expected_share:.0%}; {spike.lift:.1f}x), p ={spike.p_value:.1e}, "
            f"{spike.top_version_share:.0%} on version {spike.top_version or 'unknown'}"
        )
        for review_id in spike.review_ids[:samples]:
            typer.echo(f"  - {one_line(texts.get(review_id, ''), 110)}")


def _print_backtest(settings: Settings, app_slug: str) -> None:
    report: BacktestReport | None = asyncio.run(
        _with_service(settings, lambda service: service.backtest(app_slug))
    )
    if report is None:
        fail(f"no topic run for '{app_slug}'; run `reviewradar topics {app_slug}` first")

    result = report.backtest
    config = settings.anomalies
    typer.echo(
        f"\n{app_slug}: topic run {report.run.id}, {result.judged_days} of {len(result.days)} "
        f"days judged, {result.alert_days} with alerts, {len(result.incidents)} incidents\n"
        f"Each day is judged on up to {config.baseline_days} earlier days (at least "
        f"{config.min_baseline_days}), with a false-alarm budget of {config.daily_alpha} per day."
    )

    samples = config.samples
    example_ids = [
        review_id for incident in result.incidents for review_id in incident.review_ids[:samples]
    ]
    texts = asyncio.run(
        read(settings, lambda session: AnomalyRepository(session).review_texts(example_ids))
    )
    for incident in result.incidents:
        peak = incident.peak
        when = (
            str(incident.first_day)
            if incident.first_day == incident.last_day
            else f"{incident.first_day} to {incident.last_day} ({len(incident.alerts)} alert days)"
        )
        release = (
            f" · possible release regression in {incident.release}" if incident.release else ""
        )
        typer.echo(
            f"\n{when} · {incident.source} · {report.topics[incident.topic_id].label}{release}\n"
            f"  peak {peak.day}: {peak.count} of {peak.total} critical reviews ({peak.share:.0%}, "
            f"before {peak.expected_share:.0%}; {peak.lift:.1f}x), p = {peak.p_value:.1e}, "
            f"judged on {peak.baseline_days} earlier days"
        )
        for review_id in incident.review_ids[:samples]:
            typer.echo(f"  - {one_line(texts.get(review_id, ''), 110)}")

    if report.outcomes:
        typer.echo("\nSpikes found in hindsight:")
        for outcome in report.outcomes:
            spike = outcome.spike
            label = report.topics[spike.topic_id].label
            typer.echo(f"  {spike.day} · {spike.source} · {label}: {_verdict(outcome)}")


def _verdict(outcome: SpikeOutcome) -> str:
    delay = outcome.delay_days
    if delay is None:
        return f"missed, {outcome.reason}"
    if delay == 0:
        return "alerted the same day"
    return f"alerted {delay} days later" if delay > 0 else f"alerted {-delay} days earlier"


async def _with_service[T](
    settings: Settings, action: Callable[[AnomalyService], Awaitable[T]]
) -> T:
    engine = create_engine(settings.database_url)
    try:
        return await action(AnomalyService(create_session_factory(engine), settings.anomalies))
    finally:
        await engine.dispose()
