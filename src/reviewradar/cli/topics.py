"""``reviewradar topics``."""

import asyncio
from collections.abc import Sequence
from typing import Annotated

import typer

from reviewradar.catalog import AppSpec
from reviewradar.cli._common import AppSlug, fail, load_app
from reviewradar.config import Settings, get_settings
from reviewradar.topics.naming import TopicNamer
from reviewradar.topics.service import (
    InsufficientReviewsError,
    NoTopicRunError,
    RelabelReport,
    TopicRunReport,
    TopicScope,
    TopicSummary,
)
from reviewradar.topics.wiring import build_topic_service, create_topic_namer


def topics(
    app_slug: AppSlug,
    count: Annotated[
        int | None, typer.Option("--topics", min=2, max=200, help="Number of topics to find.")
    ] = None,
    max_rating: Annotated[
        int | None, typer.Option(min=1, max=5, help="Only reviews rated at most this.")
    ] = None,
    all_ratings: Annotated[
        bool, typer.Option("--all-ratings", help="Include reviews of every rating.")
    ] = False,
    min_words: Annotated[
        int | None, typer.Option(min=1, help="Skip reviews shorter than this many words.")
    ] = None,
    relabel: Annotated[
        bool,
        typer.Option("--relabel", help="Only label the latest run again, without clustering."),
    ] = False,
    name: Annotated[
        bool,
        typer.Option(
            "--name",
            help="Label topics with names from the local language model (experimental; GPU).",
        ),
    ] = False,
) -> None:
    """Discover topics by clustering review embeddings.

    Run `enrich` first; running `translate` gives keywords more English text to work with.
    """
    settings = get_settings()
    spec = load_app(settings, app_slug)
    namer = _load_namer(settings) if name else None

    if relabel:
        _print_relabel(_relabel(settings, spec, namer))
        return

    config = settings.topics
    scope = TopicScope(
        max_rating=None
        if all_ratings
        else (max_rating if max_rating is not None else config.max_rating),
        min_words=min_words if min_words is not None else config.min_words,
    )
    try:
        report = asyncio.run(_discover(settings, spec, scope, count))
    except InsufficientReviewsError as exc:
        fail(str(exc))
    _print_report(report, scope)
    if namer is not None:
        _print_relabel(_relabel(settings, spec, namer))


def _load_namer(settings: Settings) -> TopicNamer:
    """Loads on the main thread, as torch requires."""
    try:
        return create_topic_namer(settings)
    except ImportError as exc:
        fail(f"naming topics needs the optional 'finetune' extra: {exc}")


def _relabel(settings: Settings, spec: AppSpec, namer: TopicNamer | None) -> RelabelReport:
    async def run() -> RelabelReport:
        async with build_topic_service(settings) as service:
            return await service.relabel(spec, namer)

    try:
        return asyncio.run(run())
    except NoTopicRunError as exc:
        fail(str(exc))


async def _discover(
    settings: Settings, spec: AppSpec, scope: TopicScope, topic_count: int | None
) -> TopicRunReport:
    async with build_topic_service(settings) as service:
        return await service.discover(spec, scope=scope, topic_count=topic_count)


def _print_report(report: TopicRunReport, scope: TopicScope) -> None:
    ratings = "all ratings" if scope.max_rating is None else f"≤{scope.max_rating}★"
    typer.echo(
        f"\n{report.app_slug}: {len(report.topics)} topics from {report.review_count} reviews "
        f"({ratings}, ≥{scope.min_words} words) in {report.elapsed_seconds:.1f}s, "
        f"run {report.run_id}\n"
    )
    _print_topics(report.topics)


def _print_relabel(report: RelabelReport) -> None:
    how = (
        f"with names from {report.model} ({report.fallbacks} kept keyword labels)"
        if report.model
        else "with English keywords"
    )
    typer.echo(
        f"\n{report.app_slug}: relabelled {len(report.topics)} topics of run {report.run_id} "
        f"{how} in {report.elapsed_seconds:.1f}s\n"
    )
    _print_topics(report.topics)


def _print_topics(topics: Sequence[TopicSummary]) -> None:
    typer.echo(f"{'reviews':>7} {'avg ★':>6} {'≤2★':>5}  label")
    for topic in topics:
        stats = f"{topic.size:>7} {topic.average_rating:>6.2f} {topic.negative_share:>5.0%}"
        typer.echo(f"{stats}  {topic.label}")
