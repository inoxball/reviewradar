"""``reviewradar ingest`` and ``reviewradar stats``."""

import asyncio
from collections.abc import Collection
from datetime import UTC, datetime
from typing import Annotated

import typer

from reviewradar.catalog import AppSpec
from reviewradar.cli._common import AppSlug, load_app, read
from reviewradar.config import Settings, get_settings
from reviewradar.domain import Store
from reviewradar.ingestion.repository import MarketSummary, ReviewRepository
from reviewradar.ingestion.service import IngestionReport
from reviewradar.ingestion.wiring import build_ingestion_service


def ingest(
    app_slug: AppSlug,
    stores: Annotated[
        list[Store] | None,
        typer.Option("--store", "-s", help="Restrict to a store; repeat for several."),
    ] = None,
    since: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Backfill from this UTC date, ignoring cursors."),
    ] = None,
    max_reviews: Annotated[
        int | None,
        typer.Option(min=1, help="Cap reviews per store partition for this run."),
    ] = None,
) -> None:
    """Fetch new and edited reviews for an app across its stores and markets."""
    settings = get_settings()
    if max_reviews is not None:
        ingestion = settings.ingestion.model_copy(update={"max_reviews_per_market": max_reviews})
        settings = settings.model_copy(update={"ingestion": ingestion})

    spec = load_app(settings, app_slug)
    report = asyncio.run(
        _ingest(settings, spec, stores=stores, since=since.replace(tzinfo=UTC) if since else None)
    )
    _print_report(report)
    if report.failed:
        raise typer.Exit(code=1)


def stats(app_slug: AppSlug) -> None:
    """Summarize stored reviews per store and country."""
    summaries = asyncio.run(
        read(get_settings(), lambda session: ReviewRepository(session).summarize_app(app_slug))
    )
    if not summaries:
        typer.echo(f"No reviews stored for '{app_slug}' yet.")
        return
    _print_summaries(summaries)


async def _ingest(
    settings: Settings,
    spec: AppSpec,
    *,
    stores: Collection[Store] | None,
    since: datetime | None,
) -> IngestionReport:
    async with build_ingestion_service(settings) as service:
        return await service.ingest_app(spec, stores=stores, since=since)


def _print_report(report: IngestionReport) -> None:
    typer.echo(
        f"\n{'store':<12} {'partition':<10} {'status':<10} {'fetched':>8} "
        f"{'new':>6} {'updated':>8} {'unchanged':>10}  coverage"
    )
    for result in report.results:
        coverage = "complete" if result.coverage_complete else "partial"
        typer.echo(
            f"{result.store:<12} {result.partition_key:<10} {result.status:<10} "
            f"{result.fetched:>8} {result.stats.inserted:>6} {result.stats.updated:>8} "
            f"{result.stats.unchanged:>10}  {coverage}"
        )
        if result.error:
            typer.secho(f"  └─ {result.error}", fg=typer.colors.RED)

    totals = report.totals
    typer.echo(
        f"\n{report.app_slug}: {totals.inserted} new, {totals.updated} updated, "
        f"{totals.unchanged} unchanged across {len(report.results)} partitions"
    )


def _print_summaries(summaries: list[MarketSummary]) -> None:
    typer.echo(f"{'store':<12} {'country':<8} {'reviews':>8} {'avg ★':>6}  {'oldest':<17} newest")
    for summary in summaries:
        oldest = f"{summary.oldest_reviewed_at:%Y-%m-%d %H:%M}"
        newest = f"{summary.newest_reviewed_at:%Y-%m-%d %H:%M}"
        typer.echo(
            f"{summary.store:<12} {summary.country or '-':<8} {summary.review_count:>8} "
            f"{summary.average_rating:>6.2f}  {oldest:<17} {newest}"
        )
