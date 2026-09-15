"""``reviewradar enrich`` and ``reviewradar languages``."""

import asyncio
from collections import defaultdict
from typing import Annotated

import typer

from reviewradar.catalog import AppSpec
from reviewradar.cli._common import AppSlug, load_app, read
from reviewradar.config import Settings, get_settings
from reviewradar.domain import Store
from reviewradar.enrichment.repository import EnrichmentRepository, LanguageShare
from reviewradar.enrichment.service import EnrichmentReport
from reviewradar.enrichment.wiring import build_enrichment_service


def enrich(
    app_slug: AppSlug,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Process at most this many reviews.")
    ] = None,
) -> None:
    """Redact, detect language and embed reviews that are new or edited since the last run."""
    settings = get_settings()
    spec = load_app(settings, app_slug)
    report = asyncio.run(_enrich(settings, spec, limit=limit))
    _print_report(report)


def languages(app_slug: AppSlug) -> None:
    """Show the resolved language mix per store (populated by `enrich`)."""
    shares = asyncio.run(
        read(get_settings(), lambda session: EnrichmentRepository(session).language_mix(app_slug))
    )
    if not shares:
        typer.echo(f"No enriched reviews for '{app_slug}' yet; run `reviewradar enrich` first.")
        return
    _print_language_mix(shares)


async def _enrich(settings: Settings, spec: AppSpec, *, limit: int | None) -> EnrichmentReport:
    async with build_enrichment_service(settings) as service:
        return await service.enrich_app(spec, limit=limit)


def _print_report(report: EnrichmentReport) -> None:
    sources = ", ".join(
        f"{source} {count}" for source, count in sorted(report.language_sources.items())
    )
    typer.echo(
        f"\n{report.app_slug} · {report.model}\n"
        f"processed {report.reviews_processed} reviews in {report.elapsed_seconds:.1f}s "
        f"({report.reviews_per_second:.0f}/s)\n"
        f"embedded {report.reviews_embedded}, without text {report.reviews_without_text}\n"
        f"language sources: {sources or 'n/a'}"
    )


def _print_language_mix(shares: list[LanguageShare], *, top: int = 8) -> None:
    by_store: defaultdict[Store, list[LanguageShare]] = defaultdict(list)
    for share in shares:
        by_store[share.store].append(share)

    for store, store_shares in by_store.items():
        total = sum(share.review_count for share in store_shares)
        breakdown = "  ".join(
            f"{share.language or '?'} {share.review_count / total:.0%}"
            for share in store_shares[:top]
        )
        typer.echo(f"{store:<12} {total:>7} reviews   {breakdown}")
