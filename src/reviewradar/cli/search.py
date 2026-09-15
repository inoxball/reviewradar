"""``reviewradar search``."""

import asyncio
from datetime import UTC, datetime
from typing import Annotated

import typer

from reviewradar.cli._common import AppSlug, fail, load_app, one_line
from reviewradar.config import Settings, get_settings
from reviewradar.domain import Store
from reviewradar.search.retrievers import UnsupportedDatabaseError
from reviewradar.search.types import SearchFilters, SearchMode, SearchResult
from reviewradar.search.wiring import build_search_service


def search(
    app_slug: AppSlug,
    query: Annotated[str, typer.Argument(help="Free-text query, in any language.")],
    mode: Annotated[SearchMode, typer.Option("--mode", "-m", help="Retrieval mode.")] = (
        SearchMode.HYBRID
    ),
    limit: Annotated[int, typer.Option(min=1, max=100, help="Number of results.")] = 10,
    stores: Annotated[
        list[Store] | None, typer.Option("--store", "-s", help="Restrict to a store; repeatable.")
    ] = None,
    languages: Annotated[
        list[str] | None,
        typer.Option("--language", "-l", help="Restrict to a review language; repeatable."),
    ] = None,
    min_rating: Annotated[int | None, typer.Option(min=1, max=5)] = None,
    max_rating: Annotated[int | None, typer.Option(min=1, max=5)] = None,
    since: Annotated[
        datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Only reviews from this UTC date.")
    ] = None,
) -> None:
    """Search an app's reviews by meaning and by keywords (requires PostgreSQL)."""
    settings = get_settings()
    load_app(settings, app_slug)
    filters = SearchFilters(
        stores=frozenset(stores or ()),
        languages=frozenset(languages or ()),
        min_rating=min_rating,
        max_rating=max_rating,
        since=since.replace(tzinfo=UTC) if since else None,
    )
    try:
        results = asyncio.run(_search(settings, app_slug, query, mode, filters, limit))
    except UnsupportedDatabaseError as exc:
        fail(str(exc))
    _print_results(results)


async def _search(
    settings: Settings,
    app_slug: str,
    query: str,
    mode: SearchMode,
    filters: SearchFilters,
    limit: int,
) -> list[SearchResult]:
    async with build_search_service(settings) as service:
        return await service.search(app_slug, query, mode=mode, filters=filters, limit=limit)


def _print_results(results: list[SearchResult]) -> None:
    if not results:
        typer.echo("No matching reviews.")
        return
    for rank, result in enumerate(results, start=1):
        typer.echo(
            f"{rank:>2}. {'★' * result.rating:<5} {result.store:<11} {result.language or '?':<3} "
            f"{result.reviewed_at:%Y-%m-%d}  {one_line(result.text, 90)}"
        )
