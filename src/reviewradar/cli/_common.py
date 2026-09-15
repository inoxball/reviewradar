"""Helpers shared by CLI commands."""

from collections.abc import Awaitable, Callable
from typing import Annotated, NoReturn

import typer
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.catalog import AppSpec, Catalog, UnknownAppError
from reviewradar.config import Settings
from reviewradar.db.session import create_engine, create_session_factory

AppSlug = Annotated[str, typer.Argument(help="App slug from the catalog, e.g. 'duolingo'.")]


def fail(message: str, *, code: int = 2) -> NoReturn:
    """Print an error to stderr and exit with ``code``."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def load_app(settings: Settings, slug: str) -> AppSpec:
    try:
        return Catalog.from_yaml(settings.catalog_path).get(slug)
    except UnknownAppError as exc:
        fail(str(exc))


async def read[T](settings: Settings, query: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Run a read-only query in a short-lived engine."""
    engine = create_engine(settings.database_url)
    try:
        async with create_session_factory(engine)() as session:
            return await query(session)
    finally:
        await engine.dispose()


def one_line(text: str, width: int) -> str:
    """Collapse whitespace and truncate for tabular output."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else f"{flat[: width - 1]}…"
