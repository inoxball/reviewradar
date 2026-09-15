"""``reviewradar db``: schema management and the embedded development server."""

from typing import Annotated

import typer

from reviewradar.cli._common import fail
from reviewradar.config import get_settings
from reviewradar.db import migrate
from reviewradar.db.embedded_postgres import (
    embedded_data_dir,
    resolve_database_url,
    stop_embedded_server,
)

app = typer.Typer(help="Database schema management.", no_args_is_help=True)


@app.command()
def upgrade(
    revision: Annotated[str, typer.Argument(help="Target migration revision.")] = "head",
) -> None:
    """Apply database migrations."""
    migrate.upgrade(revision)
    typer.echo(f"Database upgraded to {revision}.")


@app.command()
def url() -> None:
    """Print the resolved database URL (starts the embedded server for pgserver:/// URLs)."""
    typer.echo(resolve_database_url(get_settings().database_url))


@app.command()
def stop() -> None:
    """Stop the embedded PostgreSQL server of a pgserver:/// database URL."""
    database_url = get_settings().database_url
    if embedded_data_dir(database_url) is None:
        fail("REVIEWRADAR_DATABASE_URL is not a pgserver:/// URL; there is no embedded server")
    if stop_embedded_server(database_url):
        typer.echo("Embedded PostgreSQL stopped. The next command starts it again.")
    else:
        typer.echo("Embedded PostgreSQL was not running.")
