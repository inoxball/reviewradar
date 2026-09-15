"""Command-line interface. Run ``reviewradar --help`` for usage."""

from typing import Annotated

import typer

from reviewradar.cli import (
    anomalies,
    db,
    enrichment,
    evaluation,
    ingestion,
    replies,
    search,
    topics,
    translation,
    web,
)
from reviewradar.config import get_settings
from reviewradar.log import configure_logging

app = typer.Typer(
    help="ReviewRadar: AI-powered app review intelligence.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="Override REVIEWRADAR_LOG_LEVEL.")
    ] = None,
) -> None:
    configure_logging(log_level or get_settings().log_level)


app.add_typer(db.app, name="db")
app.add_typer(evaluation.app, name="eval")
app.add_typer(replies.app, name="replies")
app.command()(ingestion.ingest)
app.command()(ingestion.stats)
app.command()(enrichment.enrich)
app.command()(enrichment.languages)
app.command()(translation.translate)
app.command()(search.search)
app.command()(topics.topics)
app.command()(anomalies.anomalies)
app.command()(web.serve)

__all__ = ["app"]
