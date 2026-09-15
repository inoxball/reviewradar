"""``reviewradar translate``."""

import asyncio
from typing import Annotated

import typer

from reviewradar.cli._common import AppSlug, load_app
from reviewradar.config import Settings, get_settings
from reviewradar.translation.service import TranslationReport
from reviewradar.translation.wiring import build_translation_service


def translate(
    app_slug: AppSlug,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Process at most this many reviews.")
    ] = None,
    max_rating: Annotated[
        int | None,
        typer.Option(
            min=1, max=5, help="Only reviews rated at most this, e.g. 3 for topic discovery."
        ),
    ] = None,
) -> None:
    """Translate enriched reviews into the target language with a local model.

    Run `reviewradar enrich` first: translation uses the redacted text and detected language.
    """
    settings = get_settings()
    load_app(settings, app_slug)
    report = asyncio.run(_translate(settings, app_slug, limit=limit, max_rating=max_rating))
    typer.echo(
        f"\n{report.app_slug} · {report.model} → {settings.translation.target_language}\n"
        f"translated {report.reviews_translated} reviews in {report.elapsed_seconds:.1f}s "
        f"({report.reviews_per_second:.1f}/s), unsupported languages {report.reviews_unsupported}"
    )


async def _translate(
    settings: Settings, app_slug: str, *, limit: int | None, max_rating: int | None
) -> TranslationReport:
    async with build_translation_service(settings) as service:
        return await service.translate_app(app_slug, limit=limit, max_rating=max_rating)
