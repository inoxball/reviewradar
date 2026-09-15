"""``reviewradar eval``: pooling candidates for annotation and scoring search modes."""

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from reviewradar.cli._common import fail
from reviewradar.config import Settings, get_settings
from reviewradar.search.evaluation import (
    JudgmentSet,
    ModeReport,
    PooledQuery,
    QuerySet,
    evaluate_modes,
    pool_candidates,
)
from reviewradar.search.types import SearchMode
from reviewradar.search.wiring import build_search_service

app = typer.Typer(help="Offline search evaluation.", no_args_is_help=True)

ALL_MODES = list(SearchMode)
GRADING_GUIDE = (
    "2 = directly about the query's need, 1 = partial or passing mention, 0 = not relevant."
)


@app.command()
def pool(
    queries_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o", help="Annotation file to write.")],
    depth: Annotated[int, typer.Option(min=1, max=100, help="Top results per mode.")] = 10,
    force: Annotated[bool, typer.Option(help="Overwrite an existing annotation file.")] = False,
) -> None:
    """Pool the top results of every search mode into a file for relevance grading."""
    if output.exists() and not force:
        fail(f"{output} exists and may contain annotations; pass --force to overwrite it")

    query_set = QuerySet.from_yaml(queries_file)
    pooled = asyncio.run(_pool(get_settings(), query_set, depth))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _annotation_document(query_set, pooled),
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
    candidates = sum(len(query.candidates) for query in pooled)
    typer.echo(f"Wrote {candidates} candidates for {len(pooled)} queries to {output}.")


@app.command("search")
def evaluate_search(
    judgments_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    k: Annotated[int, typer.Option(min=1, max=100, help="Cut-off rank.")] = 10,
    per_query: Annotated[bool, typer.Option(help="Also print nDCG per query.")] = False,
) -> None:
    """Score every search mode against graded relevance judgments."""
    judgment_set = JudgmentSet.from_yaml(judgments_file)
    reports = asyncio.run(_evaluate(get_settings(), judgment_set, k))
    _print_reports(reports, per_query=per_query)


async def _pool(settings: Settings, query_set: QuerySet, depth: int) -> list[PooledQuery]:
    async with build_search_service(settings) as service:
        return await pool_candidates(service, query_set, modes=ALL_MODES, depth=depth)


async def _evaluate(settings: Settings, judgment_set: JudgmentSet, k: int) -> list[ModeReport]:
    async with build_search_service(settings) as service:
        return await evaluate_modes(service, judgment_set, modes=ALL_MODES, k=k)


def _annotation_document(query_set: QuerySet, pooled: list[PooledQuery]) -> dict[str, Any]:
    return {
        "app": query_set.app,
        "annotator": "TODO: who graded these judgments",
        "notes": f"Pooled from all search modes. Grades: {GRADING_GUIDE}",
        "queries": [
            {
                "id": entry.query.id,
                "text": entry.query.text,
                "judgments": [
                    {
                        "store": str(result.store),
                        "id": result.external_id,
                        "grade": None,
                        "text": " ".join(result.text.split()),
                    }
                    for result in entry.candidates
                ],
            }
            for entry in pooled
        ],
    }


def _print_reports(reports: list[ModeReport], *, per_query: bool) -> None:
    k = reports[0].k
    typer.echo(f"{'mode':<10} {f'nDCG@{k}':>8} {f'P@{k}':>7} {'MRR':>6} {f'judged@{k}':>10}")
    for report in reports:
        typer.echo(
            f"{report.mode:<10} {report.ndcg:>8.3f} {report.precision:>7.3f} "
            f"{report.mrr:>6.3f} {report.judged_fraction:>10.2f}"
        )
    if not per_query:
        return

    typer.echo(f"\nnDCG@{k} per query")
    typer.echo(f"{'query':<22}" + "".join(f"{report.mode:>10}" for report in reports))
    for index, metrics in enumerate(reports[0].queries):
        row = "".join(f"{report.queries[index].ndcg:>10.3f}" for report in reports)
        typer.echo(f"{metrics.query_id:<22}{row}")
