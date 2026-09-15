"""``reviewradar replies``: datasets, fine-tuning and evaluation for reply drafting."""

import asyncio
from typing import Annotated

import typer

from reviewradar.cli._common import fail
from reviewradar.config import get_settings
from reviewradar.replies.dataset import ReplySource, Split
from reviewradar.replies.wiring import GENERATORS, build_reply_dataset, run_reply_benchmark

app = typer.Typer(help="Draft developer replies with a local model.", no_args_is_help=True)


@app.command()
def dataset() -> None:
    """Build train and eval files from published replies and written references."""
    settings = get_settings()
    report = asyncio.run(build_reply_dataset(settings))

    typer.echo(f"\n{'source':<10} {'train':>6} {'eval':>6}")
    for source in ReplySource:
        train, held_out = report.kept[(source, Split.TRAIN)], report.kept[(source, Split.EVAL)]
        typer.echo(f"{source.value:<10} {train:>6} {held_out:>6}")
    typer.echo("\nDropped:")
    for reason, count in report.dropped.most_common():
        typer.echo(f"  {count:>5}  {reason}")
    languages = ", ".join(f"{code} {count}" for code, count in report.languages.most_common(8))
    typer.echo(f"\nLanguages: {languages}")
    typer.echo(f"Written to {settings.replies.dataset_dir}")


@app.command()
def train(
    max_steps: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many optimizer steps (smoke runs)."),
    ] = None,
) -> None:
    """Fine-tune a LoRA adapter on the reply dataset (run `replies dataset` first)."""
    from reviewradar.replies.training import train_reply_adapter

    report = train_reply_adapter(get_settings().replies, max_steps=max_steps)
    typer.echo(
        f"\nAdapter saved to {report.adapter_dir}\n"
        f"steps {report.steps}, train loss {report.train_loss:.3f}, "
        f"eval loss {report.eval_loss:.3f}, {report.runtime_seconds / 60:.1f} min, "
        f"peak GPU memory {report.peak_memory_gb:.2f} GB"
    )


@app.command()
def evaluate(
    generator: Annotated[
        list[str] | None,
        typer.Option(
            "--generator", "-g", help=f"Repeat to pick several of: {', '.join(GENERATORS)}."
        ),
    ] = None,
    source: Annotated[
        str, typer.Option(help="Held-out examples to use: written, published or all.")
    ] = "written",
    limit: Annotated[int | None, typer.Option(min=1, help="Use at most this many.")] = None,
) -> None:
    """Draft replies for held-out reviews with each generator and score them."""
    if source not in {"written", "published", "all"}:
        fail(f"unknown source '{source}'; choose written, published or all")
    settings = get_settings()
    try:
        result = run_reply_benchmark(
            settings,
            generator_names=generator or GENERATORS,
            source=None if source == "all" else ReplySource(source),
            limit=limit,
        )
    except (ValueError, FileNotFoundError) as exc:
        fail(str(exc))

    typer.echo(
        f"\n{'generator':<11} {'n':>4} {'pass':>6} {'language':>9} {'<=350':>6} "
        f"{'openings':>9} {'chars':>6} {'s/reply':>8}"
    )
    for summary in result.summaries:
        rates = summary.check_rates
        typer.echo(
            f"{summary.name:<11} {summary.replies:>4} {summary.pass_rate:>6.0%} "
            f"{rates.get('right_language', 0):>9.0%} {rates.get('within_limit', 0):>6.0%} "
            f"{summary.opening_diversity:>9.0%} {summary.median_chars:>6.0f} "
            f"{summary.seconds_per_reply:>8.2f}"
        )
    typer.echo(f"\nDrafts and summary written to {settings.replies.benchmark_dir}")
