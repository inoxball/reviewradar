"""Runs reply generators on held-out reviews and scores what they write.

The held-out references (written or published replies) go through the same checks and
appear as the first row, so every generator is read against the bar it should reach.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reviewradar.replies.evaluation import GeneratorSummary, ReplyChecks, check_reply, summarize
from reviewradar.replies.generation import ReplyGenerator, ReplyTask

REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    key: str
    app_name: str
    support_contact: str | None
    foreign_brands: tuple[str, ...]
    task: ReplyTask
    reference: str
    record: dict[str, Any] = field(default_factory=dict)
    """The dataset record, copied into the drafts file for reading results side by side."""


@dataclass(frozen=True, slots=True)
class Draft:
    key: str
    generator: str
    reply: str
    checks: ReplyChecks


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    summaries: list[GeneratorSummary]
    drafts: list[Draft]


def run_benchmark(
    examples: Sequence[BenchmarkExample],
    generators: Sequence[ReplyGenerator],
    *,
    detect_language: Callable[[str], str | None],
    timer: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Score the references, then draft and score replies with every generator."""
    outputs: list[tuple[str, list[str], float]] = [
        (REFERENCE, [example.reference for example in examples], 0.0)
    ]
    for generator in generators:
        started = timer()
        replies = generator.generate([example.task for example in examples])
        outputs.append((generator.name, replies, timer() - started))

    summaries: list[GeneratorSummary] = []
    drafts: list[Draft] = []
    for name, replies, seconds in outputs:
        checks = [
            check_reply(
                reply,
                review=example.task.request.review,
                language=example.task.request.language,
                app_name=example.app_name,
                support_contact=example.support_contact,
                foreign_brands=example.foreign_brands,
                detect_language=detect_language,
            )
            for example, reply in zip(examples, replies, strict=True)
        ]
        summaries.append(summarize(name, replies, checks, seconds))
        drafts += [
            Draft(example.key, name, reply, result)
            for example, reply, result in zip(examples, replies, checks, strict=True)
        ]
    return BenchmarkResult(summaries, drafts)


def write_benchmark(
    result: BenchmarkResult,
    examples: Sequence[BenchmarkExample],
    directory: Path,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Write ``drafts.jsonl`` (one row per review and generator) and ``summary.json``."""
    by_key = {example.key: example for example in examples}
    rows = []
    for draft in result.drafts:
        record = by_key[draft.key].record
        rows.append(
            {
                "key": draft.key,
                "generator": draft.generator,
                "language": record.get("language"),
                "rating": record.get("rating"),
                "review": record.get("review"),
                "reply": draft.reply,
                "passed": draft.checks.passed,
                "checks": asdict(draft.checks),
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "drafts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {**metadata, "generators": [asdict(summary) for summary in result.summaries]}
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
