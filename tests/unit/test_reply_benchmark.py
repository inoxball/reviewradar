import json
from collections.abc import Sequence
from itertools import count
from pathlib import Path

from reviewradar.domain import Store
from reviewradar.replies.benchmark import (
    REFERENCE,
    BenchmarkExample,
    run_benchmark,
    write_benchmark,
)
from reviewradar.replies.cleaning import clean_developer_reply, has_enough_words
from reviewradar.replies.generation import ReplyTask
from reviewradar.replies.prompting import ReplyRequest


class CannedGenerator:
    def __init__(self, name: str, reply: str) -> None:
        self.name = name
        self._reply = reply

    def generate(self, tasks: Sequence[ReplyTask]) -> list[str]:
        return [self._reply for _ in tasks]


def example(key: str, review: str) -> BenchmarkExample:
    return BenchmarkExample(
        key=key,
        app_name="Duolingo",
        support_contact="support.duolingo.com",
        foreign_brands=("Babbel",),
        task=ReplyTask(ReplyRequest(Store.GOOGLE_PLAY, 1, "en", review)),
        reference="Sorry about the crash, please contact {support_contact} with your device.",
        record={"language": "en", "rating": 1, "review": review},
    )


def english(_: str) -> str:
    return "en"


def test_benchmark_scores_references_first_then_every_generator(tmp_path: Path) -> None:
    examples = [example("a", "The app crashes"), example("b", "Lessons do not load")]
    ticks = count()
    generators = [
        CannedGenerator("good", "We're sorry about this, please write to {support_contact}."),
        CannedGenerator("leaky", "Thanks for using Babbel, we are looking into it right now."),
    ]

    result = run_benchmark(
        examples, generators, detect_language=english, timer=lambda: float(next(ticks))
    )
    write_benchmark(result, examples, tmp_path, metadata={"source": "written"})

    assert [summary.name for summary in result.summaries] == [REFERENCE, "good", "leaky"]
    assert [summary.pass_rate for summary in result.summaries] == [1.0, 1.0, 0.0]
    assert result.summaries[1].seconds_per_reply == 0.5
    rows = [
        json.loads(line) for line in (tmp_path / "drafts.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert rows[-1]["checks"]["no_foreign_brand_or_contact"] is False
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert summary["source"] == "written"
    assert [row["name"] for row in summary["generators"]] == [REFERENCE, "good", "leaky"]


def test_scripts_without_spaces_count_as_words() -> None:
    japanese = "ご意見ありがとうございます。いただいた声はチームに共有いたします。"

    assert has_enough_words(japanese, 5)
    assert not has_enough_words("ありがとう", 5)
    assert clean_developer_reply(japanese, brand_names=[]) == japanese
