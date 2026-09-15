from collections.abc import Sequence

import numpy as np
import pytest

from reviewradar.domain import Store
from reviewradar.replies.evaluation import check_reply, summarize
from reviewradar.replies.generation import (
    Conversation,
    FineTunedGenerator,
    NearestExamples,
    PromptedGenerator,
    ReplyTask,
)
from reviewradar.replies.prompting import ReplyRequest


def request(review: str, rating: int = 1) -> ReplyRequest:
    return ReplyRequest(Store.GOOGLE_PLAY, rating, "en", review)


class RecordingModel:
    """Returns the last user message of every conversation and records the calls."""

    def __init__(self, *, has_adapter: bool = True) -> None:
        self.has_adapter = has_adapter
        self.calls: list[tuple[list[Conversation], bool]] = []

    def complete(self, conversations: Sequence[Conversation], *, use_adapter: bool) -> list[str]:
        self.calls.append((list(conversations), use_adapter))
        return [conversation[-1]["content"] for conversation in conversations]


def test_prompted_generator_runs_without_the_adapter() -> None:
    model = RecordingModel()

    replies = PromptedGenerator(model, "GUIDE", name="zero-shot").generate(
        [ReplyTask(request("App crashes"))]
    )

    ((conversations, use_adapter),) = model.calls
    assert use_adapter is False
    assert [message["role"] for message in conversations[0]] == ["system", "user"]
    assert "App crashes" in replies[0]


def test_retrieval_generator_adds_similar_examples_as_earlier_turns() -> None:
    examples = NearestExamples(
        np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32),
        [(request("energy"), "E"), (request("ads"), "A"), (request("battery"), "B")],
        count=2,
    )
    model = RecordingModel()
    generator = PromptedGenerator(model, "GUIDE", name="retrieval", examples=examples)

    generator.generate([ReplyTask(request("hearts"), embedding=np.array([1, 0], dtype=np.float32))])

    (((conversation,), _),) = model.calls
    assert [message["role"] for message in conversation] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]  # fmt: skip
    assert [conversation[2]["content"], conversation[4]["content"]] == ["B", "E"]


def test_retrieval_requires_an_embedding() -> None:
    generator = PromptedGenerator(
        RecordingModel(),
        "GUIDE",
        name="retrieval",
        examples=NearestExamples(
            np.eye(2, dtype=np.float32), [(request("a"), "A"), (request("b"), "B")]
        ),
    )

    with pytest.raises(ValueError, match="embedding"):
        generator.generate([ReplyTask(request("no vector"))])


def test_fine_tuned_generator_uses_the_adapter() -> None:
    model = RecordingModel()

    FineTunedGenerator(model, "GUIDE", name="lora").generate([ReplyTask(request("Crash"))])

    assert model.calls[0][1] is True


def english(_: str) -> str:
    return "en"


def check(reply: str, review: str = "The app crashes whenever I open a lesson on my phone"):  # type: ignore[no-untyped-def]
    return check_reply(
        reply,
        review=review,
        language="en",
        app_name="Duolingo",
        support_contact="support.duolingo.com",
        foreign_brands=["Babbel", "Mondly"],
        detect_language=english,
    )


def test_good_reply_passes_every_check() -> None:
    result = check("Sorry about the crashes. Please contact {support_contact} with your device.")

    assert result.passed


@pytest.mark.parametrize(
    ("reply", "failed"),
    [
        ("Ok", "non_empty"),
        ("x " * 200, "within_limit"),
        ("Thanks for using Babbel, we will look into the crash.", "no_foreign_brand_or_contact"),
        ("Please write to help@duolingo.com about the crash.", "no_foreign_brand_or_contact"),
        ("Hi {user_name}, sorry about the crash.", "valid_placeholders"),
        ("Sorry, the app crashes whenever I open a lesson on my phone.", "not_copied"),
        ("Sorry about the crash, please contact {support_contact} and tell us", "complete"),
    ],
)
def test_each_check_catches_its_failure(reply: str, failed: str) -> None:
    result = check(reply)

    assert not getattr(result, failed)
    assert not result.passed


def test_summary_reports_rates_and_template_diversity() -> None:
    replies = [
        "Sorry about the crash, please contact us.",
        "Sorry about the crash, please contact us.",
        "Thanks for the kind words!",
        "Ok",
    ]

    summary = summarize("gen", replies, [check(reply) for reply in replies], seconds=2.0)

    assert (summary.replies, summary.pass_rate, summary.seconds_per_reply) == (4, 0.75, 0.5)
    assert summary.check_rates["non_empty"] == 0.75
    assert summary.opening_diversity == 0.75
