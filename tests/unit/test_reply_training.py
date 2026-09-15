from typing import Any

from reviewradar.replies.prompting import MAX_REVIEW_CHARS
from reviewradar.replies.training import to_prompt_completion


class EchoTemplate:
    """Records chat-template calls and renders messages as plain text."""

    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        self.kwargs.append(kwargs)
        return "|".join(f"{message['role']}:{message['content']}" for message in conversation)


def record(source: str, review: str = "Crashes on start") -> dict[str, Any]:
    return {
        "store": "google_play",
        "rating": 1,
        "language": "en",
        "review": review,
        "reply": "Sorry about the crash.",
        "source": source,
    }


def test_renders_prompts_without_thinking_and_repeats_written_references() -> None:
    template = EchoTemplate()

    pairs = to_prompt_completion(
        [record("published"), record("written")], template, "GUIDELINES", written_repeats=3
    )

    assert len(pairs) == 4
    assert pairs[0]["prompt"].startswith("system:GUIDELINES|user:Store: Google Play")
    assert pairs[0]["completion"] == "Sorry about the crash."
    assert template.kwargs[0] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_long_reviews_are_shortened_in_the_prompt() -> None:
    (pair,) = to_prompt_completion(
        [record("published", review="word " * 1000)], EchoTemplate(), "G"
    )

    review = pair["prompt"].split("Review:\n", 1)[1].split("\n\n", 1)[0]
    assert len(review) == MAX_REVIEW_CHARS
    assert review.endswith("…")
