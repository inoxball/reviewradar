from collections.abc import Sequence

import pytest

from reviewradar.replies.generation import Conversation
from reviewradar.topics.naming import (
    LocalTopicNamer,
    TopicDescription,
    clean_name,
    naming_messages,
    resolve_labels,
)

ENERGY = TopicDescription(
    keywords=["energy", "battery", "hearts"],
    average_rating=1.6,
    examples=["Energy runs out after two lessons", "Die Energie ist sofort leer"],
)


class ScriptedModel:
    has_adapter = False

    def __init__(self, answers: Sequence[str]) -> None:
        self._answers = list(answers)
        self.conversations: list[Conversation] = []

    def complete(self, conversations: Sequence[Conversation], *, use_adapter: bool) -> list[str]:
        assert not use_adapter
        self.conversations += conversations
        return self._answers[: len(conversations)]


def test_prompt_carries_keywords_examples_and_mood() -> None:
    system, user = naming_messages("Duolingo", ENERGY)

    assert system["role"] == "system"
    assert "energy, battery, hearts" in user["content"]
    assert "Die Energie ist sofort leer" in user["content"]
    assert "mostly complaints" in user["content"]
    assert "do not mention Duolingo" in user["content"]
    assert '"issues"' in user["content"]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Energy system limits practice", "Energy system limits practice"),
        ('"energy runs out too fast."\n', "Energy runs out too fast"),
        ("Name: Ads interrupt lessons", "Ads interrupt lessons"),
        ("**Streak lost after update**", "Streak lost after update"),
        ("Energy", None),
        ("This is a very long answer that keeps going well past a name", None),
        ("エネルギーがすぐなくなる", None),
        ("Praise for {app_name}", None),
        ("", None),
    ],
)
def test_clean_name_keeps_only_short_english_phrases(completion: str, expected: str | None) -> None:
    assert clean_name(completion) == expected


def test_namer_cleans_every_answer() -> None:
    model = ScriptedModel(["Energy runs out too fast.", "广告太多"])

    names = LocalTopicNamer(model, model_name="local/qwen").name("Duolingo", [ENERGY, ENERGY])

    assert names == ["Energy runs out too fast", None]
    assert len(model.conversations) == 2


def test_labels_fall_back_to_keywords_and_disambiguate_repeats() -> None:
    labels = resolve_labels(
        ["Praise for learning languages", None, "Praise for learning languages"],
        [["languages", "idiomas"], ["energy", "battery", "hearts"], ["english", "learn"]],
    )

    assert labels == [
        "Praise for learning languages (idiomas)",
        "energy · battery · hearts",
        "Praise for learning languages (english)",
    ]
