"""Readable topic names from a small local language model.

Keyword labels ("icon · screen · bad") are distinctive but hard to read, and they break down
when a topic's reviews are not translated: "bom · muito bom · aprender". A local instruction
model reads a topic's keywords and its most typical reviews, in any language, and writes a
short English name. Every answer is checked, and a topic keeps its keyword label when the
model's answer is unusable.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from reviewradar.replies.generation import CompletionModel, Conversation

NAME_MIN_WORDS = 2
NAME_MAX_WORDS = 7
NAME_MAX_CHARS = 60
EXAMPLE_CHARS = 220
LATIN_LIMIT = 0x24F
"""Letters above this code point are not Latin script, so the name is not English."""
VAGUE_WORDS = ("issues", "problems", "reviews", "feedback", "experience")

SYSTEM_PROMPT = (
    "You label clusters of app store reviews for a product team. "
    "Labels are short, concrete and in English."
)

_EN_DASH, _ELLIPSIS = chr(0x2013), chr(0x2026)
_CURLY_QUOTES = "".join(map(chr, (0x201C, 0x201D, 0x2018, 0x2019)))
_PREFIX = re.compile(rf"^(?:topic\s+name|label|name|topic|title)\s*[:\-{_EN_DASH}]\s*", re.I)
_WRAPPING = "\"'`*#" + _CURLY_QUOTES


@dataclass(frozen=True, slots=True)
class TopicDescription:
    """What the model sees of a topic."""

    keywords: Sequence[str]
    average_rating: float
    examples: Sequence[str]
    """The most typical reviews, in English when a translation exists."""


class TopicNamer(Protocol):
    @property
    def model_name(self) -> str: ...

    def name(self, app_name: str, topics: Sequence[TopicDescription]) -> list[str | None]:
        """A usable name per topic, or None where the model gave none."""
        ...


class LocalTopicNamer:
    """Names topics with a local instruction model, such as the reply drafter's base model."""

    def __init__(self, model: CompletionModel, *, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def name(self, app_name: str, topics: Sequence[TopicDescription]) -> list[str | None]:
        conversations = [naming_messages(app_name, topic) for topic in topics]
        completions = self._model.complete(conversations, use_adapter=False)
        return [clean_name(completion) for completion in completions]


def naming_messages(app_name: str, topic: TopicDescription) -> Conversation:
    if topic.average_rating <= 2.5:
        mood = "mostly complaints"
    elif topic.average_rating < 3.8:
        mood = "mixed opinions"
    else:
        mood = "mostly praise"
    examples = "\n".join(f"- {_excerpt(text)}" for text in topic.examples)
    vague = ", ".join(f'"{word}"' for word in VAGUE_WORDS)
    request = (
        f"A cluster of {app_name} reviews: {mood}, average rating "
        f"{topic.average_rating:.1f} of 5.\n"
        f"Most distinctive words: {', '.join(topic.keywords) or 'none'}\n\n"
        f"Typical reviews, in any language:\n{examples}\n\n"
        "Write one label for this cluster.\n"
        "- 2 to 6 English words, in sentence case, without quotes or a final period.\n"
        "- Name the concrete subject the reviews share: a feature, a problem, a language or "
        "a reason for praise, and say what users think of it.\n"
        f"- Do not use vague words such as {vague}, and do not mention {app_name}.\n"
        "Answer with the label only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]


def clean_name(completion: str) -> str | None:
    """The name in a model answer, or None when it is not a short English phrase."""
    line = next((line.strip() for line in completion.splitlines() if line.strip()), "")
    line = _PREFIX.sub("", line).strip().strip(_WRAPPING).strip().rstrip(".!").strip()
    words = line.split()
    if not NAME_MIN_WORDS <= len(words) <= NAME_MAX_WORDS or len(line) > NAME_MAX_CHARS:
        return None
    if any(char.isalpha() and ord(char) > LATIN_LIMIT for char in line):
        return None
    if any(char in line for char in "{}<>[]"):
        return None
    return line[0].upper() + line[1:]


def resolve_labels(names: Sequence[str | None], keywords: Sequence[Sequence[str]]) -> list[str]:
    """Final labels: the model's name, else the keyword label; repeated names get a keyword.

    Two topics can deserve the same name ("Praise for learning languages"); the first
    keyword that the name does not already contain tells them apart.
    """
    from reviewradar.topics.keywords import keyword_label

    labels = [name or keyword_label(terms) for name, terms in zip(names, keywords, strict=True)]
    repeated = Counter(label.lower() for label in labels)
    resolved: list[str] = []
    for label, name, terms in zip(labels, names, keywords, strict=True):
        if name is not None and repeated[label.lower()] > 1:
            extra = next((term for term in terms if term.lower() not in label.lower()), None)
            label = f"{label} ({extra})" if extra else label
        resolved.append(label)
    return resolved


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= EXAMPLE_CHARS else flat[: EXAMPLE_CHARS - 1].rstrip() + _ELLIPSIS
