"""Automatic checks for drafted replies, summarized per generator.

The checks catch failures that make a reply unusable whatever one's taste:
- written in the wrong language;
- over Google Play's 350-character limit once placeholders are filled;
- leaking another app's brand or a raw e-mail address or URL;
- using a placeholder that does not exist;
- parroting the review back;
- cut off mid-sentence, because the model reached its token limit;
- empty or near-empty.

Across replies, opening diversity shows whether a generator answers every review with the
same template. Whether a reply is actually *good* needs a judge; these checks only gate it.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields

from reviewradar.replies.cleaning import PLACEHOLDERS, has_enough_words
from reviewradar.replies.prompting import REPLY_CHAR_LIMIT, render_reply

COPY_NGRAM = 8
OPENING_WORDS = 5
MIN_WORDS = 3
SENTENCE_ENDINGS = (
    ".",
    "!",
    "?",
    ")",
    '"',
    "'",
    "\u2026",
    "\u3002",
    "\uff01",
    "\uff1f",
    "\u300d",
    "\u00bb",
)

_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_CONTACT = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_WORD = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class ReplyChecks:
    non_empty: bool
    right_language: bool
    within_limit: bool
    no_foreign_brand_or_contact: bool
    valid_placeholders: bool
    not_copied: bool
    complete: bool

    @property
    def passed(self) -> bool:
        return all(getattr(self, check.name) for check in fields(self))


@dataclass(frozen=True, slots=True)
class GeneratorSummary:
    name: str
    replies: int
    pass_rate: float
    check_rates: dict[str, float]
    opening_diversity: float
    median_chars: float
    seconds_per_reply: float


def check_reply(
    reply: str,
    *,
    review: str,
    language: str | None,
    app_name: str,
    support_contact: str | None,
    foreign_brands: Iterable[str],
    detect_language: Callable[[str], str | None],
) -> ReplyChecks:
    """Evaluate one drafted reply, written with placeholders, against its review."""
    without_placeholders = _PLACEHOLDER.sub("", reply)
    lowered = reply.lower()
    return ReplyChecks(
        non_empty=has_enough_words(reply, MIN_WORDS),
        right_language=language is None or detect_language(without_placeholders) == language,
        within_limit=len(render_reply(reply, app_name=app_name, support_contact=support_contact))
        <= REPLY_CHAR_LIMIT,
        no_foreign_brand_or_contact=not _CONTACT.search(without_placeholders)
        and not any(
            re.search(rf"(?<!\w){re.escape(brand.lower())}(?!\w)", lowered)
            for brand in foreign_brands
        ),
        complete=_ends_cleanly(reply),
        valid_placeholders=set(_PLACEHOLDER.findall(reply)) <= PLACEHOLDERS,
        not_copied=not _shared_ngrams(without_placeholders, review, COPY_NGRAM),
    )


def summarize(
    name: str, replies: Sequence[str], checks: Sequence[ReplyChecks], seconds: float
) -> GeneratorSummary:
    count = len(replies)
    openings = {" ".join(reply.lower().split()[:OPENING_WORDS]) for reply in replies}
    return GeneratorSummary(
        name=name,
        replies=count,
        pass_rate=sum(check.passed for check in checks) / count if count else 0.0,
        check_rates={
            check.name: sum(getattr(result, check.name) for result in checks) / count
            for check in fields(ReplyChecks)
        }
        if count
        else {},
        opening_diversity=len(openings) / count if count else 0.0,
        median_chars=statistics.median(len(reply) for reply in replies) if count else 0.0,
        seconds_per_reply=seconds / count if count else 0.0,
    )


def _ends_cleanly(reply: str) -> bool:
    """Whether the reply ends like a finished sentence rather than mid-phrase."""
    return reply.rstrip().endswith(SENTENCE_ENDINGS)


def _shared_ngrams(text: str, source: str, size: int) -> bool:
    def ngrams(value: str) -> set[tuple[str, ...]]:
        tokens = _WORD.findall(value.lower())
        return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}

    return bool(ngrams(text) & ngrams(source))
