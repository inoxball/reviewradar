"""The prompt every reply generator receives, and rendering of what it returns.

Generators write replies with ``{app_name}`` and ``{support_contact}`` placeholders, so one
model serves every app and training data from other apps stays brand-neutral. Rendering
fills them from the catalog.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reviewradar.domain import Store
from reviewradar.replies.cleaning import APP_NAME, SUPPORT_CONTACT

REPLY_CHAR_LIMIT = 350
"""Google Play's limit for developer replies; the App Store allows far more."""

MAX_REVIEW_CHARS = 1500
"""Longer reviews are cut, so the prompt and the reply fit the model's training context."""

STORE_NAMES = {Store.GOOGLE_PLAY: "Google Play", Store.APP_STORE: "App Store"}


@dataclass(frozen=True, slots=True)
class ReplyRequest:
    """A review to answer, in the redacted form every model sees."""

    store: Store
    rating: int
    language: str | None
    review: str


def load_guidelines(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_messages(
    request: ReplyRequest,
    guidelines: str,
    examples: Sequence[tuple[ReplyRequest, str]] = (),
) -> list[dict[str, str]]:
    """Chat messages for one review.

    The guidelines are the system prompt; optional worked examples (answered reviews)
    become earlier conversation turns; the review to answer is the last user turn.
    """
    messages = [{"role": "system", "content": guidelines}]
    for example, reply in examples:
        messages.append({"role": "user", "content": format_review(example)})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": format_review(request)})
    return messages


def format_review(request: ReplyRequest) -> str:
    return (
        f"Store: {STORE_NAMES[request.store]}\n"
        f"Rating: {request.rating}/5\n"
        f"Language: {request.language or 'unknown'}\n\n"
        f"Review:\n{_shorten(request.review)}\n\n"
        "Write the developer reply."
    )


def render_reply(text: str, *, app_name: str, support_contact: str | None) -> str:
    """Fill placeholders; without a support contact the app's name stands in for it."""
    contact = support_contact or f"{app_name} support"
    return " ".join(text.replace(APP_NAME, app_name).replace(SUPPORT_CONTACT, contact).split())


def _shorten(review: str) -> str:
    if len(review) <= MAX_REVIEW_CHARS:
        return review
    return review[: MAX_REVIEW_CHARS - 1].rstrip() + "…"
