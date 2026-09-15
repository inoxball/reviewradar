"""Text preparation shared by every model: composition, normalization and PII redaction.

Everything a model sees goes through :func:`prepare_model_text`, so redaction is applied
once, consistently, before text can reach an embedding model or a third-party LLM.
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_CANDIDATE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_MIN_PHONE_DIGITS = 9  # dates ("2026-09-12") and versions ("7.139.0") stay untouched

URL_PLACEHOLDER = "[url]"
EMAIL_PLACEHOLDER = "[email]"
PHONE_PLACEHOLDER = "[phone]"


def prepare_model_text(title: str | None, body: str) -> str:
    """Compose a review into one redacted, whitespace-normalized passage.

    Returns an empty string when neither field carries words (e.g. a lone "." title
    and an emoji-only body); such reviews are not worth embedding.
    """
    return redact_personal_data(compose_review_text(title, body))


def text_fingerprint(text: str) -> str:
    """Stable hash of model input; stored embeddings stay valid while it is unchanged."""
    return hashlib.sha256(text.encode()).hexdigest()


def compose_review_text(title: str | None, body: str) -> str:
    """Join title and body, dropping parts without any letters or digits."""
    parts = [_normalize_whitespace(part) for part in (title, body) if part and _has_words(part)]
    return "\n".join(parts)


def redact_personal_data(text: str) -> str:
    """Replace URLs, email addresses and phone numbers with neutral placeholders."""
    text = _URL.sub(URL_PLACEHOLDER, text)
    text = _EMAIL.sub(EMAIL_PLACEHOLDER, text)
    return _PHONE_CANDIDATE.sub(_redact_phone, text)


def _redact_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = sum(character.isdigit() for character in candidate)
    return PHONE_PLACEHOLDER if digits >= _MIN_PHONE_DIGITS else candidate


def _has_words(text: str) -> bool:
    return any(character.isalnum() for character in text)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
