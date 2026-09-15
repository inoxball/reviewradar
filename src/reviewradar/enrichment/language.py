"""Language identification that weighs detector confidence against store and market evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from reviewradar.domain import LanguageSource


class LanguageDetector(Protocol):
    """Identifies the language of a text as ``(ISO 639-1 code, probability)``."""

    def detect(self, text: str) -> tuple[str, float]: ...


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    """The resolved language of a review and the evidence behind it."""

    language: str | None
    confidence: float | None
    candidate: str | None
    """The detector's best guess, kept even when overruled so thresholds can be tuned."""
    source: LanguageSource


class LangidDetector:
    """Offline identification of 97 languages with py3langid (naive Bayes on byte n-grams).

    Probabilities are normalized, so confidence is comparable across texts. Accuracy drops
    on short or informal texts, which :func:`resolve_language` compensates for.
    """

    def __init__(self) -> None:
        self._identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)

    def detect(self, text: str) -> tuple[str, float]:
        language, confidence = self._identifier.classify(text)
        return str(language), float(confidence)


def resolve_language(
    text: str,
    detector: LanguageDetector,
    *,
    hint: str | None,
    market_default: str | None,
    min_confidence: float,
    min_corroborated_confidence: float,
    min_chars: int,
) -> LanguageGuess:
    """Choose the most trustworthy language for a review.

    Evidence, strongest first:

    1. a confident detection;
    2. a weaker detection that agrees with the store hint or the market default;
    3. the store hint (Google Play serves reviews for a requested language);
    4. the market default (the storefront's main language, from the app catalog).

    On real App Store data most uncertain detections were correct but short and informal
    ("I love it", "DUO IS DA BEST"), which is what rule 2 recovers. Texts shorter than
    ``min_chars`` skip detection entirely.
    """
    candidate: str | None = None
    confidence: float | None = None
    if len(text) >= min_chars:
        candidate, confidence = detector.detect(text)
        corroborated = candidate in (hint, market_default)
        if confidence >= min_confidence or (
            corroborated and confidence >= min_corroborated_confidence
        ):
            return LanguageGuess(candidate, confidence, candidate, LanguageSource.DETECTED)

    if hint:
        return LanguageGuess(hint, confidence, candidate, LanguageSource.STORE_HINT)
    if market_default:
        return LanguageGuess(market_default, confidence, candidate, LanguageSource.MARKET_DEFAULT)
    return LanguageGuess(None, confidence, candidate, LanguageSource.UNKNOWN)
