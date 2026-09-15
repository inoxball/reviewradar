"""Store-agnostic domain types shared by ingestion, storage and analysis."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any


class Store(StrEnum):
    """Marketplaces reviews are collected from."""

    GOOGLE_PLAY = "google_play"
    APP_STORE = "app_store"


class RunStatus(StrEnum):
    """Lifecycle of a single ingestion run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LanguageSource(StrEnum):
    """How a review's language was determined, from most to least reliable."""

    DETECTED = "detected"  # confident, or corroborated by a hint or market default
    STORE_HINT = "store_hint"  # the store served the review for a requested language
    MARKET_DEFAULT = "market_default"  # the storefront's main language, from the catalog
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Market:
    """A storefront country paired with the review language requested from it."""

    country: str
    language: str

    def __str__(self) -> str:
        return f"{self.country}-{self.language}"


@dataclass(frozen=True, slots=True)
class FetchedReview:
    """A review as observed in a store, normalized to a store-agnostic shape.

    Sources guarantee two invariants: timestamps are timezone-aware UTC, and raw
    author identities never leave the source (only a pseudonymous hash does).

    ``country`` is the storefront the review was published in, or ``None`` when the
    store does not expose it (Google Play serves reviews per language, not per country).
    """

    external_id: str
    rating: int
    body: str
    reviewed_at: datetime
    country: str | None
    language: str | None = None
    title: str | None = None
    author_hash: str | None = None
    app_version: str | None = None
    helpful_count: int | None = None
    developer_reply: str | None = None
    developer_replied_at: datetime | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not 1 <= self.rating <= 5:
            raise ValueError(f"rating must be within 1..5, got {self.rating}")
        _require_utc("reviewed_at", self.reviewed_at)
        if self.developer_replied_at is not None:
            _require_utc("developer_replied_at", self.developer_replied_at)

    @property
    def content_hash(self) -> str:
        """Fingerprint of user-visible content; changes when a review is edited or answered.

        Vote counts are deliberately excluded: they churn constantly and would turn
        every re-fetch into a write without changing anything analysis cares about.
        """
        parts = (str(self.rating), self.title or "", self.body, self.developer_reply or "")
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewPage:
    """One page of reviews from a source, ordered newest first.

    ``hit_provider_limit`` is set on the final page when the provider refuses to serve
    older history (the App Store feed stops after 10 pages), meaning older reviews may
    exist but are unreachable in this run.
    """

    reviews: Sequence[FetchedReview]
    hit_provider_limit: bool = False


def pseudonymize_author(store: Store, author_key: str) -> str:
    """Return a stable pseudonymous identifier for a review author.

    Lets analytics count repeat reviewers without storing names. This is
    pseudonymization, not anonymization: switch to a keyed hash (HMAC with a managed
    secret) before exposing the value outside the analytics store.
    """
    return hashlib.sha256(f"{store}:{author_key}".encode()).hexdigest()


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a timezone-aware UTC datetime, got {value!r}")
