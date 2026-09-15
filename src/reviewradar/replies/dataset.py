"""Reply-drafting datasets: published developer replies plus hand-written references.

- **Published** replies come from apps whose support teams answer most reviews. They are
  cleaned into brand-neutral, anonymous targets, deduplicated, and capped per opening so
  templates do not dominate.
- **Written** references answer reviews of the target app following the reply guidelines.

Every example lands in train or eval by a stable hash of its key, so rebuilding the
dataset after new reviews arrive never moves an existing example between splits.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import Catalog
from reviewradar.config import ReplySettings
from reviewradar.db.models import App, AppListing, Review, ReviewEmbedding, ReviewEnrichment
from reviewradar.domain import Store
from reviewradar.replies.cleaning import PLACEHOLDERS, clean_developer_reply
from reviewradar.replies.prompting import REPLY_CHAR_LIMIT

OPENING_WORDS = 6
MAX_SHARED_OPENINGS = 5
"""At most 5 published replies may share their first 6 words. On the first dataset this kept
74% of replies and raised distinct 5-word openings from 36% to 48% (notes §13.3)."""
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")

LanguageDetector = Callable[[str], str | None]
ReviewKey = tuple[str, Store, str]


class ReplySource(StrEnum):
    PUBLISHED = "published"
    WRITTEN = "written"


class Split(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True, slots=True)
class ReviewRow:
    """An enriched review as the dataset needs it; ``reply`` is the published one, if any."""

    app: str
    store: Store
    external_id: str
    rating: int
    language: str | None
    review: str
    reply: str | None


@dataclass(frozen=True, slots=True)
class ReplyExample:
    app: str
    store: Store
    external_id: str
    source: ReplySource
    rating: int
    language: str | None
    review: str
    reply: str

    @property
    def key(self) -> str:
        return f"{self.app}:{self.store.value}:{self.external_id}"

    def to_record(self, split: Split) -> dict[str, Any]:
        return {
            "key": self.key,
            "split": split.value,
            "source": self.source.value,
            "app": self.app,
            "store": self.store.value,
            "rating": self.rating,
            "language": self.language,
            "review": self.review,
            "reply": self.reply,
        }


@dataclass(frozen=True, slots=True)
class WrittenReply:
    app: str
    store: Store
    external_id: str
    reply: str


@dataclass(slots=True)
class DatasetReport:
    kept: Counter[tuple[ReplySource, Split]] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    languages: Counter[str] = field(default_factory=Counter)


def split_of(key: str, eval_fraction: float) -> Split:
    """The same key always lands in the same split."""
    position = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    return Split.EVAL if position < eval_fraction else Split.TRAIN


def load_written_replies(directory: Path) -> list[WrittenReply]:
    """Read every ``*.jsonl`` file of written references, validating placeholders."""
    replies: list[WrittenReply] = []
    for path in sorted(directory.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                reply = WrittenReply(
                    record["app"], Store(record["store"]), str(record["id"]), record["reply"]
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: invalid written reply: {exc}") from exc
            unknown = set(_PLACEHOLDER.findall(reply.reply)) - PLACEHOLDERS
            if unknown:
                raise ValueError(f"{path}:{number}: unknown placeholders {sorted(unknown)}")
            replies.append(reply)
    return replies


def curate_published(
    rows: Iterable[ReviewRow],
    *,
    brand_names: Iterable[str],
    detect_language: LanguageDetector,
) -> tuple[list[ReplyExample], Counter[str]]:
    """Clean published replies and drop the ones that would teach the wrong thing."""
    brands = list(brand_names)
    examples: list[ReplyExample] = []
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    openings: Counter[str] = Counter()
    for row in rows:
        reply = clean_developer_reply(row.reply or "", brand_names=brands)
        if reply is None:
            dropped["truncated or too short"] += 1
            continue
        if len(reply) > REPLY_CHAR_LIMIT:
            dropped["over the character limit"] += 1
            continue
        if row.language and detect_language(_PLACEHOLDER.sub("", reply)) != row.language:
            dropped["not in the review's language"] += 1
            continue
        normalized = " ".join(reply.lower().split())
        opening = " ".join(normalized.split()[:OPENING_WORDS])
        if normalized in seen:
            dropped["duplicate"] += 1
            continue
        if openings[opening] >= MAX_SHARED_OPENINGS:
            dropped["template opening"] += 1
            continue
        seen.add(normalized)
        openings[opening] += 1
        examples.append(_example(row, ReplySource.PUBLISHED, reply))
    return examples, dropped


def attach_written(
    replies: Iterable[WrittenReply], reviews: Mapping[ReviewKey, ReviewRow]
) -> tuple[list[ReplyExample], Counter[str]]:
    """Pair written references with their reviews."""
    examples: list[ReplyExample] = []
    dropped: Counter[str] = Counter()
    for written in replies:
        row = reviews.get((written.app, written.store, written.external_id))
        if row is None:
            dropped["written reply without a stored review"] += 1
            continue
        examples.append(_example(row, ReplySource.WRITTEN, written.reply))
    return examples, dropped


def _example(row: ReviewRow, source: ReplySource, reply: str) -> ReplyExample:
    return ReplyExample(
        app=row.app,
        store=row.store,
        external_id=row.external_id,
        source=source,
        rating=row.rating,
        language=row.language,
        review=row.review,
        reply=reply,
    )


class ReplyDatasetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def published(self, app_slugs: Sequence[str]) -> list[ReviewRow]:
        """Enriched reviews with a published developer reply."""
        statement = _review_rows().where(
            App.slug.in_(app_slugs), Review.developer_reply.is_not(None)
        )
        return [ReviewRow(*row) for row in await self._session.execute(statement)]

    async def reviews(
        self, app_slugs: Iterable[str], external_ids: Iterable[str]
    ) -> dict[ReviewKey, ReviewRow]:
        statement = _review_rows().where(
            App.slug.in_(sorted(set(app_slugs))), Review.external_id.in_(sorted(set(external_ids)))
        )
        rows = [ReviewRow(*row) for row in await self._session.execute(statement)]
        return {(row.app, row.store, row.external_id): row for row in rows}

    async def embeddings(
        self, keys: Iterable[ReviewKey], *, model: str
    ) -> dict[ReviewKey, NDArray[np.float32]]:
        """Current embeddings of the given reviews, for retrieving similar examples."""
        wanted = set(keys)
        statement = (
            select(App.slug, AppListing.store, Review.external_id, ReviewEmbedding.embedding)
            .select_from(Review)
            .join(AppListing, Review.listing_id == AppListing.id)
            .join(App, AppListing.app_id == App.id)
            .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
            .join(
                ReviewEmbedding,
                and_(
                    ReviewEmbedding.review_id == Review.id,
                    ReviewEmbedding.model == model,
                    ReviewEmbedding.text_hash == ReviewEnrichment.text_hash,
                ),
            )
            .where(Review.external_id.in_(sorted({key[2] for key in wanted})))
        )
        found: dict[ReviewKey, NDArray[np.float32]] = {}
        for slug, store, external_id, vector in await self._session.execute(statement):
            if (slug, store, external_id) in wanted:
                found[(slug, store, external_id)] = np.asarray(vector, dtype=np.float32)
        return found


def _review_rows() -> Select[Any]:
    return (
        select(
            App.slug,
            AppListing.store,
            Review.external_id,
            Review.rating,
            ReviewEnrichment.language,
            ReviewEnrichment.model_text,
            Review.developer_reply,
        )
        .select_from(Review)
        .join(AppListing, Review.listing_id == AppListing.id)
        .join(App, AppListing.app_id == App.id)
        .join(ReviewEnrichment, ReviewEnrichment.review_id == Review.id)
        .order_by(App.slug, Review.id)
    )


class ReplyDatasetBuilder:
    """Builds ``train.jsonl`` and ``eval.jsonl`` from the database and written references."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        catalog: Catalog,
        settings: ReplySettings,
        detect_language: LanguageDetector,
    ) -> None:
        self._session_factory = session_factory
        self._catalog = catalog
        self._settings = settings
        self._detect_language = detect_language

    async def build(self) -> DatasetReport:
        written = load_written_replies(self._settings.written_dir)
        async with self._session_factory() as session:
            repository = ReplyDatasetRepository(session)
            published_rows = await repository.published(self._settings.source_apps)
            written_rows = await repository.reviews(
                (reply.app for reply in written), (reply.external_id for reply in written)
            )

        published, dropped = curate_published(
            published_rows, brand_names=self._brand_names(), detect_language=self._detect_language
        )
        references, missing = attach_written(written, written_rows)
        report = DatasetReport(dropped=dropped + missing)

        records: dict[Split, list[dict[str, Any]]] = {Split.TRAIN: [], Split.EVAL: []}
        for example in [*published, *references]:
            fraction = (
                self._settings.eval_fraction_written
                if example.source is ReplySource.WRITTEN
                else self._settings.eval_fraction_published
            )
            split = split_of(example.key, fraction)
            records[split].append(example.to_record(split))
            report.kept[(example.source, split)] += 1
            report.languages[example.language or "unknown"] += 1

        self._settings.dataset_dir.mkdir(parents=True, exist_ok=True)
        for split, items in records.items():
            path = self._settings.dataset_dir / f"{split.value}.jsonl"
            path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
                encoding="utf-8",
            )
        return report

    def _brand_names(self) -> list[str]:
        names: list[str] = []
        for slug in self._settings.source_apps:
            app = self._catalog.get(slug)
            names += [app.name, app.slug]
        return names
