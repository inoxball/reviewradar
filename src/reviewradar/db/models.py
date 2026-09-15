"""ORM models for the ingestion schema.

``apps`` ─< ``app_listings`` ─< ``reviews`` ─┬─ ``review_enrichments``  (text, language)
                                           └─< ``review_embeddings``   (one per model)
                          │
                          ├──< ``ingestion_cursors``  (incremental high-water marks)
                          └──< ``ingestion_runs``     (audit log of every attempt)

Reviews hang off a *listing* rather than an app because identity, pagination and
limits are store-specific; adding a source such as Steam or Trustpilot is a new
listing, not a schema change.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import (
    CheckConstraint,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from reviewradar.db.base import Base
from reviewradar.db.types import BigIntPrimaryKey, EmbeddingVector, JSONPayload, UTCDateTime
from reviewradar.domain import LanguageSource, RunStatus, Store, utc_now


def _string_enum(enum_cls: type[StrEnum]) -> Enum:
    """Store enums as their string values in a VARCHAR, avoiding native enum migrations."""
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )


class App(Base):
    """A product tracked across one or more stores."""

    __tablename__ = "apps"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    listings: Mapped[list[AppListing]] = relationship(
        back_populates="app", cascade="all, delete-orphan"
    )


class AppListing(Base):
    """An app's presence in one store, e.g. Duolingo on Google Play."""

    __tablename__ = "app_listings"
    __table_args__ = (UniqueConstraint("store", "external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    app_id: Mapped[int] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"), index=True)
    store: Mapped[Store] = mapped_column(_string_enum(Store))
    external_id: Mapped[str] = mapped_column(String(255))

    app: Mapped[App] = relationship(back_populates="listings")


class Review(Base):
    """The latest known state of a single store review."""

    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("listing_id", "external_id"),
        CheckConstraint("rating BETWEEN 1 AND 5", name="rating_range"),
        Index("ix_reviews_listing_id_reviewed_at", "listing_id", "reviewed_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPrimaryKey, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("app_listings.id", ondelete="CASCADE"))
    external_id: Mapped[str] = mapped_column(String(255))
    country: Mapped[str | None] = mapped_column(String(2))
    language: Mapped[str | None] = mapped_column(String(8))
    rating: Mapped[int] = mapped_column(SmallInteger)
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    author_hash: Mapped[str | None] = mapped_column(String(64))
    app_version: Mapped[str | None] = mapped_column(String(64))
    helpful_count: Mapped[int | None] = mapped_column(Integer)
    developer_reply: Mapped[str | None] = mapped_column(Text)
    developer_replied_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    reviewed_at: Mapped[datetime] = mapped_column(UTCDateTime())
    content_hash: Mapped[str] = mapped_column(String(64))
    raw: Mapped[dict[str, Any]] = mapped_column(JSONPayload)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime())
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime())


class IngestionCursor(Base):
    """Newest review timestamp successfully ingested for one listing partition."""

    __tablename__ = "ingestion_cursors"

    listing_id: Mapped[int] = mapped_column(
        ForeignKey("app_listings.id", ondelete="CASCADE"), primary_key=True
    )
    partition_key: Mapped[str] = mapped_column(String(32), primary_key=True)
    newest_reviewed_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class IngestionRun(Base):
    """Audit record of one ingestion attempt for one listing partition."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (Index("ix_ingestion_runs_listing_id_started_at", "listing_id", "started_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("app_listings.id", ondelete="CASCADE"))
    partition_key: Mapped[str] = mapped_column(String(32))
    status: Mapped[RunStatus] = mapped_column(_string_enum(RunStatus))
    cutoff: Mapped[datetime] = mapped_column(UTCDateTime())
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    fetched: Mapped[int] = mapped_column(Integer, default=0)
    inserted: Mapped[int] = mapped_column(Integer, default=0)
    updated: Mapped[int] = mapped_column(Integer, default=0)
    unchanged: Mapped[int] = mapped_column(Integer, default=0)
    coverage_complete: Mapped[bool | None]
    error: Mapped[str | None] = mapped_column(Text)


class ReviewEnrichment(Base):
    """Model-independent derived fields for one review revision and pipeline version."""

    __tablename__ = "review_enrichments"
    __table_args__ = (
        # Must match reviewradar.search.retrievers.TEXT_SEARCH_DOCUMENT; PostgreSQL only.
        Index(
            "ix_review_enrichments_model_text_fts",
            text("to_tsvector('simple'::regconfig, model_text)"),
            postgresql_using="gin",
        ).ddl_if(dialect="postgresql"),
    )

    review_id: Mapped[int] = mapped_column(
        BigIntPrimaryKey,
        ForeignKey("reviews.id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )
    content_hash: Mapped[str] = mapped_column(String(64))
    enrichment_version: Mapped[int] = mapped_column(Integer)
    model_text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(8))
    language_confidence: Mapped[float | None] = mapped_column(Float)
    language_candidate: Mapped[str | None] = mapped_column(String(8))
    language_source: Mapped[LanguageSource] = mapped_column(_string_enum(LanguageSource))
    enriched_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ReviewEmbedding(Base):
    """A review's vector under one embedding model, valid while its ``text_hash`` matches."""

    __tablename__ = "review_embeddings"

    review_id: Mapped[int] = mapped_column(
        BigIntPrimaryKey,
        ForeignKey("reviews.id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[NDArray[np.float32]] = mapped_column(EmbeddingVector())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ReviewTranslation(Base):
    """A review's model text translated by one model into one target language.

    Valid while ``text_hash`` and ``source_language`` match the review's enrichment.
    """

    __tablename__ = "review_translations"

    review_id: Mapped[int] = mapped_column(
        BigIntPrimaryKey,
        ForeignKey("reviews.id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    target_language: Mapped[str] = mapped_column(String(8), primary_key=True)
    source_language: Mapped[str] = mapped_column(String(8))
    text_hash: Mapped[str] = mapped_column(String(64))
    translated_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class TopicRun(Base):
    """One execution of topic discovery for an app: its parameters and summary."""

    __tablename__ = "topic_runs"
    __table_args__ = (Index("ix_topic_runs_app_id_created_at", "app_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    app_id: Mapped[int] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"))
    embedding_model: Mapped[str] = mapped_column(String(128))
    algorithm: Mapped[str] = mapped_column(String(32))
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONPayload)
    review_count: Mapped[int] = mapped_column(Integer)
    topic_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Topic(Base):
    """A group of semantically similar reviews found by one topic run."""

    __tablename__ = "topics"
    __table_args__ = (Index("ix_topics_run_id", "run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("topic_runs.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(255))
    keywords: Mapped[list[str]] = mapped_column(JSONPayload)
    size: Mapped[int] = mapped_column(Integer)
    average_rating: Mapped[float] = mapped_column(Float)
    negative_share: Mapped[float] = mapped_column(Float)
    languages: Mapped[dict[str, int]] = mapped_column(JSONPayload)


class ReviewTopic(Base):
    """A review's topic within one run, with its cosine distance to the topic centroid."""

    __tablename__ = "review_topics"
    __table_args__ = (Index("ix_review_topics_topic_id_distance", "topic_id", "distance"),)

    run_id: Mapped[int] = mapped_column(
        ForeignKey("topic_runs.id", ondelete="CASCADE"), primary_key=True
    )
    review_id: Mapped[int] = mapped_column(
        BigIntPrimaryKey,
        ForeignKey("reviews.id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    distance: Mapped[float] = mapped_column(Float)


class ReplyDraft(Base):
    """The latest developer reply drafted for a review by one generator.

    ``text`` keeps the ``{app_name}`` and ``{support_contact}`` placeholders. ``version``
    identifies the generator's model and prompt, so drafts made by an older adapter or with
    older guidelines are regenerated instead of reused.
    """

    __tablename__ = "reply_drafts"
    __table_args__ = (
        Index("ix_reply_drafts_review_id_generator", "review_id", "generator", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(
        BigIntPrimaryKey, ForeignKey("reviews.id", ondelete="CASCADE")
    )
    generator: Mapped[str] = mapped_column(String(32))
    version: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
