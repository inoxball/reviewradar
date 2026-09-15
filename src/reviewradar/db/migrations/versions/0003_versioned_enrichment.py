"""Versioned enrichment: pipeline version, text fingerprints and language candidates.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-13

Enrichment tables hold derived, recomputable data, so this revision recreates them
rather than backfilling new NOT NULL columns. Run ``reviewradar enrich`` afterwards.

Embeddings now record the fingerprint of the text they encode instead of the review
revision: edits that leave the text unchanged (a rating change, a developer reply) and
pipeline upgrades that produce the same text no longer trigger re-embedding.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_REVIEW_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_EMBEDDING = sa.LargeBinary().with_variant(VECTOR(), "postgresql")


def upgrade() -> None:
    op.drop_table("review_embeddings")
    op.drop_table("review_enrichments")

    op.create_table(
        "review_enrichments",
        sa.Column("review_id", _REVIEW_ID, autoincrement=False, nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("enrichment_version", sa.Integer(), nullable=False),
        sa.Column("model_text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("language_confidence", sa.Float(), nullable=True),
        sa.Column("language_candidate", sa.String(length=8), nullable=True),
        sa.Column("language_source", sa.String(length=32), nullable=False),
        sa.Column("enriched_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_enrichments_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("review_id", name=op.f("pk_review_enrichments")),
    )
    op.create_table(
        "review_embeddings",
        sa.Column("review_id", _REVIEW_ID, autoincrement=False, nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", _EMBEDDING, nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_embeddings_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("review_id", "model", name=op.f("pk_review_embeddings")),
    )


def downgrade() -> None:
    op.drop_table("review_embeddings")
    op.drop_table("review_enrichments")

    op.create_table(
        "review_enrichments",
        sa.Column("review_id", _REVIEW_ID, autoincrement=False, nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("model_text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("language_confidence", sa.Float(), nullable=True),
        sa.Column("language_source", sa.String(length=32), nullable=False),
        sa.Column("enriched_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_enrichments_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("review_id", name=op.f("pk_review_enrichments")),
    )
    op.create_table(
        "review_embeddings",
        sa.Column("review_id", _REVIEW_ID, autoincrement=False, nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", _EMBEDDING, nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_embeddings_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("review_id", "model", name=op.f("pk_review_embeddings")),
    )
