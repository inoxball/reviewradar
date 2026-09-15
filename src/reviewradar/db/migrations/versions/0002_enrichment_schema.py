"""Enrichment schema: redacted model text, language and per-model embeddings.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-13

On Postgres, embeddings use the pgvector ``vector`` type. The column is dimension-less
so several models can coexist; approximate nearest-neighbour search needs one partial
HNSW index per model that casts to that model's dimension, for example::

    CREATE INDEX ix_review_embeddings_minilm_hnsw ON review_embeddings
    USING hnsw ((embedding::vector(384)) vector_cosine_ops)
    WHERE model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';

Such indexes are created alongside the search feature, once a model is chosen.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_REVIEW_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_EMBEDDING = sa.LargeBinary().with_variant(VECTOR(), "postgresql")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

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


def downgrade() -> None:
    op.drop_table("review_embeddings")
    op.drop_table("review_enrichments")
    # The `vector` extension stays installed: other objects may depend on it.
