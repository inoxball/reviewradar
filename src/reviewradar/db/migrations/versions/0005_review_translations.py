"""Review translations produced by local machine translation models.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_REVIEW_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "review_translations",
        sa.Column("review_id", _REVIEW_ID, autoincrement=False, nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("target_language", sa.String(length=8), nullable=False),
        sa.Column("source_language", sa.String(length=8), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_translations_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "review_id", "model", "target_language", name=op.f("pk_review_translations")
        ),
    )


def downgrade() -> None:
    op.drop_table("review_translations")
