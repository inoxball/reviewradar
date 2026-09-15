"""Reply drafts: the latest draft per review and generator.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_REVIEW_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "reply_drafts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("review_id", _REVIEW_ID, nullable=False),
        sa.Column("generator", sa.String(length=32), nullable=False),
        sa.Column("version", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_reply_drafts_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reply_drafts")),
    )
    op.create_index(
        "ix_reply_drafts_review_id_generator",
        "reply_drafts",
        ["review_id", "generator"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_reply_drafts_review_id_generator", table_name="reply_drafts")
    op.drop_table("reply_drafts")
