"""Topic discovery: runs, topics and review assignments.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
_REVIEW_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "topic_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("app_id", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=False),
        sa.Column("parameters", _JSON, nullable=False),
        sa.Column("review_count", sa.Integer(), nullable=False),
        sa.Column("topic_count", sa.Integer(), nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["app_id"], ["apps.id"], name=op.f("fk_topic_runs_app_id_apps"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_topic_runs")),
    )
    op.create_index("ix_topic_runs_app_id_created_at", "topic_runs", ["app_id", "created_at"])

    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("keywords", _JSON, nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("average_rating", sa.Float(), nullable=False),
        sa.Column("negative_share", sa.Float(), nullable=False),
        sa.Column("languages", _JSON, nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["topic_runs.id"],
            name=op.f("fk_topics_run_id_topic_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_topics")),
    )
    op.create_index("ix_topics_run_id", "topics", ["run_id"])

    op.create_table(
        "review_topics",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("review_id", _REVIEW_ID, autoincrement=False, nullable=False),
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.Column("distance", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["topic_runs.id"],
            name=op.f("fk_review_topics_run_id_topic_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name=op.f("fk_review_topics_review_id_reviews"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["topic_id"],
            ["topics.id"],
            name=op.f("fk_review_topics_topic_id_topics"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "review_id", name=op.f("pk_review_topics")),
    )
    op.create_index("ix_review_topics_topic_id_distance", "review_topics", ["topic_id", "distance"])


def downgrade() -> None:
    op.drop_index("ix_review_topics_topic_id_distance", table_name="review_topics")
    op.drop_table("review_topics")
    op.drop_index("ix_topics_run_id", table_name="topics")
    op.drop_table("topics")
    op.drop_index("ix_topic_runs_app_id_created_at", table_name="topic_runs")
    op.drop_table("topic_runs")
