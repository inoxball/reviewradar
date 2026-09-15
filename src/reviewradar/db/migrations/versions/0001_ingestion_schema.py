"""Ingestion schema: apps, store listings, reviews, cursors and run log.

Revision ID: 0001
Revises:
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIMESTAMP = sa.DateTime(timezone=True)
_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
_BIGINT_PK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "apps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", _TIMESTAMP, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_apps")),
        sa.UniqueConstraint("slug", name=op.f("uq_apps_slug")),
    )

    op.create_table(
        "app_listings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("app_id", sa.Integer(), nullable=False),
        sa.Column("store", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(
            ["app_id"], ["apps.id"], name=op.f("fk_app_listings_app_id_apps"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_app_listings")),
        sa.UniqueConstraint("store", "external_id", name=op.f("uq_app_listings_store_external_id")),
    )
    op.create_index(op.f("ix_app_listings_app_id"), "app_listings", ["app_id"])

    op.create_table(
        "reviews",
        sa.Column("id", _BIGINT_PK, nullable=False),
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=True),
        sa.Column("language", sa.String(length=8), nullable=True),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("author_hash", sa.String(length=64), nullable=True),
        sa.Column("app_version", sa.String(length=64), nullable=True),
        sa.Column("helpful_count", sa.Integer(), nullable=True),
        sa.Column("developer_reply", sa.Text(), nullable=True),
        sa.Column("developer_replied_at", _TIMESTAMP, nullable=True),
        sa.Column("reviewed_at", _TIMESTAMP, nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("raw", _JSON, nullable=False),
        sa.Column("first_seen_at", _TIMESTAMP, nullable=False),
        sa.Column("last_seen_at", _TIMESTAMP, nullable=False),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name=op.f("ck_reviews_rating_range")),
        sa.ForeignKeyConstraint(
            ["listing_id"],
            ["app_listings.id"],
            name=op.f("fk_reviews_listing_id_app_listings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reviews")),
        sa.UniqueConstraint(
            "listing_id", "external_id", name=op.f("uq_reviews_listing_id_external_id")
        ),
    )
    op.create_index("ix_reviews_listing_id_reviewed_at", "reviews", ["listing_id", "reviewed_at"])

    op.create_table(
        "ingestion_cursors",
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("partition_key", sa.String(length=32), nullable=False),
        sa.Column("newest_reviewed_at", _TIMESTAMP, nullable=False),
        sa.Column("updated_at", _TIMESTAMP, nullable=False),
        sa.ForeignKeyConstraint(
            ["listing_id"],
            ["app_listings.id"],
            name=op.f("fk_ingestion_cursors_listing_id_app_listings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("listing_id", "partition_key", name=op.f("pk_ingestion_cursors")),
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("listing_id", sa.Integer(), nullable=False),
        sa.Column("partition_key", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("cutoff", _TIMESTAMP, nullable=False),
        sa.Column("started_at", _TIMESTAMP, nullable=False),
        sa.Column("finished_at", _TIMESTAMP, nullable=True),
        sa.Column("fetched", sa.Integer(), nullable=False),
        sa.Column("inserted", sa.Integer(), nullable=False),
        sa.Column("updated", sa.Integer(), nullable=False),
        sa.Column("unchanged", sa.Integer(), nullable=False),
        sa.Column("coverage_complete", sa.Boolean(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["listing_id"],
            ["app_listings.id"],
            name=op.f("fk_ingestion_runs_listing_id_app_listings"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_runs")),
    )
    op.create_index(
        "ix_ingestion_runs_listing_id_started_at", "ingestion_runs", ["listing_id", "started_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_runs_listing_id_started_at", table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
    op.drop_table("ingestion_cursors")
    op.drop_index("ix_reviews_listing_id_reviewed_at", table_name="reviews")
    op.drop_table("reviews")
    op.drop_index(op.f("ix_app_listings_app_id"), table_name="app_listings")
    op.drop_table("app_listings")
    op.drop_table("apps")
