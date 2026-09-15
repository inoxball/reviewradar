"""Full-text search index on the redacted model text (PostgreSQL only).

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-13

The indexed expression must stay identical to
``reviewradar.search.retrievers.TEXT_SEARCH_DOCUMENT``; otherwise the planner cannot use
the index. Search is PostgreSQL-only, so this revision is a no-op on other backends.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "ix_review_enrichments_model_text_fts"


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.create_index(
        _INDEX_NAME,
        "review_enrichments",
        [sa.text("to_tsvector('simple'::regconfig, model_text)")],
        postgresql_using="gin",
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.drop_index(_INDEX_NAME, table_name="review_enrichments")
