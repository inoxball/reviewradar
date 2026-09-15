"""Dialect-aware ``INSERT … ON CONFLICT`` (Postgres in production, SQLite in tests)."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from reviewradar.db.base import Base

ConflictAwareInsert = postgresql.Insert | sqlite.Insert


def conflict_aware_insert(session: AsyncSession, model: type[Base]) -> ConflictAwareInsert:
    """Start an INSERT supporting ``on_conflict_do_nothing/update`` on the session's backend."""
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return postgresql.insert(model)
    if dialect == "sqlite":
        return sqlite.insert(model)
    raise NotImplementedError(f"conflict-aware inserts are not implemented for {dialect!r}")
