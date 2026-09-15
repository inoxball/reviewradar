"""Async engine and session factory construction."""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from reviewradar.db.embedded_postgres import resolve_database_url


def create_engine(database_url: str) -> AsyncEngine:
    """Create an async engine with backend-appropriate settings.

    On SQLite, transactions start with ``BEGIN IMMEDIATE`` so concurrent ingestion
    tasks queue for the write lock (honouring the busy timeout) rather than failing
    when a read transaction tries to upgrade to a write.
    """
    database_url = resolve_database_url(database_url)
    is_sqlite = database_url.startswith("sqlite")
    engine = create_async_engine(
        database_url,
        pool_pre_ping=not is_sqlite,
        connect_args={"timeout": 30} if is_sqlite else {},
    )
    if is_sqlite:
        event.listen(engine.sync_engine, "connect", _configure_sqlite_connection)
        event.listen(engine.sync_engine, "begin", _begin_immediate)
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory whose objects remain usable after commit (no implicit refresh I/O)."""
    return async_sessionmaker(engine, expire_on_commit=False)


def _configure_sqlite_connection(dbapi_connection: Any, _connection_record: Any) -> None:
    # Hand transaction control to SQLAlchemy so `_begin_immediate` decides how to begin.
    dbapi_connection.isolation_level = None
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def _begin_immediate(connection: Connection) -> None:
    connection.exec_driver_sql("BEGIN IMMEDIATE")
