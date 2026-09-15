"""Shared fixtures.

Database-backed tests run twice: on SQLite, and on a real embedded PostgreSQL with pgvector
(marked ``postgres``) whenever the optional ``pgserver`` package is installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from reviewradar.db import models  # noqa: F401  (registers tables on the metadata)
from reviewradar.db.base import Base
from reviewradar.db.embedded_postgres import to_asyncpg_url
from reviewradar.db.session import create_engine, create_session_factory

_database_numbers = count()


@pytest.fixture(scope="session")
def postgres_server_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """An embedded PostgreSQL server for the session; yields ``postgresql://user@host:port``."""
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="delete")
    root, _, _ = server.get_uri().rpartition("/")
    # Databases created later copy template1, so every test database has pgvector installed.
    _run_admin_sql(f"{root}/template1", "CREATE EXTENSION IF NOT EXISTS vector")
    try:
        yield root
    finally:
        server.cleanup()


@pytest.fixture
def postgres_database_url(postgres_server_root: str) -> Iterator[str]:
    """A fresh, empty PostgreSQL database for one test, dropped afterwards."""
    name = f"test_{next(_database_numbers)}"
    _run_admin_sql(f"{postgres_server_root}/postgres", f'CREATE DATABASE "{name}"')
    try:
        yield to_asyncpg_url(f"{postgres_server_root}/{name}")
    finally:
        _run_admin_sql(
            f"{postgres_server_root}/postgres", f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'
        )


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def database_url(request: pytest.FixtureRequest, tmp_path: Path) -> str:
    """An empty database on each supported backend (in-memory SQLite is per-connection)."""
    if request.param == "postgres":
        return str(request.getfixturevalue("postgres_database_url"))
    return f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
async def postgres_session_factory(
    postgres_database_url: str,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """For features that exist only on PostgreSQL, such as search."""
    engine = create_engine(postgres_database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


def _run_admin_sql(uri: str, *statements: str) -> None:
    """Run administrative SQL on a private event loop in a worker thread.

    Sync fixtures must not touch the main thread's event loop, which pytest-asyncio owns.
    """

    async def execute() -> None:
        connection = await asyncpg.connect(uri)
        try:
            for statement in statements:
                await connection.execute(statement)
        finally:
            await connection.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(asyncio.run, execute()).result()
