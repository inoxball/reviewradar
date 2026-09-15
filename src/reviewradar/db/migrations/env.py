"""Alembic environment: runs migrations through the application's async engine."""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from reviewradar.config import get_settings
from reviewradar.db import models  # noqa: F401  (registers tables on the metadata)
from reviewradar.db.base import Base
from reviewradar.db.session import create_engine

target_metadata = Base.metadata


def _database_url() -> str:
    return context.config.get_main_option("sqlalchemy.url") or get_settings().database_url


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_engine(_database_url())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
