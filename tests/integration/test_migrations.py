"""Guards against drift between the ORM models and the hand-written migrations, per backend."""

import asyncio
from typing import Any

import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import Connection

from reviewradar.db import models  # noqa: F401  (registers tables on the metadata)
from reviewradar.db.base import Base
from reviewradar.db.migrate import alembic_config
from reviewradar.db.session import create_engine


def test_migrations_produce_the_model_schema(database_url: str) -> None:
    command.upgrade(alembic_config(database_url), "head")

    assert asyncio.run(_inspect(database_url, _schema_diff)) == []


def test_migrations_downgrade_to_an_empty_schema(database_url: str) -> None:
    config = alembic_config(database_url)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    assert asyncio.run(_inspect(database_url, _table_names)) == ["alembic_version"]


async def _inspect(database_url: str, inspector: Any) -> Any:
    engine = create_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(inspector)
    finally:
        await engine.dispose()


def _schema_diff(connection: Connection) -> list[Any]:
    context = MigrationContext.configure(connection, opts={"compare_type": True})
    return list(compare_metadata(context, Base.metadata))


def _table_names(connection: Connection) -> list[str]:
    return sorted(sa.inspect(connection).get_table_names())
