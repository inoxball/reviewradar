"""Programmatic access to Alembic, so migrations ship with the installed package."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(database_url: str | None = None) -> Config:
    """Build an Alembic config pointing at the bundled migrations.

    When ``database_url`` is omitted, the migration environment falls back to the
    application settings (``REVIEWRADAR_DATABASE_URL``).
    """
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if database_url is not None:
        config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade(revision: str = "head", *, database_url: str | None = None) -> None:
    """Apply migrations up to ``revision``."""
    command.upgrade(alembic_config(database_url), revision)


@lru_cache
def head_revision() -> str:
    """The newest migration bundled with this version of the package."""
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:
        raise RuntimeError(f"no migrations found in {MIGRATIONS_DIR}")
    return head


def current_revision(connection: Connection) -> str | None:
    """The migration a database is at, or None when its schema was never migrated."""
    return MigrationContext.configure(connection).get_current_revision()
