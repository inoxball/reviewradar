import asyncio
from pathlib import Path

from sqlalchemy import text

from reviewradar.db import migrate
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.web.queries import PanelQueries


def test_head_revision_is_the_newest_bundled_migration() -> None:
    newest = max(
        path.name.split("_")[0] for path in (migrate.MIGRATIONS_DIR / "versions").glob("0*.py")
    )

    assert migrate.head_revision() == newest


async def test_database_info_reports_whether_migrations_are_current(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'status.db').as_posix()}"
    engine = create_engine(database_url)
    try:
        async with create_session_factory(engine)() as session:
            before = await PanelQueries(session).database_info()

        await asyncio.to_thread(migrate.upgrade, "head", database_url=database_url)
        async with create_session_factory(engine)() as session:
            after = await PanelQueries(session).database_info()
            tables = await session.scalar(
                text("SELECT count(*) FROM sqlite_master WHERE name = 'reply_drafts'")
            )
    finally:
        await engine.dispose()

    assert (before.revision, before.migrations_current) == (None, False)
    assert (after.revision, after.migrations_current) == (migrate.head_revision(), True)
    assert tables == 1
