"""The ``pgserver:///`` development database: started on demand, reused, and stopped."""

import asyncio
import subprocess
from pathlib import Path

import asyncpg
import pytest

from reviewradar.db import embedded_postgres
from reviewradar.db.embedded_postgres import (
    embedded_data_dir,
    resolve_database_url,
    stop_embedded_server,
    to_asyncpg_url,
)


def test_other_urls_pass_through_unchanged() -> None:
    url = "postgresql+asyncpg://reviewradar@localhost:5432/reviewradar"

    assert resolve_database_url(url) == url
    assert embedded_data_dir(url) is None


def test_stop_rejects_urls_without_an_embedded_server() -> None:
    with pytest.raises(ValueError, match="not an embedded database URL"):
        stop_embedded_server("sqlite+aiosqlite:///./reviewradar.db")


def test_keeps_attaching_while_the_server_is_still_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    postgres_server = pytest.importorskip("pgserver.postgres_server")
    attempts: list[Path] = []

    class RunningServer:
        def get_uri(self) -> str:
            return "postgresql://postgres:@127.0.0.1:5432/postgres"

    def slow_start(data_dir: Path, cleanup_mode: str | None) -> RunningServer:
        attempts.append(data_dir)
        if len(attempts) < 3:
            raise subprocess.TimeoutExpired(cmd="pg_ctl start", timeout=10)
        return RunningServer()

    monkeypatch.setattr(postgres_server, "get_server", slow_start)
    monkeypatch.setattr(embedded_postgres, "STARTUP_RETRY_SECONDS", 0)

    url = resolve_database_url(f"pgserver:///{tmp_path.as_posix()}")

    assert url == "postgresql+asyncpg://postgres:@127.0.0.1:5432/postgres"
    assert len(attempts) == 3


def test_gives_up_after_the_last_startup_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    postgres_server = pytest.importorskip("pgserver.postgres_server")

    def never_starts(data_dir: Path, cleanup_mode: str | None) -> None:
        raise subprocess.TimeoutExpired(cmd="pg_ctl start", timeout=10)

    monkeypatch.setattr(postgres_server, "get_server", never_starts)
    monkeypatch.setattr(embedded_postgres, "STARTUP_RETRY_SECONDS", 0)

    with pytest.raises(subprocess.TimeoutExpired):
        resolve_database_url(f"pgserver:///{tmp_path.as_posix()}")


def test_points_postgres_uris_at_the_asyncpg_driver() -> None:
    assert to_asyncpg_url("postgresql://postgres:@127.0.0.1:5432/postgres") == (
        "postgresql+asyncpg://postgres:@127.0.0.1:5432/postgres"
    )


@pytest.mark.postgres
def test_starts_on_demand_reuses_the_running_server_and_stops(tmp_path: Path) -> None:
    pytest.importorskip("pgserver")
    url = f"pgserver:///{(tmp_path / 'pgdata').as_posix()}"

    try:
        resolved = resolve_database_url(url)
        assert resolve_database_url(url) == resolved
        assert asyncio.run(_select_one(resolved)) == 1
    finally:
        stopped = stop_embedded_server(url)

    assert stopped
    assert not stop_embedded_server(url)


async def _select_one(sqlalchemy_url: str) -> int:
    connection = await asyncpg.connect(sqlalchemy_url.replace("+asyncpg", "", 1))
    try:
        return int(await connection.fetchval("SELECT 1"))
    finally:
        await connection.close()
