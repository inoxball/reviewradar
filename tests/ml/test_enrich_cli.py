"""``reviewradar enrich`` end to end in a subprocess with the real model (``pytest -m ml``).

Guards a Windows-specific failure: importing torch/transformers for the first time inside
a worker thread made the process exit with an access violation although enrichment had
succeeded, which a scheduler would report as a failed job. Only a subprocess exit code
can catch that.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reviewradar.db.migrate import upgrade
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.domain import Store
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from tests.support import NOW, duolingo, make_review

pytestmark = pytest.mark.ml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def seed(database_url: str) -> None:
    engine = create_engine(database_url)
    try:
        async with create_session_factory(engine).begin() as session:
            listing_ids = await CatalogRepository(session).sync_app(duolingo(Store.GOOGLE_PLAY))
            await ReviewRepository(session).upsert_batch(
                listing_ids[Store.GOOGLE_PLAY],
                [
                    make_review("a", body="The app keeps crashing after the update"),
                    make_review("b", body="Love the speaking exercises"),
                ],
                seen_at=NOW,
            )
    finally:
        await engine.dispose()


def test_enrich_command_exits_cleanly(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'enrich.db').as_posix()}"
    upgrade(database_url=database_url)
    asyncio.run(seed(database_url))

    result = subprocess.run(
        [sys.executable, "-m", "reviewradar", "--log-level", "WARNING", "enrich", "duolingo"],
        cwd=PROJECT_ROOT,
        env=os.environ
        | {
            "REVIEWRADAR_DATABASE_URL": database_url,
            "HF_HUB_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "processed 2 reviews" in result.stdout
