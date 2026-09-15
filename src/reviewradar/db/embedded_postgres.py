"""Embedded PostgreSQL with pgvector for local development and tests, via ``pgserver``.

Set ``REVIEWRADAR_DATABASE_URL=pgserver:///.pgdata`` to work against a real PostgreSQL
without Docker. The server starts on first use and keeps running between commands, which
spares every command a multi-second startup; stop it with ``reviewradar db stop``. Data
lives in the given directory. This is a development convenience, not a production setup.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

EMBEDDED_SCHEME = "pgserver:///"
STARTUP_ATTEMPTS = 6
STARTUP_RETRY_SECONDS = 5.0
"""pgserver waits only 10 s for PostgreSQL to start. After an unclean shutdown on Windows,
crash recovery can take longer (it retries a locked log file for up to 30 s), so the
resolver keeps attaching until the server is ready."""
_MISSING_PACKAGE = (
    "pgserver:/// URLs require the optional 'pgserver' package: pip install -e '.[dev]'"
)


def embedded_data_dir(url: str) -> Path | None:
    """The data directory of a ``pgserver:///`` URL, or ``None`` for any other URL."""
    if not url.startswith(EMBEDDED_SCHEME):
        return None
    return Path(url.removeprefix(EMBEDDED_SCHEME)).resolve()


def resolve_database_url(url: str) -> str:
    """Return a SQLAlchemy URL, starting (or attaching to) the embedded server if requested."""
    data_dir = embedded_data_dir(url)
    if data_dir is None:
        return url
    try:
        from pgserver.postgres_server import get_server
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_MISSING_PACKAGE) from exc

    for attempt in range(1, STARTUP_ATTEMPTS + 1):
        try:
            server = get_server(data_dir, cleanup_mode=None)
            return to_asyncpg_url(server.get_uri())
        except subprocess.TimeoutExpired:
            if attempt == STARTUP_ATTEMPTS:
                raise
            logger.warning(
                "embedded PostgreSQL in %s is still starting (attempt %d/%d); retrying",
                data_dir,
                attempt,
                STARTUP_ATTEMPTS,
            )
            time.sleep(STARTUP_RETRY_SECONDS)
    raise AssertionError("unreachable: the last attempt returns or raises")


def stop_embedded_server(url: str) -> bool:
    """Stop the server behind a ``pgserver:///`` URL; return ``False`` if it was not running."""
    data_dir = embedded_data_dir(url)
    if data_dir is None:
        raise ValueError(f"not an embedded database URL: {url!r}")
    if not (data_dir / "postmaster.pid").exists():
        return False
    try:
        pgserver = importlib.import_module("pgserver")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_MISSING_PACKAGE) from exc

    # pgserver generates one wrapper per bundled PostgreSQL binary when it is imported.
    pgserver.pg_ctl(["stop", "--mode=fast", "--wait"], pgdata=data_dir)
    return True


def to_asyncpg_url(uri: str) -> str:
    """Point a ``postgresql://`` URI at SQLAlchemy's asyncpg driver."""
    return uri.replace("postgresql://", "postgresql+asyncpg://", 1)
