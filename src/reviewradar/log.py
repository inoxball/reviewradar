"""Logging setup shared by the CLI and background workers."""

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "aiosqlite",
    "sqlalchemy.engine",
    "alembic.runtime",
    "pgserver",
)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, keeping chatty third-party loggers at WARNING."""
    logging.basicConfig(
        level=level.upper(),
        format=_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
