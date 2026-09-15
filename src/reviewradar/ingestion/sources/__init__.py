"""Store-specific review sources behind a common :class:`ReviewSource` protocol."""

from reviewradar.ingestion.sources.app_store import AppStoreSource
from reviewradar.ingestion.sources.base import ReviewSource
from reviewradar.ingestion.sources.google_play import GooglePlaySource

__all__ = ["AppStoreSource", "GooglePlaySource", "ReviewSource"]
