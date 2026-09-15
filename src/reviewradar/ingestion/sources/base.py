"""The contract every store integration implements."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from reviewradar.domain import Market, ReviewPage, Store


class ReviewSource(Protocol):
    """A provider of reviews for one store.

    Contract:

    * ``fetch_pages`` yields pages ordered **newest first**, which lets the caller stop
      paginating as soon as a page reaches past the time window it cares about.
    * Transient failures are retried inside the source; anything raised is final.
    * Returned reviews satisfy the :class:`~reviewradar.domain.FetchedReview` invariants
      (UTC timestamps, no raw author identities).
    """

    @property
    def store(self) -> Store: ...

    def partition_key(self, market: Market) -> str:
        """Identify the slice of the store's review stream that ``market`` maps to.

        Markets that share a key receive identical reviews, so the orchestrator fetches
        each key once. The key also scopes incremental cursors.
        """
        ...

    def fetch_pages(self, external_app_id: str, market: Market) -> AsyncGenerator[ReviewPage]:
        """Stream review pages for one app in one market, newest first."""
        ...
