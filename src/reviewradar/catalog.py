"""App catalog: the declarative list of apps, store listings and markets to track."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reviewradar.domain import Market, Store


class UnknownAppError(LookupError):
    """Raised when a slug is not present in the catalog."""

    def __init__(self, slug: str, available: list[str]) -> None:
        known = ", ".join(sorted(available)) or "none"
        super().__init__(f"Unknown app '{slug}'. Known apps: {known}")
        self.slug = slug


class MarketSpec(BaseModel):
    """A storefront country and review language, as ISO two-letter codes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    country: str = Field(pattern=r"^[a-z]{2}$")
    language: str = Field(pattern=r"^[a-z]{2}$")

    def to_market(self) -> Market:
        return Market(country=self.country, language=self.language)


class AppSpec(BaseModel):
    """One tracked app: its store identifiers and the markets to ingest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64)
    name: str = Field(min_length=1, max_length=255)
    listings: dict[Store, str] = Field(min_length=1)
    markets: tuple[MarketSpec, ...] = Field(min_length=1)
    support_contact: str | None = Field(
        default=None,
        max_length=120,
        description="Where drafted replies send users who need help, e.g. a help centre URL.",
    )

    def iter_markets(self) -> tuple[Market, ...]:
        return tuple(spec.to_market() for spec in self.markets)

    def default_language_by_country(self) -> dict[str, str]:
        """Each storefront's main language, for countries configured with a single language.

        Serves as weak evidence when a review's language cannot be detected reliably.
        Multilingual storefronts (e.g. Canada with en and fr) get no default.
        """
        languages: dict[str, set[str]] = {}
        for market in self.markets:
            languages.setdefault(market.country, set()).add(market.language)
        return {
            country: next(iter(found)) for country, found in languages.items() if len(found) == 1
        }


class Catalog(BaseModel):
    """The full set of tracked apps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    apps: tuple[AppSpec, ...] = ()

    @model_validator(mode="after")
    def _slugs_are_unique(self) -> Self:
        slugs = [app.slug for app in self.apps]
        duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
        if duplicates:
            raise ValueError(f"duplicate app slugs in catalog: {', '.join(duplicates)}")
        return self

    def get(self, slug: str) -> AppSpec:
        for app in self.apps:
            if app.slug == slug:
                return app
        raise UnknownAppError(slug, [app.slug for app in self.apps])

    @classmethod
    def from_yaml(cls, path: Path) -> Catalog:
        with path.open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})
