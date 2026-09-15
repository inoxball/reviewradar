from pathlib import Path

import pytest
from pydantic import ValidationError

from reviewradar.catalog import Catalog, MarketSpec, UnknownAppError
from reviewradar.domain import Store

PROJECT_CATALOG = Path(__file__).resolve().parents[2] / "config" / "apps.yaml"

DUOLINGO = {
    "slug": "duolingo",
    "name": "Duolingo",
    "listings": {"google_play": "com.duolingo"},
    "markets": [{"country": "us", "language": "en"}],
}


def test_project_catalog_is_valid() -> None:
    duolingo = Catalog.from_yaml(PROJECT_CATALOG).get("duolingo")

    assert duolingo.listings == {Store.GOOGLE_PLAY: "com.duolingo", Store.APP_STORE: "570060128"}
    assert len(duolingo.iter_markets()) >= 1


def test_rejects_duplicate_slugs() -> None:
    with pytest.raises(ValidationError, match="duplicate app slugs in catalog: duolingo"):
        Catalog.model_validate({"apps": [DUOLINGO, DUOLINGO]})


def test_unknown_app_error_lists_known_slugs() -> None:
    catalog = Catalog.model_validate({"apps": [DUOLINGO]})

    with pytest.raises(UnknownAppError, match="Known apps: duolingo"):
        catalog.get("babbel")


@pytest.mark.parametrize(("country", "language"), [("USA", "en"), ("us", "EN"), ("u", "en")])
def test_rejects_non_iso_market_codes(country: str, language: str) -> None:
    with pytest.raises(ValidationError):
        MarketSpec(country=country, language=language)


def test_default_language_by_country_skips_multilingual_storefronts() -> None:
    markets = [
        {"country": "us", "language": "en"},
        {"country": "ca", "language": "en"},
        {"country": "ca", "language": "fr"},
    ]
    (app,) = Catalog.model_validate({"apps": [DUOLINGO | {"markets": markets}]}).apps

    assert app.default_language_by_country() == {"us": "en"}
