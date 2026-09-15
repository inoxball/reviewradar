from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.anomalies.service import AnomalyService
from reviewradar.catalog import Catalog
from reviewradar.config import (
    AnomalySettings,
    EnrichmentSettings,
    SearchSettings,
    TopicSettings,
    TranslationSettings,
)
from reviewradar.db.migrate import head_revision
from reviewradar.domain import FetchedReview, RunStatus, Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import (
    CatalogRepository,
    IngestionStateRepository,
    ReviewRepository,
    UpsertStats,
)
from reviewradar.search.service import SearchService
from reviewradar.topics.clustering import KMeansClusterer
from reviewradar.topics.service import TopicService
from reviewradar.translation.service import TranslationService
from reviewradar.web.app import PanelServices, create_app
from tests.fakes import FakeEmbedder, FakeTranslator, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

APP = duolingo(Store.GOOGLE_PLAY, Store.APP_STORE)
# A wide hashing space keeps the fake embedder free of token collisions in these tests.
EMBEDDING_DIMENSION = 1024

JUDGMENTS = {
    "app": "duolingo",
    "annotator": "test",
    "queries": [
        {
            "id": "ads",
            "text": "too many ads",
            "judgments": [
                {"store": "google_play", "id": "ads", "grade": 2},
                {"store": "app_store", "id": "partial", "grade": 1},
            ],
        }
    ],
}


def in_language(
    external_id: str, language: str, body: str, *, rating: int, age_hours: int
) -> FetchedReview:
    review = make_review(external_id, body=body, rating=rating, age=timedelta(hours=age_hours))
    return replace(review, country=None, language=language)


@pytest.fixture
def translator() -> FakeTranslator:
    return FakeTranslator()


@pytest.fixture
async def client(
    postgres_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    translator: FakeTranslator,
) -> AsyncIterator[httpx.AsyncClient]:
    embedder = FakeEmbedder(dimension=EMBEDDING_DIMENSION)
    async with postgres_session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        reviews = ReviewRepository(session)
        await reviews.upsert_batch(
            listing_ids[Store.GOOGLE_PLAY],
            [
                in_language("ads", "en", "too many ads in every lesson", rating=1, age_hours=1),
                in_language("owl", "en", "love the owl mascot", rating=5, age_hours=2),
                in_language("pt", "pt", "muitos anúncios", rating=2, age_hours=3),
            ],
            seen_at=NOW,
        )
        await reviews.upsert_batch(
            listing_ids[Store.APP_STORE],
            [make_review("partial", body="many ads here", rating=2, age=timedelta(hours=4))],
            seen_at=NOW,
        )
        state = IngestionStateRepository(session)
        run_id = await state.start_run(
            listing_ids[Store.APP_STORE], "us", cutoff=NOW - timedelta(days=1), started_at=NOW
        )
        await state.finish_run(
            run_id,
            status=RunStatus.SUCCEEDED,
            fetched=1,
            stats=UpsertStats(inserted=1),
            coverage_complete=False,
            error=None,
            finished_at=NOW,
        )
        await state.advance_cursor(listing_ids[Store.APP_STORE], "us", NOW, updated_at=NOW)

    # An unsure detector lets the store hint decide; the App Store review falls back to "en".
    await EnrichmentService(
        postgres_session_factory,
        FixedLanguageDetector("zz", confidence=0.1),
        embedder,
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)

    judgments_path = tmp_path / "judgments.yaml"
    judgments_path.write_text(yaml.safe_dump(JUDGMENTS), encoding="utf-8")
    services = PanelServices(
        catalog=Catalog(apps=(APP,)),
        session_factory=postgres_session_factory,
        search=SearchService(postgres_session_factory, embedder, SearchSettings()),
        translation=TranslationService(
            postgres_session_factory, translator, TranslationSettings(), clock=lambda: NOW
        ),
        judgments_path=judgments_path,
        anomalies=AnomalyService(postgres_session_factory, AnomalySettings()),
        topics=TopicService(
            postgres_session_factory,
            lambda topic_count: KMeansClusterer(topic_count, seed=0),
            TopicSettings(min_reviews=2, min_words=1),
            embedding_model=embedder.model_name,
            translation_model=translator.model_name,
            target_language="en",
            clock=lambda: NOW,
        ),
    )

    @asynccontextmanager
    async def provide_services() -> AsyncIterator[PanelServices]:
        yield services

    app = create_app(provide_services)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://panel"
        ) as client,
    ):
        yield client


async def get_json(client: httpx.AsyncClient, path: str, **params: Any) -> Any:
    response = await client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def test_serves_the_single_page_ui(client: httpx.AsyncClient) -> None:
    page = await client.get("/")

    assert page.status_code == 200
    assert "<title>ReviewRadar</title>" in page.text


async def test_documents_every_endpoint_in_openapi(client: httpx.AsyncClient) -> None:
    paths = set((await get_json(client, "/openapi.json"))["paths"])

    assert paths == {
        "/api/system",
        "/api/apps",
        "/api/apps/{slug}/overview",
        "/api/apps/{slug}/ingestion",
        "/api/apps/{slug}/reviews",
        "/api/apps/{slug}/search",
        "/api/apps/{slug}/compare",
        "/api/apps/{slug}/topics",
        "/api/apps/{slug}/topics/{topic_id}/reviews",
        "/api/apps/{slug}/topics/runs",
        "/api/apps/{slug}/anomalies",
        "/api/apps/{slug}/anomalies/monitor",
        "/api/replies/generators",
        "/api/replies/drafts",
        "/api/translations",
        "/api/evaluation",
    }


async def test_system_reports_database_and_models(client: httpx.AsyncClient) -> None:
    body = await get_json(client, "/api/system")

    assert body["database"]["backend"] == "postgresql"
    assert body["database"]["pgvector"]
    # Test databases are built from the models, not migrated, so the panel reports it.
    assert body["database"]["revision"] is None
    assert body["database"]["head_revision"] == head_revision()
    assert body["database"]["migrations_current"] is False
    assert body["translation"] == {
        "model": "fake/translator",
        "target_language": "en",
        "loaded": True,
    }


async def test_lists_catalog_apps(client: httpx.AsyncClient) -> None:
    assert await get_json(client, "/api/apps") == [{"slug": "duolingo", "name": "Duolingo"}]


async def test_overview_summarizes_corpus_ratings_and_pipeline(client: httpx.AsyncClient) -> None:
    overview = await get_json(client, "/api/apps/duolingo/overview")

    assert overview["total_reviews"] == 4
    assert overview["languages"] == 2
    assert {market["store"] for market in overview["markets"]} == {"google_play", "app_store"}
    assert {(row["store"], row["rating"], row["review_count"]) for row in overview["ratings"]} == {
        ("google_play", 1, 1),
        ("google_play", 2, 1),
        ("google_play", 5, 1),
        ("app_store", 2, 1),
    }
    assert overview["pipeline"] | {"language_sources": None} == {
        "reviews": 4,
        "enriched": 4,
        "embedded": 4,
        "translated": 0,
        "language_sources": None,
    }


async def test_unknown_app_is_not_found(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/apps/babbel/overview")

    assert response.status_code == 404
    assert "Known apps: duolingo" in response.json()["detail"]


async def test_ingestion_status_lists_runs_and_cursors(client: httpx.AsyncClient) -> None:
    body = await get_json(client, "/api/apps/duolingo/ingestion")

    (run,) = body["runs"]
    assert (run["store"], run["partition_key"], run["status"], run["inserted"]) == (
        "app_store",
        "us",
        "succeeded",
        1,
    )
    assert run["coverage_complete"] is False
    assert [cursor["partition_key"] for cursor in body["cursors"]] == ["us"]


async def test_browses_reviews_with_sorting_filters_and_pagination(
    client: httpx.AsyncClient,
) -> None:
    newest = await get_json(client, "/api/apps/duolingo/reviews", limit=2)
    lowest_on_play = await get_json(
        client, "/api/apps/duolingo/reviews", sort="lowest_rating", store="google_play"
    )
    second_page = await get_json(client, "/api/apps/duolingo/reviews", limit=2, offset=2)

    assert newest["total"] == 4
    assert [item["external_id"] for item in newest["items"]] == ["ads", "owl"]
    assert [item["external_id"] for item in lowest_on_play["items"]] == ["ads", "pt", "owl"]
    assert [item["external_id"] for item in second_page["items"]] == ["pt", "partial"]


async def test_search_returns_ranked_reviews_and_latency(client: httpx.AsyncClient) -> None:
    body = await get_json(
        client, "/api/apps/duolingo/search", q="too many ads", mode="semantic", limit=2
    )

    assert [hit["external_id"] for hit in body["results"]] == ["ads", "partial"]
    assert body["results"][0]["review_id"] > 0
    assert body["took_ms"] >= 0


async def test_search_applies_filters(client: httpx.AsyncClient) -> None:
    body = await get_json(
        client, "/api/apps/duolingo/search", q="ads", mode="semantic", store="app_store"
    )

    assert [hit["external_id"] for hit in body["results"]] == ["partial"]


@pytest.mark.parametrize(
    "params",
    [{"q": ""}, {"q": "   "}, {"q": "ads", "mode": "fuzzy"}, {"q": "ads", "limit": 0}],
    ids=["empty", "blank", "unknown-mode", "zero-limit"],
)
async def test_rejects_invalid_search_requests(
    client: httpx.AsyncClient, params: dict[str, Any]
) -> None:
    response = await client.get("/api/apps/duolingo/search", params=params)

    assert response.status_code == 422


async def test_compare_runs_every_mode(client: httpx.AsyncClient) -> None:
    body = await get_json(client, "/api/apps/duolingo/compare", q="too many ads")

    ranked = {
        entry["mode"]: [hit["external_id"] for hit in entry["results"]] for entry in body["modes"]
    }
    assert ranked["lexical"] == ["ads"]
    assert ranked["semantic"][:2] == ["ads", "partial"]


async def test_translates_on_demand_and_attaches_translations_everywhere(
    client: httpx.AsyncClient, translator: FakeTranslator
) -> None:
    page = await get_json(client, "/api/apps/duolingo/reviews", language="pt")
    (portuguese,) = page["items"]

    first = await client.post("/api/translations", json={"review_ids": [portuguese["review_id"]]})
    second = await client.post("/api/translations", json={"review_ids": [portuguese["review_id"]]})
    browsed = await get_json(client, "/api/apps/duolingo/reviews", language="pt")
    searched = await get_json(
        client, "/api/apps/duolingo/search", q="muitos anúncios", mode="lexical"
    )

    assert first.json()["items"] == [
        {
            "review_id": portuguese["review_id"],
            "outcome": "translated",
            "source_language": "pt",
            "text": "[pt→en] muitos anúncios",
        }
    ]
    assert second.json()["items"][0]["outcome"] == "cached"
    assert len(translator.calls) == 1
    assert browsed["items"][0]["translation"] == "[pt→en] muitos anúncios"
    assert searched["results"][0]["translation"] == "[pt→en] muitos anúncios"


async def test_rejects_empty_translation_requests(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/translations", json={"review_ids": []})

    assert response.status_code == 422


async def test_topics_are_empty_before_any_discovery(client: httpx.AsyncClient) -> None:
    assert await get_json(client, "/api/apps/duolingo/topics") == {"run": None, "topics": []}


async def test_discovers_topics_and_browses_their_reviews(client: httpx.AsyncClient) -> None:
    created = await client.post(
        "/api/apps/duolingo/topics/runs",
        json={"topic_count": 2, "all_ratings": True},
    )
    assert created.status_code == 201, created.text
    run = created.json()

    body = await get_json(client, "/api/apps/duolingo/topics", samples=1)
    first = body["topics"][0]
    page = await get_json(client, f"/api/apps/duolingo/topics/{first['id']}/reviews")

    assert (run["review_count"], run["topic_count"]) == (4, 2)
    assert body["run"]["id"] == run["run_id"]
    assert body["run"]["parameters"]["max_rating"] is None
    assert sum(topic["size"] for topic in body["topics"]) == 4
    assert all(len(topic["samples"]) == 1 for topic in body["topics"])
    assert {len(topic["daily"]) for topic in body["topics"]} == {1}
    assert (page["total"], len(page["items"])) == (first["size"], first["size"])
    assert page["items"][0]["review_id"] == first["samples"][0]["review_id"]
    assert page["topic"]["id"] == first["id"]


async def test_unknown_topic_is_not_found(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/apps/duolingo/topics/999/reviews")

    assert response.status_code == 404


async def test_discovery_reports_too_few_reviews_as_a_conflict(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/apps/duolingo/topics/runs", json={"max_rating": 1})

    assert response.status_code == 409
    assert "at least 2" in response.json()["detail"]


async def test_evaluation_scores_every_mode(client: httpx.AsyncClient) -> None:
    body = await get_json(client, "/api/evaluation")

    assert [mode["mode"] for mode in body["modes"]] == ["lexical", "semantic", "hybrid"]
    assert body["judgments"] == 2


async def test_anomalies_report_coverage_after_topic_discovery(client: httpx.AsyncClient) -> None:
    before = await get_json(client, "/api/apps/duolingo/anomalies")
    created = await client.post(
        "/api/apps/duolingo/topics/runs", json={"topic_count": 2, "all_ratings": True}
    )
    assert created.status_code == 201, created.text

    after = await get_json(client, "/api/apps/duolingo/anomalies")

    assert before == {"run": None, "tests": 0, "threshold": 0.0, "coverage": [], "spikes": []}
    assert after["run"]["id"] == created.json()["run_id"]
    assert after["spikes"] == []
    assert {entry["source"] for entry in after["coverage"]} == {
        "google_play:en",
        "google_play:pt",
        "app_store:us",
    }
    assert not any(entry["tested"] for entry in after["coverage"])
