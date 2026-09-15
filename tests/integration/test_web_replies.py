import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import Catalog
from reviewradar.config import EnrichmentSettings, SearchSettings, TranslationSettings
from reviewradar.domain import Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.replies.drafts import ReplyDraftService, ReplyGeneratorSet
from reviewradar.replies.generation import ReplyTask
from reviewradar.search.service import SearchService
from reviewradar.translation.service import TranslationService
from reviewradar.web.app import PanelServices, create_app
from tests.fakes import FakeEmbedder, FakeTranslator, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

APP = duolingo(Store.GOOGLE_PLAY).model_copy(update={"support_contact": "support.duolingo.com"})
CATALOG = Catalog(apps=(APP,))
EMBEDDER = FakeEmbedder(dimension=32)
BENCHMARK = {
    "created_at": "2026-09-14T20:00:00+00:00",
    "base_model": "Qwen/Qwen3-1.7B",
    "source": "written",
    "examples": 44,
    "generators": [
        {
            "name": "reference",
            "replies": 44,
            "pass_rate": 1.0,
            "check_rates": {"non_empty": 1.0},
            "opening_diversity": 1.0,
            "median_chars": 190.0,
            "seconds_per_reply": 0.0,
        }
    ],
}


class TemplateGenerator:
    name = "fine-tuned"

    def generate(self, tasks: Sequence[ReplyTask]) -> list[str]:
        return [
            "Sorry about this, please contact {support_contact} with your device." for _ in tasks
        ]


def panel_services(session_factory: SessionFactory, **overrides: object) -> PanelServices:
    async def load_generators() -> ReplyGeneratorSet:
        return ReplyGeneratorSet({"fine-tuned": TemplateGenerator()}, {"fine-tuned": "v1"})

    defaults: dict[str, object] = {
        "catalog": CATALOG,
        "session_factory": session_factory,
        "search": SearchService(session_factory, EMBEDDER, SearchSettings()),
        "translation": TranslationService(
            session_factory, FakeTranslator(), TranslationSettings(), clock=lambda: NOW
        ),
        "replies": ReplyDraftService(
            session_factory,
            CATALOG,
            load_generators,
            available=["fine-tuned"],
            embedding_model=EMBEDDER.model_name,
            detect_language=lambda _: "en",
            clock=lambda: NOW,
        ),
    }
    return PanelServices(**(defaults | overrides))  # type: ignore[arg-type]


@asynccontextmanager
async def panel_client(services: PanelServices) -> AsyncIterator[httpx.AsyncClient]:
    @asynccontextmanager
    async def provide() -> AsyncIterator[PanelServices]:
        yield services

    app = create_app(provide)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://panel"
        ) as client,
    ):
        yield client


@pytest.fixture
async def client(
    session_factory: SessionFactory, tmp_path: Path
) -> AsyncIterator[httpx.AsyncClient]:
    async with session_factory.begin() as session:
        listing_ids = await CatalogRepository(session).sync_app(APP)
        await ReviewRepository(session).upsert_batch(
            listing_ids[Store.GOOGLE_PLAY],
            [make_review("crash", body="The app crashes on every lesson", rating=1)],
            seen_at=NOW,
        )
    await EnrichmentService(
        session_factory,
        FixedLanguageDetector("en", confidence=0.99),
        EMBEDDER,
        EnrichmentSettings(min_language_chars=1),
        clock=lambda: NOW,
    ).enrich_app(APP)

    benchmark_path = tmp_path / "summary.json"
    benchmark_path.write_text(json.dumps(BENCHMARK), encoding="utf-8")
    services = panel_services(session_factory, replies_benchmark_path=benchmark_path)
    async with panel_client(services) as client:
        yield client


async def review_id(client: httpx.AsyncClient) -> int:
    page = (await client.get("/api/apps/duolingo/reviews")).json()
    return int(page["items"][0]["review_id"])


async def test_lists_generators_with_the_latest_benchmark(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/replies/generators")).json()

    assert body["generators"] == ["fine-tuned"]
    assert body["loaded"] is False
    assert body["benchmark"]["examples"] == 44
    assert [row["name"] for row in body["benchmark"]["generators"]] == ["reference"]


async def test_drafts_replies_and_reuses_stored_drafts(client: httpx.AsyncClient) -> None:
    payload = {"review_ids": [await review_id(client)], "generator": "fine-tuned"}

    first = await client.post("/api/replies/drafts", json=payload)
    second = await client.post("/api/replies/drafts", json=payload)
    generators = (await client.get("/api/replies/generators")).json()

    assert first.status_code == 200, first.text
    (drafted,) = first.json()["items"]
    assert drafted["status"] == "drafted"
    assert (
        drafted["text"] == "Sorry about this, please contact support.duolingo.com with your device."
    )
    assert drafted["template"].endswith("{support_contact} with your device.")
    assert drafted["checks"]["passed"] is True
    assert second.json()["items"][0]["status"] == "cached"
    assert generators["loaded"] is True


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"review_ids": [1], "generator": "gpt"}, "not available"),
        ({"review_ids": list(range(11)), "generator": "fine-tuned"}, "at most 10"),
        ({"review_ids": [], "generator": "fine-tuned"}, "at least 1"),
    ],
    ids=["unknown-generator", "too-many-reviews", "no-reviews"],
)
async def test_rejects_invalid_draft_requests(
    client: httpx.AsyncClient, payload: dict[str, object], detail: str
) -> None:
    response = await client.post("/api/replies/drafts", json=payload)

    assert response.status_code == 422
    assert detail in response.text


async def test_reply_endpoints_report_when_drafting_is_not_configured(
    session_factory: SessionFactory,
) -> None:
    async with panel_client(panel_services(session_factory, replies=None)) as client:
        response = await client.get("/api/replies/generators")

    assert response.status_code == 503
