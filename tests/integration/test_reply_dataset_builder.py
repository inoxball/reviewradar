import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import AppSpec, Catalog, MarketSpec
from reviewradar.config import EnrichmentSettings, ReplySettings
from reviewradar.domain import Store
from reviewradar.enrichment.service import EnrichmentService
from reviewradar.ingestion.repository import CatalogRepository, ReviewRepository
from reviewradar.replies.dataset import ReplyDatasetBuilder, ReplyDatasetRepository
from tests.fakes import FakeEmbedder, FixedLanguageDetector
from tests.support import NOW, duolingo, make_review

SessionFactory = async_sessionmaker[AsyncSession]

DUOLINGO = duolingo(Store.GOOGLE_PLAY)
BABBEL = AppSpec(
    slug="babbel",
    name="Babbel",
    listings={Store.GOOGLE_PLAY: "com.babbel.mobile.android.en"},
    markets=(MarketSpec(country="us", language="en"),),
)
EMBEDDER = FakeEmbedder(dimension=64)


async def seed(session_factory: SessionFactory) -> None:
    async with session_factory.begin() as session:
        catalog = CatalogRepository(session)
        reviews = ReviewRepository(session)
        babbel_listings = await catalog.sync_app(BABBEL)
        await reviews.upsert_batch(
            babbel_listings[Store.GOOGLE_PLAY],
            [
                replace(
                    make_review("b-1", body="The speech recognition is broken", rating=2),
                    developer_reply=(
                        "Hi Charles, sorry about the speech recognition issues. "
                        "Could you contact us at support@babbel.com?"
                    ),
                ),
                replace(make_review("b-2", body="Nice app"), developer_reply="🧡"),
                make_review("b-3", body="No reply to this one"),
            ],
            seen_at=NOW,
        )
        duolingo_listings = await catalog.sync_app(DUOLINGO)
        await reviews.upsert_batch(
            duolingo_listings[Store.GOOGLE_PLAY],
            [make_review("d-1", body="Energy runs out after one lesson", rating=1)],
            seen_at=NOW,
        )
    for app in (BABBEL, DUOLINGO):
        await EnrichmentService(
            session_factory,
            FixedLanguageDetector("en", confidence=0.99),
            EMBEDDER,
            EnrichmentSettings(min_language_chars=1),
            clock=lambda: NOW,
        ).enrich_app(app)


@pytest.fixture
async def seeded(session_factory: SessionFactory) -> SessionFactory:
    await seed(session_factory)
    return session_factory


def settings_for(tmp_path: Path) -> ReplySettings:
    written = tmp_path / "written"
    written.mkdir()
    (written / "duolingo-01.jsonl").write_text(
        json.dumps(
            {
                "app": "duolingo",
                "store": "google_play",
                "id": "d-1",
                "reply": "Sorry the energy runs out so quickly; your feedback is shared.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ReplySettings(
        written_dir=written,
        dataset_dir=tmp_path / "dataset",
        source_apps=["babbel"],
        eval_fraction_published=0.0,
        eval_fraction_written=0.0,
    )


async def test_builds_clean_training_files_from_published_and_written_replies(
    seeded: SessionFactory, tmp_path: Path
) -> None:
    settings = settings_for(tmp_path)

    report = await ReplyDatasetBuilder(
        seeded, Catalog(apps=(DUOLINGO, BABBEL)), settings, detect_language=lambda _: "en"
    ).build()

    records = [
        json.loads(line)
        for line in (settings.dataset_dir / "train.jsonl").read_text("utf-8").splitlines()
    ]
    assert (settings.dataset_dir / "eval.jsonl").read_text("utf-8") == ""
    assert [(record["key"], record["source"]) for record in records] == [
        ("babbel:google_play:b-1", "published"),
        ("duolingo:google_play:d-1", "written"),
    ]
    assert records[0]["reply"] == (
        "Hi, sorry about the speech recognition issues. Could you contact us at {support_contact}?"
    )
    assert records[0]["review"] == "The speech recognition is broken"
    assert report.dropped == {"truncated or too short": 1}


async def test_looks_up_current_embeddings_by_review_key(seeded: SessionFactory) -> None:
    async with seeded() as session:
        found = await ReplyDatasetRepository(session).embeddings(
            [("babbel", Store.GOOGLE_PLAY, "b-1"), ("duolingo", Store.GOOGLE_PLAY, "missing")],
            model=EMBEDDER.model_name,
        )

    assert list(found) == [("babbel", Store.GOOGLE_PLAY, "b-1")]
    assert found[("babbel", Store.GOOGLE_PLAY, "b-1")].shape == (EMBEDDER.dimension,)
