from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.anomalies.service import AnomalyService
from reviewradar.catalog import Catalog
from reviewradar.config import AnomalySettings
from reviewradar.ingestion.repository import CatalogRepository
from tests.integration.test_anomaly_service import APP, DAYS, seed_run
from tests.integration.test_web_replies import panel_client, panel_services

SessionFactory = async_sessionmaker[AsyncSession]
CATALOG = Catalog(apps=(APP,))


@pytest.fixture
async def client(session_factory: SessionFactory) -> AsyncIterator[httpx.AsyncClient]:
    # The seeded surge has three earlier days of coverage; production requires five.
    anomalies = AnomalyService(session_factory, AnomalySettings(min_baseline_days=3))
    services = panel_services(session_factory, catalog=CATALOG, anomalies=anomalies)
    async with panel_client(services) as client:
        yield client


async def test_backtest_reports_incidents_and_known_spikes(
    session_factory: SessionFactory, client: httpx.AsyncClient
) -> None:
    await seed_run(session_factory)

    response = await client.get("/api/apps/duolingo/anomalies/monitor", params={"samples": 2})

    assert response.status_code == 200
    body = response.json()
    assert len(body["days"]) == DAYS - 2
    assert body["alert_days"] == 1
    (incident,) = body["incidents"]
    assert incident["topic"]["label"] == "icon"
    assert incident["release"] == "7.138.0"
    assert incident["baseline_days"] == 3
    assert len(incident["samples"]) == 2
    (spike,) = body["spikes"]
    assert spike["caught"] is True
    assert spike["delay_days"] == 0
    assert spike["first_alert_day"] == incident["first_day"]


async def test_backtest_before_topic_discovery_is_empty(
    session_factory: SessionFactory, client: httpx.AsyncClient
) -> None:
    async with session_factory.begin() as session:
        await CatalogRepository(session).sync_app(APP)

    body = (await client.get("/api/apps/duolingo/anomalies/monitor")).json()

    assert body["run"] is None
    assert body["incidents"] == []
    assert body["min_baseline_days"] == 3
