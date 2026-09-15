"""Finds spikes in the latest topic run of an app, and backtests the daily monitor on it.

Both are computed when asked rather than stored: they read the run's topic assignments and
run in well under a second, and they always reflect the latest topic run.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.anomalies import monitor
from reviewradar.anomalies.detector import Detection, Observation, detect_spikes
from reviewradar.anomalies.repository import AnomalyRepository
from reviewradar.config import AnomalySettings
from reviewradar.topics.repository import TopicRepository, TopicRunView, TopicView


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    run: TopicRunView
    topics: dict[int, TopicView]
    detection: Detection


@dataclass(frozen=True, slots=True)
class BacktestReport:
    run: TopicRunView
    topics: dict[int, TopicView]
    backtest: monitor.Backtest
    outcomes: tuple[monitor.SpikeOutcome, ...]
    """Each retrospective spike, and whether the monitor would have alerted on it."""


class AnomalyService:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], settings: AnomalySettings
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings

    @property
    def settings(self) -> AnomalySettings:
        return self._settings

    async def detect(self, app_slug: str) -> AnomalyReport | None:
        """Spikes in the app's latest topic run, or None before any topic discovery."""
        loaded = await self._load(app_slug)
        if loaded is None:
            return None
        run, topics, observations = loaded
        return AnomalyReport(run, topics, detect_spikes(observations, self._settings))

    async def backtest(self, app_slug: str) -> BacktestReport | None:
        """Replay the daily monitor over the latest topic run, or None before discovery."""
        loaded = await self._load(app_slug)
        if loaded is None:
            return None
        run, topics, observations = loaded
        result = monitor.backtest(observations, self._settings)
        spikes = detect_spikes(observations, self._settings).spikes
        return BacktestReport(run, topics, result, monitor.compare_with_spikes(spikes, result))

    async def _load(
        self, app_slug: str
    ) -> tuple[TopicRunView, dict[int, TopicView], list[Observation]] | None:
        async with self._session_factory() as session:
            topics = TopicRepository(session)
            run = await topics.latest_run(app_slug)
            if run is None:
                return None
            views = await topics.topics(run.id)
            observations = await AnomalyRepository(session).observations(
                run.id, max_rating=self._settings.max_rating
            )
        return run, {view.id: view for view in views}, observations
