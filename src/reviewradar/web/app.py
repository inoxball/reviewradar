"""The demo panel: a FastAPI app serving a static single-page UI and a small JSON API.

Dependencies arrive through a :class:`PanelServices` factory, so production builds them
from settings (loading models once) while tests inject fakes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar import __version__
from reviewradar.anomalies.monitor import Incident
from reviewradar.anomalies.service import AnomalyService
from reviewradar.catalog import AppSpec, Catalog, UnknownAppError
from reviewradar.domain import Store
from reviewradar.enrichment.repository import EnrichmentRepository
from reviewradar.ingestion.repository import ReviewRepository
from reviewradar.replies.drafts import ReplyDraftService, UnknownGeneratorError
from reviewradar.search.evaluation import JudgmentSet, ModeReport, evaluate_modes
from reviewradar.search.retrievers import UnsupportedDatabaseError
from reviewradar.search.service import SearchService
from reviewradar.search.types import SearchFilters, SearchMode
from reviewradar.topics.repository import TopicRepository, TopicView
from reviewradar.topics.service import InsufficientReviewsError, TopicScope, TopicService
from reviewradar.translation.repository import TranslationRepository
from reviewradar.translation.service import TranslationService
from reviewradar.web import schemas
from reviewradar.web.queries import PanelQueries, ReviewSort, ReviewView

STATIC_DIR = Path(__file__).resolve().parent / "static"
EVALUATION_K = 10


class EvaluationCache:
    """Scores every search mode against the judgments once per process; it takes seconds."""

    def __init__(self, judgments_path: Path | None) -> None:
        self._judgments_path = judgments_path
        self._lock = asyncio.Lock()
        self._response: schemas.EvaluationResponse | None = None

    async def get(self, search: SearchService) -> schemas.EvaluationResponse | None:
        if self._judgments_path is None or not self._judgments_path.exists():
            return None
        async with self._lock:
            if self._response is None:
                judgment_set = JudgmentSet.from_yaml(self._judgments_path)
                reports = await evaluate_modes(
                    search, judgment_set, modes=list(SearchMode), k=EVALUATION_K
                )
                self._response = _evaluation_response(judgment_set, reports)
        return self._response


@dataclass
class PanelServices:
    """Everything the panel needs: built from settings in production, injected in tests."""

    catalog: Catalog
    session_factory: async_sessionmaker[AsyncSession]
    search: SearchService
    translation: TranslationService
    judgments_path: Path | None = None
    topics: TopicService | None = None
    anomalies: AnomalyService | None = None
    replies: ReplyDraftService | None = None
    replies_benchmark_path: Path | None = None
    translation_model_loaded: Callable[[], bool] = lambda: True
    evaluation: EvaluationCache = field(init=False)

    def __post_init__(self) -> None:
        self.evaluation = EvaluationCache(self.judgments_path)


ServicesFactory = Callable[[], AbstractAsyncContextManager[PanelServices]]


def create_app(services_factory: ServicesFactory) -> FastAPI:
    """Build the panel; services live for the lifetime of the application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with services_factory() as services:
            app.state.services = services
            yield

    app = FastAPI(
        title="ReviewRadar API",
        version=__version__,
        description="Review intelligence over app store reviews: corpus, pipeline, search, "
        "translation and evaluation.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(_router)
    return app


def _get_services(request: Request) -> PanelServices:
    return cast(PanelServices, request.app.state.services)


Services = Annotated[PanelServices, Depends(_get_services)]
QueryText = Annotated[str, Query(min_length=1, max_length=500, description="Any language.")]
Limit = Annotated[int, Query(ge=1, le=100)]
StoreFilter = Annotated[list[Store] | None, Query(description="Repeat to allow several.")]
LanguageFilter = Annotated[list[str] | None, Query(description="ISO 639-1; repeatable.")]
RatingBound = Annotated[int | None, Query(ge=1, le=5)]

_router = APIRouter()


@_router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@_router.get("/api/system", tags=["system"], summary="Versions, database and models")
async def system(services: Services) -> schemas.SystemInfo:
    async with services.session_factory() as session:
        database = await PanelQueries(session).database_info()
    return schemas.SystemInfo(
        version=__version__,
        database=schemas.DatabaseInfo.model_validate(database),
        embedding_model=services.search.model_name,
        translation=schemas.TranslationInfo(
            model=services.translation.model_name,
            target_language=services.translation.target_language,
            loaded=services.translation_model_loaded(),
        ),
    )


@_router.get("/api/apps", tags=["apps"], summary="Apps in the catalog")
async def list_apps(services: Services) -> list[schemas.AppInfo]:
    return [schemas.AppInfo(slug=app.slug, name=app.name) for app in services.catalog.apps]


@_router.get("/api/apps/{slug}/overview", tags=["apps"], summary="Corpus and pipeline overview")
async def overview(slug: str, services: Services) -> schemas.Overview:
    """Review counts per market, rating distribution, language mix and pipeline coverage."""
    app = _require_app(services, slug)
    async with services.session_factory() as session:
        markets = await ReviewRepository(session).summarize_app(slug)
        language_mix = await EnrichmentRepository(session).language_mix(slug)
        queries = PanelQueries(session)
        ratings = await queries.rating_distribution(slug)
        pipeline = await queries.pipeline_counts(
            slug,
            embedding_model=services.search.model_name,
            translation_model=services.translation.model_name,
            target_language=services.translation.target_language,
        )

    total = sum(market.review_count for market in markets)
    weighted_rating = sum(market.average_rating * market.review_count for market in markets)
    return schemas.Overview(
        app=schemas.AppInfo(slug=app.slug, name=app.name),
        total_reviews=total,
        average_rating=weighted_rating / total if total else None,
        languages=len({share.language for share in language_mix if share.language}),
        markets=[schemas.MarketBreakdown.model_validate(market) for market in markets],
        language_mix=[schemas.LanguageCount.model_validate(share) for share in language_mix],
        ratings=[schemas.RatingCount.model_validate(rating) for rating in ratings],
        pipeline=schemas.PipelineCounts.model_validate(pipeline),
    )


@_router.get("/api/apps/{slug}/ingestion", tags=["pipeline"], summary="Ingestion runs and cursors")
async def ingestion_status(
    slug: str,
    services: Services,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> schemas.IngestionStatus:
    """The most recent ingestion runs and the incremental cursor of every partition."""
    _require_app(services, slug)
    async with services.session_factory() as session:
        queries = PanelQueries(session)
        runs = await queries.recent_runs(slug, limit=limit)
        cursors = await queries.cursors(slug)
    return schemas.IngestionStatus(
        runs=[schemas.IngestionRunItem.model_validate(run) for run in runs],
        cursors=[schemas.CursorItem.model_validate(cursor) for cursor in cursors],
    )


@_router.get("/api/apps/{slug}/reviews", tags=["reviews"], summary="Browse reviews")
async def browse_reviews(
    slug: str,
    services: Services,
    sort: ReviewSort = ReviewSort.NEWEST,
    limit: Limit = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    store: StoreFilter = None,
    language: LanguageFilter = None,
    min_rating: RatingBound = None,
    max_rating: RatingBound = None,
) -> schemas.ReviewPage:
    """A sorted, filtered page of reviews with stored translations."""
    _require_app(services, slug)
    filters = _filters(store, language, min_rating, max_rating)
    async with services.session_factory() as session:
        total, reviews = await PanelQueries(session).browse_reviews(
            slug, filters=filters, sort=sort, limit=limit, offset=offset
        )
    return schemas.ReviewPage(
        total=total,
        limit=limit,
        offset=offset,
        items=await _with_translations(services, reviews),
    )


@_router.get("/api/apps/{slug}/search", tags=["search"], summary="Search reviews")
async def search(
    slug: str,
    services: Services,
    q: QueryText,
    mode: SearchMode = SearchMode.HYBRID,
    limit: Limit = 10,
    store: StoreFilter = None,
    language: LanguageFilter = None,
    min_rating: RatingBound = None,
    max_rating: RatingBound = None,
) -> schemas.SearchResponse:
    """Keyword, semantic or hybrid search over one app's reviews."""
    _require_app(services, slug)
    filters = _filters(store, language, min_rating, max_rating)
    return await _timed_search(services, slug, q, mode, filters, limit)


@_router.get("/api/apps/{slug}/compare", tags=["search"], summary="Compare search modes")
async def compare(
    slug: str,
    services: Services,
    q: QueryText,
    limit: Limit = 10,
    store: StoreFilter = None,
    language: LanguageFilter = None,
    min_rating: RatingBound = None,
    max_rating: RatingBound = None,
) -> schemas.CompareResponse:
    """The same query in every mode, one after another so latencies are comparable."""
    _require_app(services, slug)
    filters = _filters(store, language, min_rating, max_rating)
    responses = [
        await _timed_search(services, slug, q, mode, filters, limit) for mode in SearchMode
    ]
    return schemas.CompareResponse(query=q, modes=responses)


@_router.get("/api/apps/{slug}/topics", tags=["topics"], summary="Topics of the latest run")
async def list_topics(
    slug: str,
    services: Services,
    samples: Annotated[
        int, Query(ge=0, le=10, description="Representative reviews per topic.")
    ] = 3,
) -> schemas.TopicsResponse:
    """Every topic of the most recent discovery run with its keywords, rating profile,
    daily volume and the reviews closest to its centre."""
    _require_app(services, slug)
    async with services.session_factory() as session:
        repository = TopicRepository(session)
        run = await repository.latest_run(slug)
        if run is None:
            return schemas.TopicsResponse(run=None, topics=[])
        topic_views = await repository.topics(run.id)
        daily = await repository.daily_counts(run.id)
        representatives = (
            await repository.representatives(run.id, per_topic=samples) if samples else {}
        )
        reviews = await PanelQueries(session).reviews_by_ids(
            slug, [review_id for ids in representatives.values() for review_id in ids]
        )

    items = {item.review_id: item for item in await _with_translations(services, reviews.values())}
    samples_by_topic = {
        topic_id: [items[review_id] for review_id in ids if review_id in items]
        for topic_id, ids in representatives.items()
    }
    days = _day_range(daily)
    return schemas.TopicsResponse(
        run=schemas.TopicRunInfo.model_validate(run),
        topics=[
            _topic_item(topic).model_copy(
                update={
                    "daily": [
                        schemas.DailyCount(day=day, count=daily.get(topic.id, {}).get(day, 0))
                        for day in days
                    ],
                    "samples": samples_by_topic.get(topic.id, []),
                }
            )
            for topic in topic_views
        ],
    )


@_router.get(
    "/api/apps/{slug}/anomalies/monitor", tags=["topics"], summary="Daily monitor backtest"
)
async def monitor_backtest(
    slug: str,
    services: Services,
    samples: Annotated[int, Query(ge=0, le=10, description="Example reviews per incident.")] = 3,
) -> schemas.MonitorBacktestResponse:
    """Replay a daily alert job over the latest topic run, judging each day on earlier days only.

    Consecutive alert days form incidents, and incidents concentrated on a newly seen app
    version are flagged as possible release regressions. Each spike found in hindsight is
    reported as caught or missed, with the reason.
    """
    _require_app(services, slug)
    if services.anomalies is None:
        raise HTTPException(status_code=503, detail="anomaly detection is not configured")
    settings = services.anomalies.settings
    report = await services.anomalies.backtest(slug)
    if report is None:
        return schemas.MonitorBacktestResponse(
            run=None,
            baseline_days=settings.baseline_days,
            min_baseline_days=settings.min_baseline_days,
            daily_alpha=settings.daily_alpha,
            days=[],
            judged_days=0,
            alert_days=0,
            tests=0,
            incidents=[],
            spikes=[],
        )

    result = report.backtest
    wanted = [
        review_id for incident in result.incidents for review_id in incident.review_ids[:samples]
    ]
    async with services.session_factory() as session:
        reviews = await PanelQueries(session).reviews_by_ids(slug, wanted)
    items = {item.review_id: item for item in await _with_translations(services, reviews.values())}
    return schemas.MonitorBacktestResponse(
        run=schemas.TopicRunInfo.model_validate(report.run),
        baseline_days=settings.baseline_days,
        min_baseline_days=settings.min_baseline_days,
        daily_alpha=settings.daily_alpha,
        days=[
            schemas.MonitorDay(
                day=day.day, sources=day.sources, tests=day.tests, alerts=len(day.alerts)
            )
            for day in result.days
        ],
        judged_days=result.judged_days,
        alert_days=result.alert_days,
        tests=result.tests,
        incidents=[
            _incident_item(incident, report.topics, items, samples) for incident in result.incidents
        ],
        spikes=[
            schemas.SpikeOutcomeItem(
                topic=_topic_item(report.topics[outcome.spike.topic_id]),
                source=outcome.spike.source,
                day=outcome.spike.day,
                caught=outcome.incident is not None,
                first_alert_day=outcome.incident.first_day if outcome.incident else None,
                delay_days=outcome.delay_days,
                reason=outcome.reason,
            )
            for outcome in report.outcomes
        ],
    )


@_router.get("/api/apps/{slug}/anomalies", tags=["topics"], summary="Topic spikes")
async def topic_anomalies(
    slug: str,
    services: Services,
    samples: Annotated[int, Query(ge=0, le=10, description="Example reviews per spike.")] = 3,
) -> schemas.AnomalyResponse:
    """Days when one topic suddenly dominated a source's critical reviews, in the latest run.

    Shares are tested per store and market, so changes in how many reviews a store exposes
    are not mistaken for incidents.
    """
    _require_app(services, slug)
    if services.anomalies is None:
        raise HTTPException(status_code=503, detail="anomaly detection is not configured")
    report = await services.anomalies.detect(slug)
    if report is None:
        return schemas.AnomalyResponse(run=None, tests=0, threshold=0.0, coverage=[], spikes=[])

    detection = report.detection
    wanted = [review_id for spike in detection.spikes for review_id in spike.review_ids[:samples]]
    async with services.session_factory() as session:
        reviews = await PanelQueries(session).reviews_by_ids(slug, wanted)
    items = {item.review_id: item for item in await _with_translations(services, reviews.values())}
    return schemas.AnomalyResponse(
        run=schemas.TopicRunInfo.model_validate(report.run),
        tests=detection.tests,
        threshold=detection.threshold,
        coverage=[schemas.SourceCoverageItem.model_validate(entry) for entry in detection.coverage],
        spikes=[
            schemas.SpikeItem(
                topic=_topic_item(report.topics[spike.topic_id]),
                source=spike.source,
                day=spike.day,
                count=spike.count,
                total=spike.total,
                share=spike.share,
                expected_share=spike.expected_share,
                lift=spike.lift,
                p_value=spike.p_value,
                top_version=spike.top_version,
                top_version_share=spike.top_version_share,
                samples=[
                    items[review_id]
                    for review_id in spike.review_ids[:samples]
                    if review_id in items
                ],
            )
            for spike in detection.spikes
        ],
    )


@_router.get(
    "/api/apps/{slug}/topics/{topic_id}/reviews", tags=["topics"], summary="Reviews in a topic"
)
async def topic_reviews(
    slug: str,
    topic_id: int,
    services: Services,
    limit: Limit = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> schemas.TopicReviewPage:
    """A page of a topic's reviews, closest to the topic's centre first."""
    _require_app(services, slug)
    async with services.session_factory() as session:
        repository = TopicRepository(session)
        topic = await repository.topic_in_app(topic_id, slug)
        if topic is None:
            raise HTTPException(status_code=404, detail=f"no topic {topic_id} for '{slug}'")
        total, review_ids = await repository.topic_review_ids(topic_id, limit=limit, offset=offset)
        reviews = await PanelQueries(session).reviews_by_ids(slug, review_ids)
    ordered = [reviews[review_id] for review_id in review_ids if review_id in reviews]
    return schemas.TopicReviewPage(
        topic=_topic_item(topic),
        total=total,
        limit=limit,
        offset=offset,
        items=await _with_translations(services, ordered),
    )


@_router.post(
    "/api/apps/{slug}/topics/runs", tags=["topics"], summary="Discover topics", status_code=201
)
async def discover_topics(
    slug: str, request: schemas.TopicRunRequest, services: Services
) -> schemas.TopicRunResponse:
    """Cluster the app's current review embeddings into topics and store them as the latest run.

    Unset fields use the configured defaults. Takes a few seconds for thousands of reviews.
    """
    app = _require_app(services, slug)
    if services.topics is None:
        raise HTTPException(status_code=503, detail="topic discovery is not configured")
    defaults = services.topics.default_scope
    scope = TopicScope(
        max_rating=None if request.all_ratings else (request.max_rating or defaults.max_rating),
        min_words=request.min_words or defaults.min_words,
    )
    started = time.perf_counter()
    try:
        report = await services.topics.discover(app, scope=scope, topic_count=request.topic_count)
    except InsufficientReviewsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return schemas.TopicRunResponse(
        run_id=report.run_id,
        review_count=report.review_count,
        topic_count=len(report.topics),
        took_ms=_elapsed_ms(started),
    )


@_router.get("/api/replies/generators", tags=["replies"], summary="Reply generators")
async def reply_generators(services: Services) -> schemas.ReplyGeneratorsResponse:
    """Available reply generators, whether the model is loaded, and the latest benchmark."""
    replies = _require_replies(services)
    path = services.replies_benchmark_path
    benchmark = (
        schemas.ReplyBenchmark.model_validate_json(path.read_text(encoding="utf-8"))
        if path is not None and path.exists()
        else None
    )
    return schemas.ReplyGeneratorsResponse(
        generators=list(replies.available_generators),
        loaded=replies.loaded,
        benchmark=benchmark,
    )


@_router.post("/api/replies/drafts", tags=["replies"], summary="Draft replies")
async def draft_replies(
    request: schemas.ReplyDraftRequest, services: Services
) -> schemas.ReplyDraftResponse:
    """Draft developer replies to reviews with a local model and store them.

    Stored drafts from the current generator version are returned without running the
    model. The first request loads the model, which takes several seconds.
    """
    replies = _require_replies(services)
    started = time.perf_counter()
    try:
        results = await replies.draft(request.review_ids, request.generator)
    except UnknownGeneratorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return schemas.ReplyDraftResponse(
        generator=request.generator,
        took_ms=_elapsed_ms(started),
        items=[schemas.ReplyDraftItem.model_validate(result) for result in results],
    )


@_router.post("/api/translations", tags=["translation"], summary="Translate reviews")
async def translate(
    request: schemas.TranslationRequest, services: Services
) -> schemas.TranslationResponse:
    """Translate reviews into the target language with the local model, storing the results.

    Stored translations are returned without touching the model. The first call loads the
    model, which takes several seconds.
    """
    started = time.perf_counter()
    results = await services.translation.translate_reviews(request.review_ids)
    return schemas.TranslationResponse(
        model=services.translation.model_name,
        target_language=services.translation.target_language,
        took_ms=_elapsed_ms(started),
        items=[schemas.TranslationItem.model_validate(result) for result in results],
    )


@_router.get("/api/evaluation", tags=["search"], summary="Search evaluation")
async def evaluation(services: Services) -> schemas.EvaluationResponse:
    """Offline metrics of every search mode against the configured relevance judgments."""
    response = await services.evaluation.get(services.search)
    if response is None:
        raise HTTPException(status_code=404, detail="no relevance judgments are configured")
    return response


def _require_replies(services: PanelServices) -> ReplyDraftService:
    if services.replies is None:
        raise HTTPException(status_code=503, detail="reply drafting is not configured")
    return services.replies


def _require_app(services: PanelServices, slug: str) -> AppSpec:
    try:
        return services.catalog.get(slug)
    except UnknownAppError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _filters(
    stores: list[Store] | None,
    languages: list[str] | None,
    min_rating: int | None,
    max_rating: int | None,
) -> SearchFilters:
    return SearchFilters(
        stores=frozenset(stores or ()),
        languages=frozenset(languages or ()),
        min_rating=min_rating,
        max_rating=max_rating,
    )


async def _timed_search(
    services: PanelServices,
    slug: str,
    query: str,
    mode: SearchMode,
    filters: SearchFilters,
    limit: int,
) -> schemas.SearchResponse:
    started = time.perf_counter()
    try:
        results = await services.search.search(slug, query, mode=mode, filters=filters, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnsupportedDatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    took_ms = _elapsed_ms(started)

    translations = await _stored_translations(services, [result.review_id for result in results])
    return schemas.SearchResponse(
        query=query,
        mode=mode,
        took_ms=took_ms,
        results=[
            schemas.ReviewHit.model_validate(result).model_copy(
                update={"translation": translations.get(result.review_id)}
            )
            for result in results
        ],
    )


async def _stored_translations(
    services: PanelServices, review_ids: Sequence[int]
) -> dict[int, str]:
    if not review_ids:
        return {}
    async with services.session_factory() as session:
        stored = await TranslationRepository(session).fresh(
            review_ids,
            model=services.translation.model_name,
            target_language=services.translation.target_language,
        )
    return {review_id: translation.text for review_id, translation in stored.items()}


async def _with_translations(
    services: PanelServices, reviews: Iterable[ReviewView]
) -> list[schemas.ReviewItem]:
    reviews = list(reviews)
    translations = await _stored_translations(services, [review.review_id for review in reviews])
    return [
        schemas.ReviewItem.model_validate(review).model_copy(
            update={"translation": translations.get(review.review_id)}
        )
        for review in reviews
    ]


def _incident_item(
    incident: Incident,
    topics: dict[int, TopicView],
    reviews: dict[int, schemas.ReviewItem],
    samples: int,
) -> schemas.IncidentItem:
    peak = incident.peak
    return schemas.IncidentItem(
        topic=_topic_item(topics[incident.topic_id]),
        source=incident.source,
        first_day=incident.first_day,
        last_day=incident.last_day,
        alert_days=len(incident.alerts),
        review_count=incident.review_count,
        peak_day=peak.day,
        peak_count=peak.count,
        peak_total=peak.total,
        peak_share=peak.share,
        expected_share=peak.expected_share,
        lift=peak.lift,
        p_value=peak.p_value,
        baseline_days=peak.baseline_days,
        top_version=incident.top_version,
        top_version_share=incident.top_version_share,
        release=incident.release,
        samples=[
            reviews[review_id]
            for review_id in incident.review_ids[:samples]
            if review_id in reviews
        ],
    )


def _topic_item(topic: TopicView) -> schemas.TopicItem:
    return schemas.TopicItem.model_validate(topic)


def _day_range(daily: dict[int, dict[date, int]]) -> list[date]:
    """Every day from the first to the last review of a run, so topic series align."""
    days = {day for counts in daily.values() for day in counts}
    if not days:
        return []
    first, last = min(days), max(days)
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _evaluation_response(
    judgment_set: JudgmentSet, reports: list[ModeReport]
) -> schemas.EvaluationResponse:
    ndcg_by_query: dict[str, dict[SearchMode, float]] = {
        query.id: {} for query in judgment_set.queries
    }
    for report in reports:
        for metrics in report.queries:
            ndcg_by_query[metrics.query_id][report.mode] = metrics.ndcg

    return schemas.EvaluationResponse(
        k=EVALUATION_K,
        annotator=judgment_set.annotator,
        judgments=sum(len(query.judgments) for query in judgment_set.queries),
        modes=[
            schemas.ModeMetrics(
                mode=report.mode,
                ndcg=report.ndcg,
                precision=report.precision,
                mrr=report.mrr,
                judged_fraction=report.judged_fraction,
            )
            for report in reports
        ],
        queries=[
            schemas.QueryScores(id=query.id, text=query.text, ndcg=ndcg_by_query[query.id])
            for query in judgment_set.queries
        ],
    )
