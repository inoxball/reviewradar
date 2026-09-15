"""Request and response models of the panel API, also documented at ``/docs``."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reviewradar.domain import LanguageSource, RunStatus, Store
from reviewradar.replies.drafts import DraftStatus
from reviewradar.search.types import SearchMode
from reviewradar.translation.service import TranslationOutcome


class _FromAttributes(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DatabaseInfo(_FromAttributes):
    backend: str
    server_version: str
    pgvector: str | None
    revision: str | None = Field(description="Applied migration; null if never migrated.")
    head_revision: str = Field(description="Newest migration bundled with this version.")
    migrations_current: bool = Field(description="False means: run reviewradar db upgrade.")


class TranslationInfo(BaseModel):
    model: str
    target_language: str
    loaded: bool = Field(description="Whether the model is in memory; it loads on first use.")


class SystemInfo(BaseModel):
    version: str
    database: DatabaseInfo
    embedding_model: str
    translation: TranslationInfo


class AppInfo(BaseModel):
    slug: str
    name: str


class MarketBreakdown(_FromAttributes):
    store: Store
    country: str | None
    review_count: int
    average_rating: float
    oldest_reviewed_at: datetime
    newest_reviewed_at: datetime


class LanguageCount(_FromAttributes):
    store: Store
    language: str | None
    review_count: int


class RatingCount(_FromAttributes):
    store: Store
    rating: int
    review_count: int


class PipelineCounts(_FromAttributes):
    reviews: int
    enriched: int
    embedded: int
    translated: int
    language_sources: dict[LanguageSource, int]


class Overview(BaseModel):
    app: AppInfo
    total_reviews: int
    average_rating: float | None
    languages: int
    markets: list[MarketBreakdown]
    language_mix: list[LanguageCount]
    ratings: list[RatingCount]
    pipeline: PipelineCounts


class ReviewItem(_FromAttributes):
    review_id: int
    store: Store
    external_id: str
    rating: int
    country: str | None
    language: str | None
    app_version: str | None
    reviewed_at: datetime
    text: str
    translation: str | None = Field(
        default=None, description="Stored English translation, when one exists."
    )


class ReviewHit(ReviewItem):
    score: float


class ReviewPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ReviewItem]


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    took_ms: float
    results: list[ReviewHit]


class CompareResponse(BaseModel):
    query: str
    modes: list[SearchResponse]


class IngestionRunItem(_FromAttributes):
    id: int
    store: Store
    partition_key: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    fetched: int
    inserted: int
    updated: int
    unchanged: int
    coverage_complete: bool | None
    error: str | None


class CursorItem(_FromAttributes):
    store: Store
    partition_key: str
    newest_reviewed_at: datetime
    updated_at: datetime


class IngestionStatus(BaseModel):
    runs: list[IngestionRunItem]
    cursors: list[CursorItem]


class TranslationRequest(BaseModel):
    review_ids: list[int] = Field(min_length=1, max_length=50)


class TranslationItem(_FromAttributes):
    review_id: int
    outcome: TranslationOutcome
    source_language: str | None
    text: str | None


class TranslationResponse(BaseModel):
    model: str
    target_language: str
    took_ms: float
    items: list[TranslationItem]


class ModeMetrics(BaseModel):
    mode: SearchMode
    ndcg: float
    precision: float
    mrr: float
    judged_fraction: float


class QueryScores(BaseModel):
    id: str
    text: str
    ndcg: dict[SearchMode, float]


class EvaluationResponse(BaseModel):
    k: int
    annotator: str
    judgments: int
    modes: list[ModeMetrics]
    queries: list[QueryScores]


class TopicRunInfo(_FromAttributes):
    id: int
    algorithm: str
    embedding_model: str
    parameters: dict[str, Any] = Field(description="Clustering settings and review scope.")
    review_count: int
    topic_count: int
    created_at: datetime


class DailyCount(BaseModel):
    day: date
    count: int


class TopicItem(_FromAttributes):
    id: int
    label: str
    keywords: list[str] = Field(description="Most distinctive English terms, strongest first.")
    size: int
    average_rating: float
    negative_share: float = Field(description="Share of reviews rated 1 or 2 stars.")
    languages: dict[str, int]
    daily: list[DailyCount] = Field(
        default_factory=list, description="Reviews per day over the run's full date range."
    )
    samples: list[ReviewItem] = Field(
        default_factory=list, description="The reviews closest to the topic's centre."
    )


class TopicsResponse(BaseModel):
    run: TopicRunInfo | None = Field(description="The latest run; null before any discovery.")
    topics: list[TopicItem]


class TopicReviewPage(ReviewPage):
    topic: TopicItem


class TopicRunRequest(BaseModel):
    topic_count: int | None = Field(default=None, ge=2, le=100)
    max_rating: int | None = Field(default=None, ge=1, le=5)
    all_ratings: bool = Field(default=False, description="Include reviews of every rating.")
    min_words: int | None = Field(default=None, ge=1)


class TopicRunResponse(BaseModel):
    run_id: int
    review_count: int
    topic_count: int
    took_ms: float


class ReplyDraftRequest(BaseModel):
    review_ids: list[int] = Field(min_length=1, max_length=10)
    generator: str = Field(default="fine-tuned", description="zero-shot, retrieval or fine-tuned.")


class ReplyChecksItem(_FromAttributes):
    non_empty: bool
    right_language: bool
    within_limit: bool = Field(description="At most 350 characters once placeholders are filled.")
    no_foreign_brand_or_contact: bool
    valid_placeholders: bool
    not_copied: bool = Field(description="Shares no 8-word sequence with the review.")
    complete: bool = Field(description="Ends like a finished sentence, not cut off.")
    passed: bool


class ReplyDraftItem(_FromAttributes):
    review_id: int
    generator: str
    status: DraftStatus
    text: str | None = Field(description="Rendered for the review's app, ready to post.")
    template: str | None = Field(description="As generated, with placeholders.")
    checks: ReplyChecksItem | None
    created_at: datetime | None


class ReplyDraftResponse(BaseModel):
    generator: str
    took_ms: float
    items: list[ReplyDraftItem]


class GeneratorBenchmark(BaseModel):
    name: str
    replies: int
    pass_rate: float
    check_rates: dict[str, float]
    opening_diversity: float = Field(description="Distinct 5-word openings per reply.")
    median_chars: float
    seconds_per_reply: float


class ReplyBenchmark(BaseModel):
    created_at: datetime
    source: str
    examples: int
    generators: list[GeneratorBenchmark]


class ReplyGeneratorsResponse(BaseModel):
    generators: list[str]
    loaded: bool = Field(description="Whether the model is in memory; it loads on first use.")
    benchmark: ReplyBenchmark | None = Field(description="Latest offline benchmark, if any.")


class SourceCoverageItem(_FromAttributes):
    source: str = Field(description="Store and market, e.g. app_store:de or google_play:pt.")
    first_day: date
    last_day: date
    usable_days: int = Field(description="Days strictly between the first and the last.")
    tested: bool = Field(description="Sources with too few usable days are not tested.")


class SpikeItem(BaseModel):
    topic: TopicItem
    source: str
    day: date
    count: int = Field(description="Reviews in the topic from this source on this day.")
    total: int = Field(description="All critical reviews from this source on this day.")
    share: float
    expected_share: float = Field(description="The topic's share on the source's other days.")
    lift: float
    p_value: float = Field(description="Exact binomial test of the share against the baseline.")
    top_version: str | None
    top_version_share: float
    samples: list[ReviewItem]


class AnomalyResponse(BaseModel):
    run: TopicRunInfo | None = Field(description="The topic run tested; null before discovery.")
    tests: int = Field(description="Topic-days tested across all sources.")
    threshold: float = Field(description="p-value a spike must stay below (Bonferroni).")
    coverage: list[SourceCoverageItem]
    spikes: list[SpikeItem]


class MonitorDay(BaseModel):
    day: date
    sources: int = Field(description="Sources with enough earlier coverage to judge the day.")
    tests: int
    alerts: int


class IncidentItem(BaseModel):
    topic: TopicItem
    source: str
    first_day: date
    last_day: date
    alert_days: int
    review_count: int = Field(description="Reviews in the topic on the alerting days.")
    peak_day: date = Field(description="The most significant alerting day.")
    peak_count: int
    peak_total: int = Field(description="All critical reviews from the source on the peak day.")
    peak_share: float
    expected_share: float = Field(description="The topic's share on the earlier days judged.")
    lift: float
    p_value: float
    baseline_days: int = Field(description="Earlier usable days the peak day was judged on.")
    top_version: str | None
    top_version_share: float
    release: str | None = Field(
        description="A recently first-seen app version most of the incident is on: a possible "
        "release regression."
    )
    samples: list[ReviewItem]


class SpikeOutcomeItem(BaseModel):
    topic: TopicItem
    source: str
    day: date = Field(description="Day of the spike found in hindsight.")
    caught: bool
    first_alert_day: date | None
    delay_days: int | None = Field(description="Negative when the monitor alerted earlier.")
    reason: str | None = Field(description="Why the monitor missed it.")


class MonitorBacktestResponse(BaseModel):
    run: TopicRunInfo | None = Field(description="The topic run replayed; null before discovery.")
    baseline_days: int = Field(description="Trailing calendar days each day is judged on.")
    min_baseline_days: int
    daily_alpha: float = Field(description="False-alarm budget per monitored day.")
    days: list[MonitorDay]
    judged_days: int
    alert_days: int
    tests: int
    incidents: list[IncidentItem]
    spikes: list[SpikeOutcomeItem] = Field(
        description="Spikes found in hindsight, and whether the monitor would have alerted."
    )
