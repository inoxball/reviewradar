"""Application settings, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from reviewradar.domain import Store


class IngestionSettings(BaseModel):
    """Tuning knobs for review collection."""

    max_concurrent_fetches: int = Field(default=4, ge=1, le=32)
    google_play_page_size: int = Field(default=200, ge=1, le=200)
    page_delay_seconds: float = Field(default=0.5, ge=0)
    http_timeout_seconds: float = Field(default=20.0, gt=0)
    max_attempts: int = Field(default=5, ge=1)
    default_lookback_days: int = Field(default=30, ge=1)
    incremental_overlap_hours: int = Field(
        default=72,
        ge=0,
        description=(
            "How far behind the saved cursor an incremental run starts. Stores publish "
            "reviews after moderation delays of up to ~72h, so late arrivals can carry "
            "timestamps older than the newest review already ingested."
        ),
    )
    max_reviews_per_market: int = Field(default=5000, ge=1)
    overlap_hours_by_store: dict[Store, NonNegativeInt] = Field(
        default_factory=lambda: {Store.GOOGLE_PLAY: 14 * 24},
        description=(
            "Per-store overrides of `incremental_overlap_hours`. Google Play's public stream "
            "was observed surfacing reviews up to ~2 weeks after their timestamp."
        ),
    )

    def overlap_for(self, store: Store) -> timedelta:
        """How far behind the saved cursor an incremental run for ``store`` starts."""
        return timedelta(
            hours=self.overlap_hours_by_store.get(store, self.incremental_overlap_hours)
        )


class EnrichmentSettings(BaseModel):
    """Tuning knobs for text preparation, language detection and embeddings."""

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str | None = Field(
        default=None, description="Torch device such as 'cuda' or 'cpu'; auto-detected if unset."
    )
    embedding_batch_size: int = Field(default=64, ge=1)
    document_prefix: str = Field(
        default="", description="Prefix some models require for passages, e.g. 'passage: ' (E5)."
    )
    query_prefix: str = Field(
        default="", description="Prefix some models require for queries, e.g. 'query: ' (E5)."
    )
    review_batch_size: int = Field(
        default=512, ge=1, le=2000, description="Reviews loaded, processed and committed together."
    )
    min_language_confidence: float = Field(default=0.9, ge=0, le=1)
    min_corroborated_language_confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Lower bar for detections that agree with the store hint or market default.",
    )
    min_language_chars: int = Field(
        default=20, ge=1, description="Shorter texts use the store's language hint instead."
    )


class SearchSettings(BaseModel):
    """Tuning knobs for review search."""

    candidate_pool_size: int = Field(
        default=50, ge=1, le=1000, description="Results each retriever contributes to fusion."
    )
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal rank fusion damping constant.")


class TranslationSettings(BaseModel):
    """Local machine translation of reviews into one target language."""

    model: str = Field(
        default="facebook/m2m100_418M",
        description="M2M100 checkpoint (MIT licence); covers 100 languages with ISO 639-1 codes.",
    )
    device: str | None = Field(
        default=None, description="Torch device such as 'cuda' or 'cpu'; auto-detected if unset."
    )
    target_language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    batch_size: int = Field(default=16, ge=1)
    max_input_tokens: int = Field(default=256, ge=16, le=1024)
    num_beams: int = Field(default=2, ge=1, le=8)
    review_batch_size: int = Field(
        default=256, ge=1, le=2000, description="Reviews loaded, translated and committed together."
    )


class TopicSettings(BaseModel):
    """Topic discovery over stored review embeddings."""

    topic_count: int = Field(default=20, ge=2, le=200)
    max_rating: int | None = Field(
        default=3,
        ge=1,
        le=5,
        description="Only reviews rated at most this take part; unset to include every rating.",
    )
    min_words: int = Field(
        default=4, ge=1, description="Very short reviews ('good app') carry no topic."
    )
    keywords_per_topic: int = Field(default=6, ge=1, le=20)
    naming_model: str = Field(
        default="Qwen/Qwen3-1.7B", description="Local instruction model that names topics."
    )
    name_examples: int = Field(
        default=10, ge=1, le=30, description="Typical reviews the naming model reads per topic."
    )
    seed: int = 0
    min_reviews: int = Field(default=50, ge=2)
    runs_to_keep: int = Field(default=3, ge=1)


class AnomalySettings(BaseModel):
    """Spike detection on topic volume, per review source (store and market)."""

    max_rating: int | None = Field(
        default=3,
        ge=1,
        le=5,
        description="Only reviews rated at most this are tested, whatever the topic run covers.",
    )
    min_count: int = Field(
        default=5, ge=1, description="A spike needs at least this many reviews in the topic."
    )
    min_lift: float = Field(
        default=3.0, gt=1, description="Topic share of the day versus its usual share."
    )
    family_alpha: float = Field(
        default=0.01,
        gt=0,
        lt=1,
        description="False-alarm rate across all tests; split evenly across them (Bonferroni).",
    )
    min_usable_days: int = Field(
        default=5, ge=2, description="Sources with fewer fully covered days are not tested."
    )
    samples: int = Field(default=3, ge=0, le=10, description="Example reviews per spike.")
    # The daily monitor judges each day against earlier days only (anomalies/monitor.py).
    baseline_days: int = Field(
        default=28, ge=2, le=365, description="Monitor: trailing calendar days a day is judged on."
    )
    min_baseline_days: int = Field(
        default=5, ge=1, description="Monitor: earlier usable days a source needs to be judged."
    )
    daily_alpha: float = Field(
        default=0.01,
        gt=0,
        lt=1,
        description="Monitor: false-alarm rate per monitored day, split across that day's tests.",
    )
    incident_gap_days: int = Field(
        default=1, ge=0, le=7, description="Monitor: quiet days allowed inside one incident."
    )
    new_version_days: int = Field(
        default=7,
        ge=1,
        description="Monitor: days before an incident to compare versions with.",
    )
    max_prior_version_share: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Monitor: a version already this common before an incident is not blamed.",
    )
    release_share: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Monitor: share of an incident's reviews on one new version to blame it.",
    )


class ReplyTrainingSettings(BaseModel):
    """QLoRA fine-tuning; defaults fit a 4 GB laptop GPU."""

    epochs: float = Field(default=2.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_steps: int = Field(default=20, ge=0)
    max_length: int = Field(default=1024, ge=128, description="Prompt plus reply, in tokens.")
    written_repeats: int = Field(
        default=3, ge=1, description="Written references follow the guidelines; upweight them."
    )
    eval_examples: int = Field(default=150, ge=1, description="Held-out examples scored for loss.")
    eval_steps: int = Field(default=50, ge=1)
    seed: int = 0


class ReplySettings(BaseModel):
    """Reply drafting: datasets, the local model and its fine-tuned adapter."""

    guidelines_path: Path = Field(
        default=Path("config/reply_guidelines.md"),
        description="System prompt of every generator and brief for written references.",
    )
    written_dir: Path = Field(
        default=Path("data/replies/written"), description="Hand-written reference replies."
    )
    dataset_dir: Path = Path("data/replies/dataset")
    source_apps: list[str] = Field(
        default_factory=lambda: ["babbel", "mondly", "memrise"],
        description="Catalog apps whose published developer replies become training data.",
    )
    eval_fraction_published: float = Field(default=0.1, ge=0, lt=1)
    eval_fraction_written: float = Field(
        default=0.35, ge=0, lt=1, description="Written references are scarce and in-domain."
    )
    base_model: str = Field(default="Qwen/Qwen3-1.7B", description="Apache-2.0 licensed.")
    adapter_dir: Path = Path("models/replies/qwen3-1.7b-lora")
    fine_tune_system_prompt: str = Field(
        default=(
            "You write short developer replies to app store reviews, in the review's language "
            "and within 350 characters. Write {app_name} and {support_contact} instead of brand "
            "names and contact details."
        ),
        description=(
            "System prompt used to train and run the adapter. The full guidelines made up 78% of "
            "training tokens without lowering the base model's reply loss (5.17 vs 5.14)."
        ),
    )
    training: ReplyTrainingSettings = Field(default_factory=ReplyTrainingSettings)
    benchmark_dir: Path = Field(
        default=Path("eval/replies"),
        description="Latest benchmark: every generator's drafts and their checks.",
    )


class WebSettings(BaseModel):
    """The demo web panel."""

    judgments_path: Path = Field(
        default=Path("eval/duolingo_search_judgments.yaml"),
        description="Relevance judgments scored on the panel's Evaluation tab.",
    )


class Settings(BaseSettings):
    """Top-level settings. Every field maps to a ``REVIEWRADAR_*`` environment variable."""

    model_config = SettingsConfigDict(
        env_prefix="REVIEWRADAR_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    database_url: str = "sqlite+aiosqlite:///./reviewradar.db"
    log_level: str = "INFO"
    catalog_path: Path = Path("config/apps.yaml")
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    enrichment: EnrichmentSettings = Field(default_factory=EnrichmentSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    topics: TopicSettings = Field(default_factory=TopicSettings)
    anomalies: AnomalySettings = Field(default_factory=AnomalySettings)
    replies: ReplySettings = Field(default_factory=ReplySettings)
    web: WebSettings = Field(default_factory=WebSettings)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
