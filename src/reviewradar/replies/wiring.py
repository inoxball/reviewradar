"""Composition root for reply drafting."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import py3langid
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewradar.catalog import Catalog
from reviewradar.config import Settings
from reviewradar.db.session import create_engine, create_session_factory
from reviewradar.domain import Store, utc_now
from reviewradar.replies.benchmark import (
    BenchmarkExample,
    BenchmarkResult,
    run_benchmark,
    write_benchmark,
)
from reviewradar.replies.dataset import (
    DatasetReport,
    ReplyDatasetBuilder,
    ReplyDatasetRepository,
    ReplySource,
    ReviewKey,
)
from reviewradar.replies.drafts import ReplyDraftService, ReplyGeneratorSet
from reviewradar.replies.generation import (
    FineTunedGenerator,
    LocalReplyModel,
    NearestExamples,
    PromptedGenerator,
    ReplyGenerator,
    ReplyTask,
)
from reviewradar.replies.prompting import ReplyRequest, load_guidelines
from reviewradar.replies.training import MANIFEST_NAME, read_records

ZERO_SHOT = "zero-shot"
RETRIEVAL = "retrieval"
FINE_TUNED = "fine-tuned"
GENERATORS = (ZERO_SHOT, RETRIEVAL, FINE_TUNED)
RETRIEVED_EXAMPLES = 3


async def build_reply_dataset(settings: Settings) -> DatasetReport:
    engine = create_engine(settings.database_url)
    try:
        builder = ReplyDatasetBuilder(
            create_session_factory(engine),
            Catalog.from_yaml(settings.catalog_path),
            settings.replies,
            detect_language=detect_language,
        )
        return await builder.build()
    finally:
        await engine.dispose()


def run_reply_benchmark(
    settings: Settings,
    *,
    generator_names: Sequence[str] = GENERATORS,
    source: ReplySource | None = ReplySource.WRITTEN,
    limit: int | None = None,
) -> BenchmarkResult:
    """Draft replies for held-out reviews with every requested generator and score them.

    Examples are ordered by a hash of their key, so ``limit`` always picks the same subset.
    Loads the model on the calling thread, which must be the main thread (torch on Windows).
    """
    unknown = set(generator_names) - set(GENERATORS)
    if unknown:
        raise ValueError(f"unknown generators {sorted(unknown)}; choose from {GENERATORS}")
    replies = settings.replies
    catalog = Catalog.from_yaml(settings.catalog_path)
    eval_records = [
        record
        for record in read_records(replies.dataset_dir / "eval.jsonl")
        if source is None or record["source"] == source.value
    ]
    eval_records.sort(key=lambda record: hashlib.sha256(record["key"].encode()).hexdigest())
    eval_records = eval_records[:limit]
    train_records = (
        read_records(replies.dataset_dir / "train.jsonl") if RETRIEVAL in generator_names else []
    )
    embeddings = asyncio.run(
        _embeddings(settings, (_review_key(record) for record in [*eval_records, *train_records]))
    )

    app_names = {app.slug: app.name for app in catalog.apps}
    examples = [
        BenchmarkExample(
            key=record["key"],
            app_name=app_names[record["app"]],
            support_contact=catalog.get(record["app"]).support_contact,
            foreign_brands=tuple(name for slug, name in app_names.items() if slug != record["app"]),
            task=ReplyTask(_request(record), embeddings.get(_review_key(record))),
            reference=record["reply"],
            record=record,
        )
        for record in eval_records
    ]

    manifest = _adapter_manifest(settings) if FINE_TUNED in generator_names else None
    model = LocalReplyModel(
        replies.base_model, adapter_dir=replies.adapter_dir if manifest is not None else None
    )
    guidelines = load_guidelines(replies.guidelines_path)
    generators: list[ReplyGenerator] = []
    if ZERO_SHOT in generator_names:
        generators.append(PromptedGenerator(model, guidelines, name=ZERO_SHOT))
    if RETRIEVAL in generator_names:
        indexed = [record for record in train_records if _review_key(record) in embeddings]
        generators.append(
            PromptedGenerator(
                model,
                guidelines,
                name=RETRIEVAL,
                examples=NearestExamples(
                    np.stack([embeddings[_review_key(record)] for record in indexed]),
                    [(_request(record), record["reply"]) for record in indexed],
                    count=RETRIEVED_EXAMPLES,
                ),
            )
        )
    if manifest is not None:
        generators.append(FineTunedGenerator(model, manifest["system_prompt"], name=FINE_TUNED))

    result = run_benchmark(examples, generators, detect_language=detect_language)
    adapter = (
        {key: manifest[key] for key in ("created_at", "dataset", "report", "system_prompt")}
        if manifest is not None
        else None
    )
    write_benchmark(
        result,
        examples,
        replies.benchmark_dir,
        metadata={
            "created_at": utc_now().isoformat(),
            "base_model": replies.base_model,
            "source": source.value if source else "all",
            "examples": len(examples),
            "adapter": adapter,
        },
    )
    return result


def create_reply_draft_service(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    catalog: Catalog,
) -> ReplyDraftService:
    """A draft service whose generators load on the first request.

    Generators are offered when their inputs exist: zero-shot always, retrieval once the
    dataset is built, fine-tuned once an adapter is trained. Call
    :func:`import_reply_runtime` on the main thread before the first request.
    """
    replies = settings.replies
    train_path = replies.dataset_dir / "train.jsonl"
    manifest_path = replies.adapter_dir / MANIFEST_NAME
    available = [ZERO_SHOT]
    if train_path.exists():
        available.append(RETRIEVAL)
    if manifest_path.exists():
        available.append(FINE_TUNED)

    async def load_generators() -> ReplyGeneratorSet:
        guidelines = load_guidelines(replies.guidelines_path)
        prompt_version = f"{replies.base_model}+guidelines:{_fingerprint(guidelines)}"
        manifest = _adapter_manifest(settings) if FINE_TUNED in available else None
        model = await asyncio.to_thread(
            LocalReplyModel,
            replies.base_model,
            adapter_dir=replies.adapter_dir if manifest is not None else None,
        )
        generators: dict[str, ReplyGenerator] = {
            ZERO_SHOT: PromptedGenerator(model, guidelines, name=ZERO_SHOT)
        }
        versions = {ZERO_SHOT: prompt_version}
        if RETRIEVAL in available:
            records = read_records(train_path)
            async with session_factory() as session:
                embeddings = await ReplyDatasetRepository(session).embeddings(
                    (_review_key(record) for record in records),
                    model=settings.enrichment.embedding_model,
                )
            indexed = [record for record in records if _review_key(record) in embeddings]
            generators[RETRIEVAL] = PromptedGenerator(
                model,
                guidelines,
                name=RETRIEVAL,
                examples=NearestExamples(
                    np.stack([embeddings[_review_key(record)] for record in indexed]),
                    [(_request(record), record["reply"]) for record in indexed],
                    count=RETRIEVED_EXAMPLES,
                ),
            )
            versions[RETRIEVAL] = (
                f"{prompt_version}+examples:{_fingerprint(train_path.read_text('utf-8'))}"
            )
        if manifest is not None:
            generators[FINE_TUNED] = FineTunedGenerator(
                model, manifest["system_prompt"], name=FINE_TUNED
            )
            versions[FINE_TUNED] = f"{replies.base_model}+adapter:{manifest['created_at']}"
        return ReplyGeneratorSet(generators, versions, requires_embedding=frozenset({RETRIEVAL}))

    return ReplyDraftService(
        session_factory,
        catalog,
        load_generators,
        available=available,
        embedding_model=settings.enrichment.embedding_model,
        detect_language=detect_language,
    )


def import_reply_runtime() -> None:
    """Import torch and the quantization and adapter libraries on the calling thread.

    Call it on the main thread: first importing these native libraries in a worker thread
    crashes the interpreter at exit on Windows (engineering notes, §9.3).
    """
    import bitsandbytes  # noqa: F401
    import peft  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401


def detect_language(text: str) -> str | None:
    language, _ = py3langid.classify(text)
    return str(language) if language else None


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


async def _embeddings(
    settings: Settings, keys: Iterable[ReviewKey]
) -> dict[ReviewKey, NDArray[np.float32]]:
    engine = create_engine(settings.database_url)
    try:
        async with create_session_factory(engine)() as session:
            return await ReplyDatasetRepository(session).embeddings(
                keys, model=settings.enrichment.embedding_model
            )
    finally:
        await engine.dispose()


def _adapter_manifest(settings: Settings) -> dict[str, Any]:
    path = settings.replies.adapter_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no trained adapter at {path}; run `reviewradar replies train`")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def _review_key(record: dict[str, Any]) -> ReviewKey:
    app, store, external_id = str(record["key"]).split(":", 2)
    return (app, Store(store), external_id)


def _request(record: dict[str, Any]) -> ReplyRequest:
    return ReplyRequest(
        store=Store(record["store"]),
        rating=int(record["rating"]),
        language=record["language"],
        review=record["review"],
    )
