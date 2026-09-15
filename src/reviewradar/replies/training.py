"""QLoRA fine-tuning of the reply generator.

Training runs offline from the dataset files, not from the database, so a run is fully
described by its inputs. It writes a LoRA adapter and a ``training.json`` manifest with the
base model, dataset fingerprints, settings and losses next to it.

The adapter is trained with a compact system prompt (``fine_tune_system_prompt``) instead of
the full reply guidelines: the guidelines made up 78% of training tokens without lowering the
base model's reply loss. The prompt is stored in the manifest, so inference uses exactly the
prompt the adapter was trained with, in Qwen3's chat template with thinking disabled. Loss is
computed on the reply tokens only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from reviewradar.config import ReplySettings
from reviewradar.domain import Store, utc_now
from reviewradar.replies.prompting import ReplyRequest, build_messages

logger = logging.getLogger(__name__)

MANIFEST_NAME = "training.json"


class ChatTemplate(Protocol):
    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrainingReport:
    adapter_dir: Path
    steps: int
    train_loss: float
    eval_loss: float
    runtime_seconds: float
    peak_memory_gb: float


def read_records(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def to_prompt_completion(
    records: Iterable[dict[str, Any]],
    tokenizer: ChatTemplate,
    system_prompt: str,
    *,
    written_repeats: int = 1,
) -> list[dict[str, str]]:
    """Render dataset records as prompt/completion pairs in the model's chat format."""
    pairs: list[dict[str, str]] = []
    for record in records:
        request = ReplyRequest(
            store=Store(record["store"]),
            rating=int(record["rating"]),
            language=record["language"],
            review=record["review"],
        )
        prompt = tokenizer.apply_chat_template(
            build_messages(request, system_prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        repeats = written_repeats if record["source"] == "written" else 1
        pairs.extend({"prompt": str(prompt), "completion": record["reply"]} for _ in range(repeats))
    return pairs


def train_reply_adapter(settings: ReplySettings, *, max_steps: int | None = None) -> TrainingReport:
    """Fine-tune a LoRA adapter on the 4-bit base model and save it with its manifest.

    Imports torch, so call it on the main thread (see the Windows note in the engineering notes).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    options = settings.training
    train_path = settings.dataset_dir / "train.jsonl"
    eval_path = settings.dataset_dir / "eval.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    system_prompt = settings.fine_tune_system_prompt

    train_pairs = to_prompt_completion(
        read_records(train_path), tokenizer, system_prompt, written_repeats=options.written_repeats
    )
    random.Random(options.seed).shuffle(train_pairs)
    eval_records = read_records(eval_path)
    random.Random(options.seed).shuffle(eval_records)
    eval_pairs = to_prompt_completion(
        eval_records[: options.eval_examples], tokenizer, system_prompt
    )
    logger.info("reply training pairs=%d eval=%d", len(train_pairs), len(eval_pairs))

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model,
        quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    checkpoint_dir = settings.adapter_dir.parent / f"{settings.adapter_dir.name}-checkpoints"
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(checkpoint_dir),
            num_train_epochs=options.epochs,
            max_steps=max_steps or -1,
            per_device_train_batch_size=options.batch_size,
            per_device_eval_batch_size=options.batch_size,
            gradient_accumulation_steps=options.gradient_accumulation_steps,
            learning_rate=options.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=options.warmup_steps,
            max_length=options.max_length,
            completion_only_loss=True,
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=options.eval_steps,
            save_strategy="no",
            report_to="none",
            seed=options.seed,
            dataloader_num_workers=0,
        ),
        train_dataset=Dataset.from_list(train_pairs),
        eval_dataset=Dataset.from_list(eval_pairs),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=options.lora_rank,
            lora_alpha=options.lora_alpha,
            lora_dropout=options.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    _log_first_example(trainer, tokenizer)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    runtime = time.perf_counter() - started
    eval_loss = float(trainer.evaluate()["eval_loss"])

    settings.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(settings.adapter_dir)
    tokenizer.save_pretrained(settings.adapter_dir)
    report = TrainingReport(
        adapter_dir=settings.adapter_dir,
        steps=int(result.global_step),
        train_loss=float(result.training_loss),
        eval_loss=eval_loss,
        runtime_seconds=runtime,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    _write_manifest(settings, report, [train_path, eval_path], len(train_pairs), trainer)
    return report


def _log_first_example(trainer: Any, tokenizer: Any) -> None:
    """Show which tokens carry loss, to catch template or masking mistakes before training.

    TRL folds the completion mask into ``labels`` (-100 on prompt tokens) and drops the mask.
    """
    example = trainer.train_dataset[0]
    ids: Sequence[int] = example["input_ids"]
    labels: Sequence[int] | None = example.get("labels")
    if labels is None:
        mask: Sequence[int] = example.get("completion_mask") or [1] * len(ids)
        labels = [token if keep else -100 for token, keep in zip(ids, mask, strict=True)]
    trained = [token for token in labels if token != -100]
    logger.info(
        "first training example: %d tokens, %d with loss: %r",
        len(ids),
        len(trained),
        tokenizer.decode(trained),
    )


def _write_manifest(
    settings: ReplySettings,
    report: TrainingReport,
    dataset_files: Sequence[Path],
    train_pairs: int,
    trainer: Any,
) -> None:
    manifest = {
        "created_at": utc_now().isoformat(),
        "base_model": settings.base_model,
        "system_prompt": settings.fine_tune_system_prompt,
        "dataset": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset_files
        },
        "train_pairs": train_pairs,
        "settings": settings.training.model_dump(),
        "report": {**asdict(report), "adapter_dir": str(report.adapter_dir)},
        "log_history": trainer.state.log_history,
    }
    (settings.adapter_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
