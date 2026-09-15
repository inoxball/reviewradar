"""Local reply generators behind one protocol.

Three generators share one prompt (``prompting.build_messages``) and one loaded model:

- **zero-shot**: the base model reading the reply guidelines;
- **retrieval**: the same, plus the most similar answered reviews as worked examples;
- **fine-tuned**: the base model with the LoRA adapter trained on reply examples.

The 4-bit base model is loaded once. The prompted generators run with the adapter
disabled, so comparing them costs no extra GPU memory.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from reviewradar.replies.prompting import ReplyRequest, build_messages

logger = logging.getLogger(__name__)

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)
_QUOTED_PLACEHOLDER = re.compile(r"[`'\"](\{(?:app_name|support_contact)\})[`'\"]")

Conversation = list[dict[str, str]]
WorkedExample = tuple[ReplyRequest, str]


@dataclass(frozen=True, slots=True)
class ReplyTask:
    """A review to answer, with its stored embedding for retrieving similar examples."""

    request: ReplyRequest
    embedding: NDArray[np.float32] | None = None


class ReplyGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, tasks: Sequence[ReplyTask]) -> list[str]: ...


class CompletionModel(Protocol):
    @property
    def has_adapter(self) -> bool: ...

    def complete(self, conversations: Sequence[Conversation], *, use_adapter: bool) -> list[str]:
        """The assistant's next message for each conversation."""
        ...


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int = 160
    batch_size: int = 4
    repetition_penalty: float = 1.1


class LocalReplyModel:
    """A 4-bit quantized base model, optionally carrying the fine-tuned LoRA adapter.

    Decoding is greedy, so the same prompt always yields the same reply. Construct on the
    main thread: it imports torch (see the Windows note in the engineering notes).
    """

    def __init__(
        self,
        base_model: str,
        *,
        adapter_dir: Path | None = None,
        options: GenerationOptions | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch: Any = torch
        self._options = options or GenerationOptions()
        self._tokenizer: Any = AutoTokenizer.from_pretrained(base_model)
        self._tokenizer.padding_side = "left"
        model: Any = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self._has_adapter = (
            adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
        )
        if self._has_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        elif adapter_dir is not None:
            logger.warning("no adapter at %s; only prompted generators are available", adapter_dir)
        self._model: Any = model.eval()

    @property
    def has_adapter(self) -> bool:
        return self._has_adapter

    def complete(self, conversations: Sequence[Conversation], *, use_adapter: bool) -> list[str]:
        if use_adapter and not self._has_adapter:
            raise RuntimeError(
                "the fine-tuned adapter is not loaded; run `reviewradar replies train`"
            )
        adapter_off = (
            self._model.disable_adapter()
            if self._has_adapter and not use_adapter
            else contextlib.nullcontext()
        )
        replies: list[str] = []
        with adapter_off, self._torch.inference_mode():
            for start in range(0, len(conversations), self._options.batch_size):
                replies += self._complete_batch(
                    conversations[start : start + self._options.batch_size]
                )
        return replies

    def _complete_batch(self, conversations: Sequence[Conversation]) -> list[str]:
        prompts = [
            self._tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for conversation in conversations
        ]
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._model.device)
        generated = self._model.generate(
            **encoded,
            max_new_tokens=self._options.max_new_tokens,
            do_sample=False,
            repetition_penalty=self._options.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [_clean_completion(text) for text in texts]


class PromptedGenerator:
    """The base model following the guidelines, optionally with retrieved worked examples."""

    def __init__(
        self,
        model: CompletionModel,
        guidelines: str,
        *,
        name: str,
        examples: Callable[[NDArray[np.float32]], Sequence[WorkedExample]] | None = None,
    ) -> None:
        self._model = model
        self._guidelines = guidelines
        self._name = name
        self._examples = examples

    @property
    def name(self) -> str:
        return self._name

    def generate(self, tasks: Sequence[ReplyTask]) -> list[str]:
        conversations = [
            build_messages(task.request, self._guidelines, self._worked_examples(task))
            for task in tasks
        ]
        return self._model.complete(conversations, use_adapter=False)

    def _worked_examples(self, task: ReplyTask) -> Sequence[WorkedExample]:
        if self._examples is None:
            return ()
        if task.embedding is None:
            raise ValueError("retrieval needs the review's embedding; enrich the review first")
        return self._examples(task.embedding)


class FineTunedGenerator:
    """The base model with the LoRA adapter, prompted exactly as in training.

    ``system_prompt`` must be the one the adapter was trained with; it is recorded in the
    adapter's ``training.json``.
    """

    def __init__(self, model: CompletionModel, system_prompt: str, *, name: str) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, tasks: Sequence[ReplyTask]) -> list[str]:
        conversations = [build_messages(task.request, self._system_prompt) for task in tasks]
        return self._model.complete(conversations, use_adapter=True)


class NearestExamples:
    """The answered reviews most similar to a review, by cosine similarity of embeddings."""

    def __init__(
        self,
        embeddings: NDArray[np.float32],
        examples: Sequence[WorkedExample],
        *,
        count: int = 3,
    ) -> None:
        if len(embeddings) != len(examples):
            raise ValueError("every example needs exactly one embedding")
        self._vectors = _normalize(np.asarray(embeddings, dtype=np.float32))
        self._examples = list(examples)
        self._count = count

    def __call__(self, embedding: NDArray[np.float32]) -> list[WorkedExample]:
        query = _normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        similarities = self._vectors @ query
        closest = np.argsort(-similarities, kind="stable")[: self._count]
        # The most similar example goes last, right before the review to answer.
        return [self._examples[int(index)] for index in closest[::-1]]


def _clean_completion(text: str) -> str:
    """Drop any thinking block and quotes models put around placeholders (`{support_contact}`)."""
    return _QUOTED_PLACEHOLDER.sub(r"", " ".join(_THINKING.sub("", text).split()))


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized: NDArray[np.float32] = (vectors / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalized
