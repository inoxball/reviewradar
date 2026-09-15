"""Translation models behind a small protocol, plus lazy loading for interactive use."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Translator(Protocol):
    """Translates batches of texts that share one source language.

    Implementations are not required to be thread-safe; callers serialize access.
    """

    @property
    def model_name(self) -> str: ...

    def supports(self, language: str) -> bool:
        """Whether ``language`` (ISO 639-1) can be translated from."""
        ...

    def translate(
        self, texts: Sequence[str], *, source_language: str, target_language: str
    ) -> list[str]: ...


class M2M100Translator:
    """Facebook's M2M100 many-to-many model (MIT licence), running locally.

    One model covers 100 languages and uses the same ISO 639-1 codes as language detection,
    so no code mapping is needed. On CUDA the weights run in float16 (~1 GB of VRAM).
    Construct it on the main thread, or after :func:`import_translation_runtime` ran there:
    first importing torch inside a worker thread crashes the interpreter at exit on Windows.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_input_tokens: int = 256,
        num_beams: int = 2,
    ) -> None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self._torch: Any = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Any = M2M100Tokenizer.from_pretrained(model_name)
        model: Any = M2M100ForConditionalGeneration.from_pretrained(model_name)
        if self._device.startswith("cuda"):
            model = model.half()
        self._model: Any = model.to(self._device).eval()
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_input_tokens = max_input_tokens
        self._num_beams = num_beams
        logger.info("loaded translation model=%s device=%s", model_name, self._device)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def supports(self, language: str) -> bool:
        return language in self._tokenizer.lang_code_to_id

    def translate(
        self, texts: Sequence[str], *, source_language: str, target_language: str
    ) -> list[str]:
        if not texts:
            return []
        self._tokenizer.src_lang = source_language
        target_token = self._tokenizer.get_lang_id(target_language)

        translations = [""] * len(texts)
        for batch in length_sorted_batches(texts, self._batch_size):
            encoded = self._tokenizer(
                [texts[index] for index in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_input_tokens,
            ).to(self._device)
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    forced_bos_token_id=target_token,
                    num_beams=self._num_beams,
                    max_length=self._max_input_tokens,
                )
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, translation in zip(batch, decoded, strict=True):
                translations[index] = translation
        return translations


def length_sorted_batches(texts: Sequence[str], batch_size: int) -> list[list[int]]:
    """Indices of ``texts`` grouped into batches of similar length, longest first.

    Similar lengths stop a one-word review from being padded to a paragraph's length.
    On 256 Portuguese reviews this raised throughput from 8.0 to 14.2 reviews/s, with
    255 of 256 translations unchanged (notes §11.3). Longest first surfaces running out
    of memory on the first batch rather than the last.
    """
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]), reverse=True)
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


class LazyTranslator:
    """Loads the wrapped translator on first use, so interactive services start instantly.

    Loading happens on whichever thread first needs the model, typically a worker thread;
    call :func:`import_translation_runtime` on the main thread beforehand.
    """

    def __init__(self, factory: Callable[[], Translator], *, model_name: str) -> None:
        self._factory = factory
        self._model_name = model_name
        self._lock = threading.Lock()
        self._translator: Translator | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._translator is not None

    def supports(self, language: str) -> bool:
        return self._load().supports(language)

    def translate(
        self, texts: Sequence[str], *, source_language: str, target_language: str
    ) -> list[str]:
        return self._load().translate(
            texts, source_language=source_language, target_language=target_language
        )

    def _load(self) -> Translator:
        with self._lock:
            if self._translator is None:
                self._translator = self._factory()
            return self._translator


def import_translation_runtime() -> None:
    """Import torch and the M2M100 classes on the calling thread (call it on the main thread)."""
    import torch  # noqa: F401
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # noqa: F401
