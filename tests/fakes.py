"""Deterministic stand-ins for the models used by enrichment."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, L2-normalized, cheap.

    Texts sharing words get similar vectors, which keeps it meaningful for search tests.
    Records every document it embeds.
    """

    def __init__(self, model_name: str = "fake/hashing", dimension: int = 16) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.documents: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.documents.extend(texts)
        return np.stack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            vector[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class FixedLanguageDetector:
    """Always answers with the same language and confidence; records what it was asked."""

    def __init__(self, language: str, confidence: float) -> None:
        self.language = language
        self.confidence = confidence
        self.texts: list[str] = []

    def detect(self, text: str) -> tuple[str, float]:
        self.texts.append(text)
        return self.language, self.confidence


class FakeTranslator:
    """Tags texts instead of translating them; records every call and its source language."""

    def __init__(
        self,
        model_name: str = "fake/translator",
        supported: frozenset[str] = frozenset({"de", "en", "es", "ja", "pt"}),
    ) -> None:
        self.model_name = model_name
        self.supported = supported
        self.calls: list[tuple[str, list[str]]] = []

    def supports(self, language: str) -> bool:
        return language in self.supported

    def translate(
        self, texts: Sequence[str], *, source_language: str, target_language: str
    ) -> list[str]:
        self.calls.append((source_language, list(texts)))
        return [f"[{source_language}→{target_language}] {text}" for text in texts]
