"""Text embedding models behind a small protocol."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Maps texts to L2-normalized float32 vectors, so a dot product is cosine similarity.

    Documents and queries are embedded separately because asymmetric retrieval models
    (E5, BGE) expect a different instruction prefix for each.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed passages into an array of shape ``(len(texts), dimension)``."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a search query into an array of shape ``(dimension,)``."""
        ...


class SentenceTransformerEmbedder:
    """A local Hugging Face model served through sentence-transformers.

    Loading takes seconds, so create one instance per process. The import is deferred so
    commands that never embed do not pay for importing torch. Construct instances on the
    main thread; ``embed_*`` calls may then run in worker threads (see
    :func:`reviewradar.enrichment.wiring.build_enrichment_service`).
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 64,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"model {model_name!r} does not report an embedding dimension")

        self._model_name = model_name
        self._dimension = int(dimension)
        self._batch_size = batch_size
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        logger.info(
            "loaded embedding model=%s device=%s dimension=%d",
            model_name,
            self._model.device,
            self._dimension,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode([f"{self._document_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self._encode([f"{self._query_prefix}{text}"])[0]
        return vector

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        encoded = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors: NDArray[np.float32] = np.asarray(encoded, dtype=np.float32)
        return vectors
