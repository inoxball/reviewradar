"""Clustering of review embeddings into topics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Clustering:
    """The topic of every input vector and its cosine distance to that topic's centroid."""

    labels: NDArray[np.int64]
    distances: NDArray[np.float32]


class Clusterer(Protocol):
    @property
    def name(self) -> str: ...

    def parameters(self) -> dict[str, Any]:
        """Settings worth recording with a run, so its topics can be reproduced."""
        ...

    def fit(self, vectors: NDArray[np.float32]) -> Clustering: ...


class KMeansClusterer:
    """K-means on L2-normalized embeddings, so Euclidean clustering follows cosine similarity.

    Chosen after comparing on real Duolingo reviews (≤3★, 4+ words): mean aspect F1 of 0.42
    versus 0.38 for UMAP + HDBSCAN and 0.13 for HDBSCAN on PCA-50. It also has no noise
    bucket, is deterministic for a seed, and runs in about a second (notes §12).
    """

    name = "kmeans"

    def __init__(self, n_clusters: int, *, seed: int = 0, n_init: int = 4) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        self._n_clusters = n_clusters
        self._seed = seed
        self._n_init = n_init

    def parameters(self) -> dict[str, Any]:
        return {"n_clusters": self._n_clusters, "seed": self._seed, "n_init": self._n_init}

    def fit(self, vectors: NDArray[np.float32]) -> Clustering:
        from sklearn.cluster import KMeans

        if len(vectors) == 0:
            raise ValueError("cannot cluster an empty set of vectors")
        normalized = _normalize(np.asarray(vectors, dtype=np.float32))
        n_clusters = min(self._n_clusters, len(normalized))
        model = KMeans(n_clusters=n_clusters, n_init=self._n_init, random_state=self._seed)
        model.fit(normalized)

        labels: NDArray[np.int64] = np.asarray(model.labels_, dtype=np.int64)
        centroids = _normalize(np.asarray(model.cluster_centers_, dtype=np.float32))
        similarities = np.einsum("ij,ij->i", normalized, centroids[labels])
        distances: NDArray[np.float32] = np.clip(1.0 - similarities, 0.0, 2.0).astype(np.float32)
        return Clustering(labels=labels, distances=distances)


def import_clustering_runtime() -> None:
    """Import scikit-learn on the calling thread.

    Clustering runs in a worker thread. On Windows, a process whose first import of
    scikit-learn's native libraries (OpenMP, SciPy) happened in a worker thread crashes at
    interpreter exit, the same failure as with torch (notes §9). Composition roots call
    this on the main thread first.
    """
    import sklearn.cluster
    import sklearn.feature_extraction.text  # noqa: F401


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized: NDArray[np.float32] = (vectors / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalized
