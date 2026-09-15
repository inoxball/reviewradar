"""The real embedding model from the local Hugging Face cache. Run with ``pytest -m ml``."""

import numpy as np
import pytest

from reviewradar.config import EnrichmentSettings
from reviewradar.enrichment.embeddings import SentenceTransformerEmbedder

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(EnrichmentSettings().embedding_model)


def test_documents_are_normalized_float32_vectors(embedder: SentenceTransformerEmbedder) -> None:
    vectors = embedder.embed_documents(["Great app", "Terrible update"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, embedder.dimension)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_cross_lingual_paraphrase_beats_unrelated_text(
    embedder: SentenceTransformerEmbedder,
) -> None:
    query = embedder.embed_query("the app keeps crashing")
    german_paraphrase, unrelated = embedder.embed_documents(
        ["Die App stürzt ständig ab", "I love the cute owl mascot"]
    )

    assert float(query @ german_paraphrase) > float(query @ unrelated) + 0.2
