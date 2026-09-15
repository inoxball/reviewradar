import numpy as np
import pytest

from reviewradar.topics.clustering import KMeansClusterer


def blobs(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Three tight groups around orthogonal directions in 8 dimensions."""
    generator = np.random.default_rng(seed)
    centers = np.eye(8, dtype=np.float32)[:3] * 5
    points = np.concatenate([center + generator.normal(0, 0.1, (20, 8)) for center in centers])
    truth = np.repeat(np.arange(3), 20)
    return points.astype(np.float32), truth


def test_separates_well_defined_groups() -> None:
    points, truth = blobs()

    clustering = KMeansClusterer(3).fit(points)

    for group in range(3):
        assert len(set(clustering.labels[truth == group].tolist())) == 1
    assert len(set(clustering.labels.tolist())) == 3
    assert float(clustering.distances.max()) < 0.05


def test_is_deterministic_for_a_seed() -> None:
    points, _ = blobs()

    first = KMeansClusterer(3, seed=7).fit(points)
    second = KMeansClusterer(3, seed=7).fit(points)

    np.testing.assert_array_equal(first.labels, second.labels)


def test_uses_at_most_one_cluster_per_vector() -> None:
    points, _ = blobs()

    clustering = KMeansClusterer(50).fit(points[:4])

    assert len(clustering.labels) == 4


def test_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        KMeansClusterer(0)
    with pytest.raises(ValueError, match="empty"):
        KMeansClusterer(2).fit(np.empty((0, 8), dtype=np.float32))
