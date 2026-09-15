import pytest

from reviewradar.search.fusion import reciprocal_rank_fusion


def test_documents_ranked_well_by_both_lists_win() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)

    assert [hit.review_id for hit in fused] == [1, 3, 2, 4]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_single_list_keeps_its_order() -> None:
    assert [hit.review_id for hit in reciprocal_rank_fusion([[7, 5, 9]])] == [7, 5, 9]


def test_ties_break_on_best_rank_then_id() -> None:
    fused = reciprocal_rank_fusion([[10, 20], [20, 10]], k=60)

    assert [hit.review_id for hit in fused] == [10, 20]


def test_duplicates_within_a_list_count_once() -> None:
    fused = reciprocal_rank_fusion([[1, 1, 2]], k=60)

    assert [hit.review_id for hit in fused] == [1, 2]
    assert fused[1].score == pytest.approx(1 / 62)


def test_empty_rankings_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([[1]], k=0)
