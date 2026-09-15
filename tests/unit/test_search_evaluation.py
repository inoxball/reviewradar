import math

import pytest
from pydantic import ValidationError

from reviewradar.domain import Store
from reviewradar.search.evaluation import JudgmentSet, ModeReport, evaluate_ranking
from reviewradar.search.types import SearchMode

GP = Store.GOOGLE_PLAY
GRADES = {(GP, "a"): 2, (GP, "b"): 1, (GP, "c"): 0, (GP, "d"): 2}


def test_perfect_ranking_scores_one() -> None:
    metrics = evaluate_ranking("q", [(GP, "a"), (GP, "d"), (GP, "b")], GRADES, k=3)

    assert metrics.ndcg == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.reciprocal_rank == 1.0
    assert metrics.judged_fraction == 1.0


def test_graded_gains_and_log_discount() -> None:
    metrics = evaluate_ranking("q", [(GP, "b"), (GP, "a")], GRADES, k=2)

    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    ideal = (2**2 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    assert metrics.ndcg == pytest.approx(dcg / ideal)


def test_unjudged_results_count_as_irrelevant_and_lower_judged_fraction() -> None:
    metrics = evaluate_ranking("q", [(GP, "unseen"), (GP, "c"), (GP, "a")], GRADES, k=3)

    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank == pytest.approx(1 / 3)
    assert metrics.judged_fraction == pytest.approx(2 / 3)


def test_short_result_lists_are_penalized_in_precision() -> None:
    assert evaluate_ranking("q", [(GP, "a")], GRADES, k=4).precision == pytest.approx(0.25)


def test_query_without_relevant_judgments_scores_zero() -> None:
    metrics = evaluate_ranking("q", [(GP, "c")], {(GP, "c"): 0}, k=1)

    assert (metrics.ndcg, metrics.reciprocal_rank) == (0.0, 0.0)


def test_mode_report_averages_queries() -> None:
    report = ModeReport(
        mode=SearchMode.HYBRID,
        k=2,
        queries=(
            evaluate_ranking("q1", [(GP, "a"), (GP, "d")], GRADES, k=2),
            evaluate_ranking("q2", [(GP, "c"), (GP, "x")], GRADES, k=2),
        ),
    )

    assert report.ndcg == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)
    assert report.judged_fraction == pytest.approx(0.75)


class TestJudgmentSet:
    def test_rejects_duplicate_judgments_within_a_query(self) -> None:
        judgment = {"store": "google_play", "id": "a", "grade": 1}

        with pytest.raises(ValidationError, match="more than once"):
            JudgmentSet.model_validate(
                {
                    "app": "duolingo",
                    "annotator": "test",
                    "queries": [{"id": "q", "text": "ads", "judgments": [judgment, judgment]}],
                }
            )

    def test_rejects_out_of_range_grades(self) -> None:
        with pytest.raises(ValidationError):
            JudgmentSet.model_validate(
                {
                    "app": "duolingo",
                    "annotator": "test",
                    "queries": [
                        {
                            "id": "q",
                            "text": "ads",
                            "judgments": [{"store": "app_store", "id": "a", "grade": 3}],
                        }
                    ],
                }
            )
