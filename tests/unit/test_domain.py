from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tests.support import make_review


def test_content_hash_changes_when_review_is_edited() -> None:
    review = make_review("r1")

    assert replace(review, body="Edited text").content_hash != review.content_hash
    assert replace(review, rating=1).content_hash != review.content_hash
    assert replace(review, developer_reply="Thanks!").content_hash != review.content_hash


def test_content_hash_ignores_vote_churn() -> None:
    review = make_review("r1")

    assert replace(review, helpful_count=42).content_hash == review.content_hash


@pytest.mark.parametrize(
    "reviewed_at",
    [
        datetime(2026, 9, 1, 12, 0),
        datetime(2026, 9, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))),
    ],
    ids=["naive", "non-utc"],
)
def test_rejects_timestamps_that_are_not_utc(reviewed_at: datetime) -> None:
    with pytest.raises(ValueError, match="UTC"):
        replace(make_review("r1"), reviewed_at=reviewed_at)


@pytest.mark.parametrize("rating", [0, 6])
def test_rejects_out_of_range_ratings(rating: int) -> None:
    with pytest.raises(ValueError, match="rating"):
        make_review("r1", rating=rating)
