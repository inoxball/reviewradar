import pytest

from reviewradar.enrichment.text import (
    compose_review_text,
    prepare_model_text,
    redact_personal_data,
)


class TestComposeReviewText:
    def test_joins_title_and_body_with_normalized_whitespace(self) -> None:
        assert compose_review_text("  Streak   lost ", "Crashed\n\nmid-lesson") == (
            "Streak lost\nCrashed mid-lesson"
        )

    def test_drops_parts_without_words(self) -> None:
        assert compose_review_text(".", "Love the speaking drills") == "Love the speaking drills"

    def test_keeps_non_latin_scripts(self) -> None:
        assert compose_review_text(None, "アプリが落ちる") == "アプリが落ちる"

    def test_emoji_only_review_becomes_empty(self) -> None:
        assert prepare_model_text(None, "👍👍") == ""


class TestRedactPersonalData:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("mail me at jane.doe+app@example.co.uk", "mail me at [email]"),
            ("see https://example.com/help?id=42 now", "see [url] now"),
            ("call +1 (415) 555-0134 please", "call [phone] please"),
            ("my number 0049 151 23456789", "my number [phone]"),
        ],
    )
    def test_replaces_personal_data(self, text: str, expected: str) -> None:
        assert redact_personal_data(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "broken since 2026-09-12",
            "version 7.139.0 crashes",
            "lost my 400 day streak",
        ],
    )
    def test_leaves_dates_versions_and_counts_alone(self, text: str) -> None:
        assert redact_personal_data(text) == text
