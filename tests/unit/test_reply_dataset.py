import json
from pathlib import Path

import pytest

from reviewradar.domain import Store
from reviewradar.replies.dataset import (
    ReplySource,
    ReviewRow,
    Split,
    WrittenReply,
    attach_written,
    curate_published,
    load_written_replies,
    split_of,
)
from reviewradar.replies.prompting import ReplyRequest, build_messages, render_reply


def row(external_id: str, reply: str, *, language: str = "en") -> ReviewRow:
    return ReviewRow("babbel", Store.GOOGLE_PLAY, external_id, 2, language, "review text", reply)


def english(_: str) -> str:
    return "en"


def test_split_is_stable_and_roughly_proportional() -> None:
    splits = [split_of(f"babbel:google_play:{index}", 0.2) for index in range(2000)]

    assert splits == [split_of(f"babbel:google_play:{index}", 0.2) for index in range(2000)]
    assert 0.17 < splits.count(Split.EVAL) / len(splits) < 0.23


def test_curation_cleans_deduplicates_and_caps_templates() -> None:
    from reviewradar.replies.dataset import MAX_SHARED_OPENINGS

    template = "Thank you for your feedback, we are sorry to hear about this issue. "
    templated = MAX_SHARED_OPENINGS + 2
    rows = [
        row("1", "Hi Anna, thanks for telling us about the Babbel audio issue."),
        row("2", "Hi Ben, thanks for telling us about the Babbel audio issue."),
        *(row(f"t{index}", f"{template}Detail number {index}.") for index in range(templated)),
        row("de", "Danke für dein Feedback zu den Übungen im Kurs.", language="de"),
        row("short", "🧡"),
    ]

    examples, dropped = curate_published(rows, brand_names=["Babbel"], detect_language=english)

    assert [example.external_id for example in examples] == [
        "1",
        *(f"t{index}" for index in range(MAX_SHARED_OPENINGS)),
    ]
    assert examples[0].reply == "Hi, thanks for telling us about the {app_name} audio issue."
    assert examples[0].source is ReplySource.PUBLISHED
    assert dropped == {
        "duplicate": 1,
        "template opening": 2,
        "not in the review's language": 1,
        "truncated or too short": 1,
    }


def test_written_replies_attach_to_stored_reviews(tmp_path: Path) -> None:
    (tmp_path / "part-01.jsonl").write_text(
        json.dumps(
            {"app": "duolingo", "store": "app_store", "id": "42", "reply": "Thanks, {app_name}!"}
        )
        + "\n\n"
        + json.dumps({"app": "duolingo", "store": "app_store", "id": "404", "reply": "Hello there"})
        + "\n",
        encoding="utf-8",
    )
    stored = ReviewRow("duolingo", Store.APP_STORE, "42", 5, "en", "Great app", None)

    written = load_written_replies(tmp_path)
    examples, dropped = attach_written(written, {("duolingo", Store.APP_STORE, "42"): stored})

    assert written[0] == WrittenReply("duolingo", Store.APP_STORE, "42", "Thanks, {app_name}!")
    assert [example.key for example in examples] == ["duolingo:app_store:42"]
    assert examples[0].source is ReplySource.WRITTEN
    assert dropped == {"written reply without a stored review": 1}


def test_written_replies_reject_unknown_placeholders(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text(
        json.dumps({"app": "duolingo", "store": "app_store", "id": "1", "reply": "Hi {name}"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown placeholders"):
        load_written_replies(tmp_path)


def test_prompt_carries_guidelines_and_review_and_rendering_fills_placeholders() -> None:
    request = ReplyRequest(Store.GOOGLE_PLAY, 1, "de", "Die App stürzt ab.")

    system, user = build_messages(request, "GUIDELINES")

    assert system == {"role": "system", "content": "GUIDELINES"}
    assert "Store: Google Play\nRating: 1/5\nLanguage: de" in user["content"]
    assert "Die App stürzt ab." in user["content"]
    assert (
        render_reply(
            "Schreib an {support_contact}, {app_name}  hilft.",
            app_name="Duolingo",
            support_contact="support.duolingo.com",
        )
        == "Schreib an support.duolingo.com, Duolingo hilft."
    )
    assert render_reply("Contact {support_contact}.", app_name="Mondly", support_contact=None) == (
        "Contact Mondly support."
    )
