from reviewradar.topics.keywords import topic_keywords

ENERGY = [
    "energy runs out after one lesson",
    "no energy left for the next lesson",
    "energy refill is slow",
]
ADS = ["too many ads after every lesson", "ads everywhere in each lesson", "watch ads again"]


def test_ranks_terms_specific_to_each_topic_above_shared_ones() -> None:
    energy, ads = topic_keywords([ENERGY, ADS], top_n=3)

    assert energy[0] == "energy"
    assert ads[0] == "ads"


def test_counts_a_term_once_per_review_and_requires_support() -> None:
    (keywords,) = topic_keywords([["refund refund refund refund", "streak lost", "streak gone"]])

    assert keywords == ["streak"]


def test_skips_repeated_word_bigrams_and_covered_terms() -> None:
    (keywords,) = topic_keywords([["sorry sorry sorry", "sorry sorry again"]])

    assert keywords == ["sorry"]


def test_removes_generic_and_extra_stop_words() -> None:
    (keywords,) = topic_keywords(
        [["the app duolingo is great app", "duolingo app streak lost"]],
        min_reviews=1,
        extra_stop_words={"duolingo"},
    )

    words = {word for term in keywords for word in term.split()}
    assert "app" not in words
    assert "duolingo" not in words
    assert "streak" in words


def test_handles_empty_input() -> None:
    assert topic_keywords([]) == []
    assert topic_keywords([["the", "a"], ["and"]]) == [[], []]


def test_plural_forms_do_not_repeat_a_keyword() -> None:
    (keywords,) = topic_keywords(
        [["languages are hard", "language course", "languages again", "the language icons"]],
        min_reviews=1,
    )

    singles = [term for term in keywords if " " not in term]
    assert ("language" in singles) != ("languages" in singles)


def test_praise_words_are_not_keywords() -> None:
    (keywords,) = topic_keywords(
        [["good app love it super", "great good love the streak", "best streak ever"]],
        min_reviews=1,
    )

    words = {word for term in keywords for word in term.split()}
    assert words.isdisjoint({"good", "love", "super", "great", "best"})
    assert "streak" in words


def test_ads_and_ad_count_as_one_keyword() -> None:
    (keywords,) = topic_keywords([["too many ads", "an ad every lesson", "ads again", "ad break"]])

    assert ("ads" in keywords) != ("ad" in keywords)


def test_romanised_hindi_function_words_are_not_keywords() -> None:
    (keywords,) = topic_keywords(
        [["yeh app bahut acha hai streak", "mujhe streak pasand hai", "aap ki streak"]]
    )

    assert keywords[0] == "streak"
    assert {"hai", "aap", "bahut", "ki"}.isdisjoint(keywords)


def test_keywords_use_latin_letters_only() -> None:
    arabic = "".join(map(chr, (0x062C, 0x0645, 0x064A, 0x0644)))
    (keywords,) = topic_keywords([[f"{arabic} streak", f"{arabic} streak lost", "streak"]])

    assert keywords == ["streak"]


def test_inflections_of_a_word_take_one_keyword() -> None:
    (keywords,) = topic_keywords(
        [["learning spanish", "learned spanish fast", "learn spanish daily", "learning again"]],
        min_reviews=1,
    )

    learn_forms = [term for term in keywords if term in {"learn", "learning", "learned"}]
    assert len(learn_forms) == 1
    assert "spanish" in keywords
