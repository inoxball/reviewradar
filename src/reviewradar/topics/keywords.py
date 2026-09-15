"""Distinctive keywords per topic with class-based TF-IDF (c-TF-IDF)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

GENERIC_STOP_WORDS = frozenset(
    {
        "amazing", "app", "application", "apps", "awesome", "bad", "best", "better", "didn",
        "doesn", "don", "excellent", "good", "great", "isn", "just", "like", "ll", "lot",
        "love", "make", "makes", "nice", "really", "super", "thank", "thanks", "thing",
        "things", "use", "used", "using", "ve", "want", "way",
        # filler that survives the English stop words
        "far", "going", "got", "hi", "know", "let", "ok", "okay", "think", "wow", "yes",
        # romanised Hindi function words, which language detection often calls English
        "aap", "acha", "achcha", "accha", "aur", "bahut", "bhi", "bohot", "hai", "hain", "ho",
        "hota", "hoti", "ka", "kar", "ke", "ki", "ko", "kuch", "kuchh", "liye", "maine",
        "mein", "mera", "mujhe", "na", "nahi", "nahin", "raha", "rahe", "rahi", "sab", "sakte",
        "se", "sikh", "sikhane", "sikhne", "yah", "yeh",
    }
)  # fmt: skip
"""Words that dominate app reviews without saying what a review is about."""

ENGLISH_TOKEN = r"(?u)\b[a-zA-Z]{2,}\b"
"""Keywords are English: Latin letters only, so text in another script cannot slip in."""
LABEL_KEYWORDS = 3
NO_TERMS_LABEL = "(no distinctive terms)"
MIN_STEM = 4


def keyword_label(keywords: Sequence[str]) -> str:
    """The label of a topic: its strongest keywords."""
    return " · ".join(keywords[:LABEL_KEYWORDS]) or NO_TERMS_LABEL


def topic_keywords(
    documents: Sequence[Sequence[str]],
    *,
    top_n: int = 6,
    min_reviews: int = 2,
    extra_stop_words: Iterable[str] = (),
) -> list[list[str]]:
    """The terms most specific to each topic, compared with the other topics.

    ``documents`` holds the English review texts of every topic. A term scores high when
    many of the topic's reviews use it and few reviews of other topics do (the c-TF-IDF of
    BERTopic). Terms are counted once per review, so one long or repetitive review cannot
    dominate a label, and a term needs ``min_reviews`` reviews in the topic to qualify.
    Selection skips bigrams that repeat a word ("sorry sorry") and terms sharing a word
    with a higher-scoring keyword in any simple inflection ("language", "languages",
    "learn", "learning"), so labels stay informative.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    if not documents:
        return []
    stop_words = sorted(
        ENGLISH_STOP_WORDS | GENERIC_STOP_WORDS | {word.lower() for word in extra_stop_words}
    )
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        token_pattern=ENGLISH_TOKEN,
        binary=True,
    )
    texts = [text for topic in documents for text in topic]
    try:
        per_review = vectorizer.fit_transform(texts).tocsr()
    except ValueError:  # every text is empty or made only of stop words
        return [[] for _ in documents]

    boundaries = np.cumsum([0, *(len(topic) for topic in documents)])
    counts = np.vstack(
        [
            np.asarray(per_review[start:end].sum(axis=0), dtype=np.float64).ravel()
            for start, end in pairwise(boundaries)
        ]
    )
    counts[counts < min_reviews] = 0.0
    term_frequency = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    average_topic_total = counts.sum(axis=1).mean()
    inverse_frequency = np.log(1.0 + average_topic_total / np.maximum(counts.sum(axis=0), 1.0))
    scores = term_frequency * inverse_frequency
    vocabulary = [str(term) for term in vectorizer.get_feature_names_out()]
    return [_diverse_top_terms(vocabulary, row, top_n) for row in scores]


def _diverse_top_terms(vocabulary: list[str], scores: NDArray[np.float64], top_n: int) -> list[str]:
    chosen: list[str] = []
    covered: set[str] = set()
    for index in np.argsort(scores, kind="stable")[::-1]:
        if len(chosen) == top_n or scores[index] <= 0:
            break
        tokens = vocabulary[index].split()
        stems = {_stem(token) for token in tokens}
        if len(stems) < len(tokens) or not stems.isdisjoint(covered):
            continue
        chosen.append(vocabulary[index])
        covered.update(stems)
    return chosen


def _stem(token: str) -> str:
    """A crude English stem, only to spot inflected duplicates among keywords.

    "languages" and "language", "learning" and "learned" and "learn", "changing" and
    "change" share a stem. Short words are left alone, so "need" never becomes "ne".
    """
    word = token.lower()
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 2 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > MIN_STEM:
        word = word[:-1]
    return word
