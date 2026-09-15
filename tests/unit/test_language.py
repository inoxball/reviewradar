import pytest

from reviewradar.domain import LanguageSource
from reviewradar.enrichment.language import LangidDetector, LanguageGuess, resolve_language
from tests.fakes import FixedLanguageDetector

LONG_TEXT = "The lessons are great but the app keeps crashing"


def resolve(
    detector: FixedLanguageDetector,
    *,
    text: str = LONG_TEXT,
    hint: str | None = None,
    market_default: str | None = None,
) -> LanguageGuess:
    return resolve_language(
        text,
        detector,
        hint=hint,
        market_default=market_default,
        min_confidence=0.9,
        min_corroborated_confidence=0.5,
        min_chars=20,
    )


class TestResolveLanguage:
    def test_confident_detection_wins_over_all_other_evidence(self) -> None:
        guess = resolve(FixedLanguageDetector("en", 0.97), hint="de", market_default="de")

        assert guess == LanguageGuess("en", 0.97, "en", LanguageSource.DETECTED)

    @pytest.mark.parametrize("evidence", [{"hint": "en"}, {"market_default": "en"}])
    def test_weaker_detection_is_accepted_when_corroborated(self, evidence: dict[str, str]) -> None:
        guess = resolve(FixedLanguageDetector("en", 0.6), **evidence)

        assert guess == LanguageGuess("en", 0.6, "en", LanguageSource.DETECTED)

    def test_uncorroborated_weak_detection_defers_to_the_store_hint(self) -> None:
        guess = resolve(FixedLanguageDetector("en", 0.6), hint="de")

        assert guess == LanguageGuess("de", 0.6, "en", LanguageSource.STORE_HINT)

    def test_store_hint_outranks_market_default(self) -> None:
        guess = resolve(FixedLanguageDetector("zu", 0.2), hint="pt", market_default="en")

        assert guess == LanguageGuess("pt", 0.2, "zu", LanguageSource.STORE_HINT)

    def test_market_default_is_the_last_resort_before_unknown(self) -> None:
        guess = resolve(FixedLanguageDetector("zu", 0.2), market_default="en")

        assert guess == LanguageGuess("en", 0.2, "zu", LanguageSource.MARKET_DEFAULT)

    def test_short_text_skips_detection(self) -> None:
        detector = FixedLanguageDetector("en", 0.99)

        guess = resolve(detector, text="ok", hint="pt")

        assert guess == LanguageGuess("pt", None, None, LanguageSource.STORE_HINT)
        assert detector.texts == []

    def test_no_evidence_means_unknown(self) -> None:
        guess = resolve(FixedLanguageDetector("zu", 0.2))

        assert guess == LanguageGuess(None, 0.2, "zu", LanguageSource.UNKNOWN)


@pytest.fixture(scope="module")
def detector() -> LangidDetector:
    return LangidDetector()


class TestLangidDetector:
    @pytest.mark.parametrize(
        ("text", "language"),
        [
            ("The lessons are great but the app keeps crashing after the update", "en"),
            ("Die Lektionen sind super, aber die App stürzt nach dem Update ständig ab", "de"),
            ("As lições são ótimas, mas o aplicativo trava depois da atualização", "pt"),
            ("レッスンは素晴らしいですが、アップデート後にアプリが頻繁に落ちます", "ja"),
        ],
    )
    def test_identifies_common_review_languages(
        self, detector: LangidDetector, text: str, language: str
    ) -> None:
        detected, confidence = detector.detect(text)

        assert detected == language
        assert confidence > 0.9
