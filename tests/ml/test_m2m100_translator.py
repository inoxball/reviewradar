"""The real M2M100 translation model from the local Hugging Face cache (``pytest -m ml``)."""

import pytest

from reviewradar.config import TranslationSettings
from reviewradar.translation.translator import M2M100Translator
from reviewradar.translation.wiring import create_translator

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def translator() -> M2M100Translator:
    return create_translator(TranslationSettings())


def test_supports_the_corpus_languages_but_not_non_linguistic_content(
    translator: M2M100Translator,
) -> None:
    assert all(translator.supports(language) for language in ("pt", "es", "de", "ja", "ru"))
    assert not translator.supports("zxx")


def test_translates_a_batch_into_english_in_order(translator: M2M100Translator) -> None:
    texts = [
        "Las energías se acaban muy rápido",
        "Zu viel Werbung nach jeder Lektion",
    ]

    spanish = translator.translate(texts[:1], source_language="es", target_language="en")
    german = translator.translate(texts[1:], source_language="de", target_language="en")

    assert "energ" in spanish[0].lower()
    assert "advertis" in german[0].lower() or "ads" in german[0].lower()
