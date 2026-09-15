import pytest

from reviewradar.replies.cleaning import clean_developer_reply

BRANDS = ["Babbel", "Mondly", "Memrise"]


def clean(reply: str) -> str | None:
    return clean_developer_reply(reply, brand_names=BRANDS)


def test_replaces_contacts_and_brand_names_with_placeholders() -> None:
    reply = (
        "Bitte schreib über die E-Mail-Adresse deines Babbel-Kontos an support@de.babbel.com "
        "oder besuche https://babbel.com/help, dann helfen wir gerne."
    )

    assert clean(reply) == (
        "Bitte schreib über die E-Mail-Adresse deines {app_name}-Kontos an {support_contact} "
        "oder besuche {support_contact} dann helfen wir gerne."
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "Hi Veronika, your learning experience sounds amazing. Thank you!",
            "Hi, your learning experience sounds amazing. Thank you!",
        ),
        (
            "Hallo Tobias! Wir freuen uns sehr über dein Feedback.",
            "Hallo! Wir freuen uns sehr über dein Feedback.",
        ),
        (
            "Hola, Lyox:\n\nSe pregunta por la edad para métricas internas.",
            "Hola, Se pregunta por la edad para métricas internas.",
        ),
        (
            "Thank you for your feedback, Irene. Glad you enjoy the lessons.",
            "Thank you for your feedback. Glad you enjoy the lessons.",
        ),
        (
            "Gracias por tu comentario, Fernando Pacay. Nos alegra saber que te gusta.",
            "Gracias por tu comentario. Nos alegra saber que te gusta.",
        ),
        (
            "Thank you for your feedback Sadiq. We are glad it helps.",
            "Thank you for your feedback. We are glad it helps.",
        ),
        (
            "Omar, thank you! We are continuously improving the app.",
            "Thank you! We are continuously improving the app.",
        ),
    ],
    ids=[
        "greeting",
        "greeting-exclamation",
        "greeting-comma-colon",
        "thanks",
        "full-name",
        "no-comma",
        "leading",
    ],
)
def test_removes_reviewer_names(reply: str, expected: str) -> None:
    assert clean(reply) == expected


def test_keeps_greetings_without_names() -> None:
    reply = "Hi there, could you please reach out to us at support@babbel.com?"

    assert clean(reply) == "Hi there, could you please reach out to us at {support_contact}?"


def test_removes_sign_offs_and_emojis() -> None:
    reply = (
        "Olá, obrigada pela sua mensagem. 🧡️ Vamos compartilhar com a equipe.\nSeu Time Babbel\n"
    )

    assert clean(reply) == "Olá, obrigada pela sua mensagem. Vamos compartilhar com a equipe."
    assert clean("Danke für dein Feedback zu den Übungen.\nMondly Team") == (
        "Danke für dein Feedback zu den Übungen."
    )


@pytest.mark.parametrize(
    "reply",
    ["🧡️", "Thanks, Anna!", "La primera lección es gratuita, " + "x" * 300 + " suscripc..."],
    ids=["emoji-only", "too-short", "truncated"],
)
def test_rejects_unusable_replies(reply: str) -> None:
    assert clean(reply) is None


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "Obrigado pela sua avaliação, ficamos felizes que goste. Saudações",
            "Obrigado pela sua avaliação, ficamos felizes que goste.",
        ),
        (
            "We shared your idea with the team! Best, Babbel team",
            "We shared your idea with the team!",
        ),
        (
            "Danke für dein Feedback zu den Übungen! Liebe Grüße :)",
            "Danke für dein Feedback zu den Übungen!",
        ),
        (
            "Bitte schreib uns mit Details an support@babbel.com. "
            "Freundliche Grüße, dein Babbel-Team",
            "Bitte schreib uns mit Details an {support_contact}.",
        ),
    ],
    ids=["portuguese", "english-signature", "german-emoticon", "german-signature"],
)
def test_removes_closings_after_the_last_sentence(reply: str, expected: str) -> None:
    assert clean(reply) == expected


def test_keeps_a_final_sentence_without_punctuation() -> None:
    reply = "We are glad you like the new lessons. Thanks for your patience with us"

    assert clean(reply) == reply
