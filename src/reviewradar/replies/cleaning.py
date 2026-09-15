"""Turns published developer replies into brand-neutral, anonymous training targets.

A reply written by one app's support team mentions its brand, its support address and
often the reviewer's name. Training on it verbatim would teach a model to greet strangers
by name and send Duolingo users to Babbel's inbox. Cleaning replaces the brand and contact
with placeholders filled per app at render time, and removes names, sign-offs and emojis.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

APP_NAME = "{app_name}"
SUPPORT_CONTACT = "{support_contact}"
PLACEHOLDERS = frozenset({APP_NAME, SUPPORT_CONTACT})
MIN_REPLY_WORDS = 5
CHARACTERS_PER_WORD_WITHOUT_SPACES = 3
"""Rough word length in scripts written without spaces (Japanese, Chinese, Korean)."""
TRUNCATION_LENGTH = 340
"""Google Play cuts replies at 350 characters; replies this long ending in '...' were cut."""

_EMAIL_OR_URL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
# Patterns spell accented letters as regex escapes (\u00e1 is "a" with an acute accent).
_NAME_WORD = r"[A-Z\u00c0-\u00d6\u00d8-\u00dd][\w'\u2019-]+"
_NAME = rf"{_NAME_WORD}(?:[ \t]+{_NAME_WORD})?"
_GREETINGS = r"hi|hello|hey|dear|hallo|liebe|lieber|hola|ol\u00e1|ola|oi|bonjour|salut|ciao|merhaba"
_NOT_A_NAME = rf"(?!(?i:{_GREETINGS}|there|all|team)\b)"
_GREETING_WITH_NAME = re.compile(
    rf"^(?P<greeting>(?i:{_GREETINGS})),?[ \t]+{_NOT_A_NAME}{_NAME}[ \t]*(?P<punct>[,!:.])"
)
_THANKS_OBJECTS = (
    r"feedback|review|message|comment|words|rese\u00f1a|comentario|mensaje|opini\u00f3n|"
    r"coment\u00e1rio|avalia\u00e7\u00e3o|mensagem|opini\u00e3o|bewertung|nachricht|rezension"
)
_NAME_AFTER_THANKS = re.compile(
    rf"(?P<object>\b(?i:{_THANKS_OBJECTS}))(?:,[ \t]*|[ \t]+){_NOT_A_NAME}{_NAME}(?=[.!,])"
)
_LEADING_NAME = re.compile(
    rf"^{_NOT_A_NAME}{_NAME},[ \t]+(?=(?i:thank|thanks|danke|gracias|obrigad|merci))"
)
_SIGN_OFF_WORDS = re.compile(
    r"\b(?i:team|time|equipe|equipa|equipo|\u00e9quipe|gr\u00fc\u00dfe|saludos|abra\u00e7os|cheers|regards)\b"
)
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
_NO_SPACE_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.!?:;])")
_FINAL_FRAGMENT = re.compile(r"(?<=[.!?\u2026)])\s+(?P<fragment>[^.!?\u2026]{1,60})$")
_CLOSING = (
    r"(?:(?:best|kind|warm|many|with|beste|liebe|freundliche|viele|mit\s+freundlichen)\s+)?"
    r"(?:regards|wishes|thanks|best|gr\u00fc\u00dfe|gru\u00df|gr\u00fc\u00dfen|lg|saludos|"
    r"sauda\u00e7\u00f5es|atenciosamente|cumprimentos|abra\u00e7os|cheers|take\s+care)"
)
_SIGNATURE = (
    r"(?:(?:dein|euer|ihr|your|the|das|seu|sua|equipe|equipa)\s*)?\{app_name\}"
    r"(?:[\s-]*(?:team|time|equipe|support))?|kundenservice|support\s+team"
)
_SIGN_OFF = re.compile(rf"(?i:(?:{_CLOSING})?[\s,:;)!\-\u2013\u2014]*(?:{_SIGNATURE})?[\s,:;)!]*)")


def clean_developer_reply(reply: str, *, brand_names: Iterable[str]) -> str | None:
    """A brand-neutral, anonymous version of a published reply, or None when unusable.

    Unusable means truncated by the store or too short to teach anything ("🧡").
    Name removal is heuristic: greetings ("Hi Veronika,"), thanks ("feedback, Irene.")
    and a leading name ("Omar, thank you"); names elsewhere in a reply are not detected.
    """
    text = reply.strip()
    if len(text) >= TRUNCATION_LENGTH and text.endswith(("...", "…")):
        return None
    text = _EMAIL_OR_URL.sub(SUPPORT_CONTACT, text)
    for brand in sorted({name for name in brand_names if name}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w{{]){re.escape(brand)}(?!\w)", APP_NAME, text, flags=re.IGNORECASE)
    text = _strip_sign_off(text)
    text = _strip_names(text)
    text = _EMOJI.sub("", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", " ".join(text.split()))
    text = _strip_final_sign_off(text)
    if not has_enough_words(text, MIN_REPLY_WORDS):
        return None
    return text[0].upper() + text[1:]


def has_enough_words(text: str, minimum: int) -> bool:
    """Whether ``text`` has at least ``minimum`` words, also in scripts without spaces."""
    if len(text.split()) >= minimum:
        return True
    return len(_NO_SPACE_SCRIPT.findall(text)) >= minimum * CHARACTERS_PER_WORD_WITHOUT_SPACES


def _strip_sign_off(text: str) -> str:
    """Drop trailing short lines such as 'Mondly Team' or 'Seu Time Babbel'."""
    lines = text.splitlines()
    while lines and (
        not lines[-1].strip()
        or (
            len(lines[-1].split()) <= 4
            and (APP_NAME in lines[-1] or _SIGN_OFF_WORDS.search(lines[-1]))
        )
    ):
        lines.pop()
    return "\n".join(lines)


def _strip_final_sign_off(text: str) -> str:
    """Drop a closing written on the last line itself: 'Saudações', 'Best, {app_name} team'.

    Only a fragment made entirely of closing words and a signature is removed, so a final
    sentence that merely lacks punctuation ("Thanks for your patience with us") stays. The
    first adapter learned such closings from published replies (notes §13.3).
    """
    while (match := _FINAL_FRAGMENT.search(text)) is not None:
        if not _SIGN_OFF.fullmatch(match["fragment"]):
            break
        text = text[: match.start()].rstrip()
    return text


def _strip_names(text: str) -> str:
    text = _GREETING_WITH_NAME.sub(
        lambda match: match["greeting"] + ("!" if match["punct"] == "!" else ","), text
    )
    text = _NAME_AFTER_THANKS.sub(lambda match: match["object"], text)
    return _LEADING_NAME.sub("", text)
