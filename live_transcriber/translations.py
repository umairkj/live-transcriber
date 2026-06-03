from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Literal


DEFAULT_TRANSLATION_TARGET_LANGUAGE = "en"
TranslationKind = Literal["noun", "verb", "adjective", "adverb", "phrase"]


@dataclass(frozen=True)
class SelectiveTranslation:
    source: str
    translation: str
    kind: TranslationKind

    def to_record(self) -> dict[str, str]:
        return asdict(self)


_LEXICON: dict[str, tuple[str, TranslationKind]] = {
    "abend": ("evening", "noun"),
    "angefangen": ("started", "verb"),
    "arbeit": ("work", "noun"),
    "arbeitet": ("works", "verb"),
    "aufgeklappt": ("opened up", "verb"),
    "bereit": ("ready", "adjective"),
    "bezahlen": ("pay", "verb"),
    "kommt": ("comes", "verb"),
    "diskutieren": ("discuss", "verb"),
    "fruhsport": ("morning exercise", "noun"),
    "fruhstucke": ("eat breakfast", "verb"),
    "fruhstucken": ("eat breakfast", "verb"),
    "gearbeitet": ("worked", "verb"),
    "gehe": ("go", "verb"),
    "gehen": ("go", "verb"),
    "geld": ("money", "noun"),
    "genau": ("exactly", "adverb"),
    "gerade": ("just now", "adverb"),
    "gesund": ("healthy", "adjective"),
    "glas": ("glass", "noun"),
    "glucklich": ("happy", "adjective"),
    "grundlage": ("foundation", "noun"),
    "handy": ("phone", "noun"),
    "leben": ("life", "noun"),
    "laptop": ("laptop", "noun"),
    "langsam": ("slowly", "adverb"),
    "mache": ("do", "verb"),
    "machen": ("do", "verb"),
    "meditiere": ("meditate", "verb"),
    "meditieren": ("meditate", "verb"),
    "morgen": ("morning", "noun"),
    "morgenroutine": ("morning routine", "noun"),
    "putze": ("clean", "verb"),
    "putzen": ("clean", "verb"),
    "rechnung": ("bill", "noun"),
    "routine": ("routine", "noun"),
    "sonne": ("sun", "noun"),
    "sport": ("exercise", "noun"),
    "starten": ("start", "verb"),
    "tag": ("day", "noun"),
    "tatsaechlich": ("actually", "adverb"),
    "tatsachlich": ("actually", "adverb"),
    "trinke": ("drink", "verb"),
    "trinken": ("drink", "verb"),
    "vorhange": ("curtains", "noun"),
    "wach": ("awake", "adjective"),
    "warm": ("warm", "adjective"),
    "warmes": ("warm", "adjective"),
    "waschen": ("wash", "verb"),
    "wasser": ("water", "noun"),
    "wichtig": ("important", "adjective"),
    "wohlstand": ("prosperity", "noun"),
    "zaehne": ("teeth", "noun"),
    "zahne": ("teeth", "noun"),
    "zahneputzen": ("brush teeth", "verb"),
    "zeit": ("time", "noun"),
}


def build_selective_translations(
    text: str,
    target_language: str = DEFAULT_TRANSLATION_TARGET_LANGUAGE,
    max_items: int = 18,
) -> list[dict[str, str]]:
    if target_language != "en":
        return []

    hints: list[SelectiveTranslation] = []
    seen: set[str] = set()
    for word in _words(text):
        key = _lookup_key(word)
        if key in seen:
            continue

        entry = _LEXICON.get(key)
        if entry is None:
            continue

        translation, kind = entry
        hints.append(SelectiveTranslation(source=word, translation=translation, kind=kind))
        seen.add(key)
        if len(hints) >= max_items:
            break

    return [hint.to_record() for hint in hints]


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"\b\w+\b", text, flags=re.UNICODE) if any(char.isalpha() for char in word)]


def _lookup_key(word: str) -> str:
    word = word.casefold().replace("\u00df", "ss")
    normalized = unicodedata.normalize("NFKD", word)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped
