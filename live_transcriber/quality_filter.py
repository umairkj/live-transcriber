from __future__ import annotations

import re

from live_transcriber.sentence_utils import normalize_text_for_compare


class SentenceQualityFilter:
    """Reject obvious broken fragments before they become final transcript."""

    SHORT_ALLOWLIST = {
        "ja.",
        "nein.",
        "genau.",
        "okay.",
        "hallo.",
        "danke.",
        "bitte.",
        "stimmt.",
        "gut.",
        "tschüss.",
        "ciao.",
        "servus.",
        "moin.",
    }
    BROKEN_ONE_WORDS = {"uhr.", "uf.", "ähm.", "hm.", "gearbeitet."}
    BROKEN_PREFIXES = ("uf.", "sam wach", "ühsport")

    def __init__(
        self,
        min_words: int = 3,
        min_chars: int = 12,
        allow_short_utterances: bool = True,
    ) -> None:
        self.min_words = int(min_words)
        self.min_chars = int(min_chars)
        self.allow_short_utterances = bool(allow_short_utterances)

    def is_good_final_sentence(self, sentence: str) -> bool:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            return False
        if self.is_short_allowlisted(sentence):
            return True
        if self.looks_like_broken_fragment(sentence):
            return False
        if len(sentence) < self.min_chars:
            return False
        if self.word_count(sentence) < self.min_words:
            return False
        if self.has_repeated_ngram(sentence):
            return False
        return True

    def word_count(self, sentence: str) -> int:
        return len(re.findall(r"\b[\wäöüÄÖÜß]+\b", sentence, flags=re.UNICODE))

    def is_short_allowlisted(self, sentence: str) -> bool:
        if not self.allow_short_utterances:
            return False
        return normalize_text_for_compare(sentence) in self.SHORT_ALLOWLIST

    def has_repeated_ngram(self, sentence: str, n: int = 3) -> bool:
        words = re.findall(r"\b[\wäöüÄÖÜß]+\b", normalize_text_for_compare(sentence), flags=re.UNICODE)
        if len(words) < n * 2:
            return False

        grams = [" ".join(words[index : index + n]) for index in range(len(words) - n + 1)]
        for index, gram in enumerate(grams[:-1]):
            if gram in grams[index + 1 :]:
                return True
        return False

    def looks_like_broken_fragment(self, sentence: str) -> bool:
        normalized = normalize_text_for_compare(sentence)
        if normalized in self.BROKEN_ONE_WORDS:
            return True
        if any(normalized.startswith(prefix) for prefix in self.BROKEN_PREFIXES):
            return True
        if self.word_count(sentence) <= 1 and not self.is_short_allowlisted(sentence):
            return True
        if len(sentence) < self.min_chars and sentence[:1].islower():
            return True
        return False
