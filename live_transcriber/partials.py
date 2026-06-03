from __future__ import annotations

import re

from live_transcriber.text_cleanup import cleanup_transcript_text


class PartialTextTracker:
    """Emit only safe append-only suffixes from provisional transcripts."""

    def __init__(self, rewrite_tolerance: float = 0.7, min_new_words: int = 1) -> None:
        self.rewrite_tolerance = float(rewrite_tolerance)
        self.min_new_words = int(min_new_words)
        self._last_text_by_utterance: dict[int, str] = {}

    def update(self, utterance_id: int, text: str) -> str | None:
        cleaned = cleanup_transcript_text(text)
        if not cleaned:
            return None

        previous = self._last_text_by_utterance.get(utterance_id)
        self._last_text_by_utterance = {utterance_id: cleaned}
        if previous is None:
            return cleaned

        previous_tokens = _tokens(previous)
        current_tokens = _tokens(cleaned)
        if not current_tokens:
            return None

        common = _common_prefix_length(previous_tokens, current_tokens)
        if common >= len(current_tokens):
            return None

        if previous_tokens and common < len(previous_tokens):
            coverage = float(common) / float(len(previous_tokens))
            if coverage < self.rewrite_tolerance:
                return None

        suffix_words = [token.word for token in current_tokens[common:]]
        if len(suffix_words) < self.min_new_words:
            return None

        return " ".join(suffix_words).strip()

    def finish(self, utterance_id: int) -> None:
        self._last_text_by_utterance.pop(utterance_id, None)


class _Token:
    def __init__(self, word: str) -> None:
        self.word = word
        self.normalized = _normalize(word)


def _tokens(text: str) -> list[_Token]:
    return [_Token(match.group(0)) for match in re.finditer(r"\S+", text) if _normalize(match.group(0))]


def _normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\wäöüß]+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _common_prefix_length(left: list[_Token], right: list[_Token]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token.normalized != right_token.normalized:
            break
        count += 1
    return count
