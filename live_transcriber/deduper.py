from __future__ import annotations

from collections import deque
from difflib import SequenceMatcher
import re

from live_transcriber.sentence_utils import (
    SENTENCE_ENDINGS,
    clean_whisper_artifacts,
    normalize_text_for_compare,
    split_complete_sentences,
)


class FuzzyTranscriptDeduper:
    """Return only new text from repeated rolling-window Whisper hypotheses."""

    def __init__(
        self,
        min_overlap_chars: int = 20,
        min_new_chars: int = 8,
        max_compare_chars: int = 500,
    ) -> None:
        self.committed_text = ""
        self.min_overlap_chars = int(min_overlap_chars)
        self.min_new_chars = int(min_new_chars)
        self.max_compare_chars = int(max_compare_chars)

    def process(self, new_text: str) -> str:
        new_text = clean_whisper_artifacts(new_text)
        if not new_text:
            return ""

        if not self.committed_text:
            return self._commit_if_ready(new_text)

        committed_norm = normalize_text_for_compare(self.committed_text)
        new_norm = normalize_text_for_compare(new_text)
        if not new_norm or new_norm in committed_norm:
            return ""

        unique_text = self._find_unique_suffix(new_text, committed_norm, new_norm)
        return self._commit_if_ready(unique_text)

    def reset(self) -> None:
        self.committed_text = ""

    def _commit_if_ready(self, unique_text: str) -> str:
        unique_text = clean_whisper_artifacts(unique_text)
        if not unique_text:
            return ""

        if len(unique_text) < self.min_new_chars and not unique_text.endswith(tuple(SENTENCE_ENDINGS)):
            return ""

        self._append_committed(unique_text)
        return unique_text

    def _append_committed(self, text: str) -> None:
        text = clean_whisper_artifacts(text)
        if not text:
            return

        if not self.committed_text:
            self.committed_text = text
        elif text[:1] in SENTENCE_ENDINGS + ",;:":
            self.committed_text = f"{self.committed_text}{text}"
        else:
            self.committed_text = f"{self.committed_text} {text}"

    def _find_unique_suffix(self, new_text: str, committed_norm: str, new_norm: str) -> str:
        committed_tail = committed_norm[-self.max_compare_chars :]
        new_compare = new_norm[: self.max_compare_chars]

        overlap_end = self._find_exact_overlap_end(committed_tail, new_compare)
        if overlap_end <= 0:
            overlap_end = self._find_fuzzy_overlap_end(committed_tail, new_compare)

        if overlap_end <= 0:
            return new_text

        return self._slice_original_after_normalized_chars(new_text, overlap_end)

    def _find_exact_overlap_end(self, committed_tail: str, new_compare: str) -> int:
        best_end = 0

        max_overlap = min(len(committed_tail), len(new_compare))
        for overlap in range(max_overlap, self.min_overlap_chars - 1, -1):
            if committed_tail[-overlap:] == new_compare[:overlap]:
                return overlap

        for start in range(0, max(0, len(new_compare) - self.min_overlap_chars + 1)):
            tail = new_compare[start:]
            if len(tail) < self.min_overlap_chars:
                break
            if tail in committed_tail:
                best_end = len(new_compare)
                break

        return best_end

    def _find_fuzzy_overlap_end(self, committed_tail: str, new_compare: str) -> int:
        matcher = SequenceMatcher(None, committed_tail, new_compare, autojunk=False)
        best_end = 0
        best_score = 0.0

        for match in matcher.get_matching_blocks():
            if match.size < self.min_overlap_chars:
                continue
            committed_reaches_tail = match.a + match.size >= len(committed_tail) - 3
            starts_near_window_start = match.b <= 40
            if not (committed_reaches_tail or starts_near_window_start):
                continue

            score = match.size / max(1, len(new_compare[: match.b + match.size]))
            if score > best_score:
                best_score = score
                best_end = match.b + match.size

        return best_end if best_score >= 0.45 else 0

    def _slice_original_after_normalized_chars(self, original: str, normalized_char_count: int) -> str:
        if normalized_char_count <= 0:
            return original

        normalized_seen = 0
        previous_was_space = True
        for index, char in enumerate(original):
            if char.isspace():
                if previous_was_space:
                    continue
                normalized_seen += 1
                previous_was_space = True
            else:
                normalized_seen += 1
                previous_was_space = False

            if normalized_seen >= normalized_char_count:
                return original[index + 1 :].lstrip(" ,;:-")

        return ""


TranscriptDeduper = FuzzyTranscriptDeduper


class RecentSentenceCache:
    """Track recent final sentences and reject exact or fuzzy duplicates."""

    def __init__(
        self,
        max_sentences: int = 20,
        similarity_threshold: float = 0.82,
    ) -> None:
        self.max_sentences = int(max_sentences)
        self.similarity_threshold = float(similarity_threshold)
        self._sentences: deque[str] = deque(maxlen=self.max_sentences)

    def is_duplicate(self, sentence: str) -> bool:
        normalized = normalize_text_for_compare(sentence)
        if not normalized:
            return True

        for previous in self._sentences:
            previous_normalized = normalize_text_for_compare(previous)
            if not previous_normalized:
                continue
            if normalized == previous_normalized:
                return True
            shorter, longer = sorted((normalized, previous_normalized), key=len)
            if len(shorter) >= 10 and shorter in longer:
                return True
            ratio = SequenceMatcher(None, normalized, previous_normalized, autojunk=False).ratio()
            if ratio >= self.similarity_threshold:
                return True
            if self._token_set_similarity(normalized, previous_normalized) >= self.similarity_threshold:
                return True
        return False

    def add(self, sentence: str) -> None:
        sentence = clean_whisper_artifacts(sentence)
        if sentence:
            self._sentences.append(sentence)

    def check_and_add(self, sentence: str) -> bool:
        if self.is_duplicate(sentence):
            return False
        self.add(sentence)
        return True

    def _token_set_similarity(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"\b[\wäöüÄÖÜß]+\b", left, flags=re.UNICODE))
        right_tokens = set(re.findall(r"\b[\wäöüÄÖÜß]+\b", right, flags=re.UNICODE))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
