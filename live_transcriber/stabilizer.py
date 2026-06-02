from __future__ import annotations

from collections import deque
from difflib import SequenceMatcher

from live_transcriber.deduper import RecentSentenceCache
from live_transcriber.quality_filter import SentenceQualityFilter
from live_transcriber.sentence_utils import (
    SENTENCE_ENDINGS,
    clean_whisper_artifacts,
    normalize_text_for_compare,
    split_complete_sentences,
)


class HypothesisStabilityTracker:
    """Mark a candidate stable after it appears across consecutive windows."""

    def __init__(
        self,
        stability_windows: int = 2,
        similarity_threshold: float = 0.86,
    ) -> None:
        self.stability_windows = max(1, int(stability_windows))
        self.similarity_threshold = float(similarity_threshold)
        self._last_candidate = ""
        self._stable_count = 0

    def observe(self, candidate_sentence: str) -> bool:
        normalized = normalize_text_for_compare(candidate_sentence)
        if not normalized:
            return False

        if self.stability_windows <= 1:
            self._last_candidate = normalized
            self._stable_count = 1
            return True

        if self._last_candidate:
            ratio = SequenceMatcher(None, normalized, self._last_candidate, autojunk=False).ratio()
            shorter, longer = sorted((normalized, self._last_candidate), key=len)
            contained = len(shorter) >= 12 and shorter in longer
            if ratio >= self.similarity_threshold or contained:
                self._stable_count += 1
                self._last_candidate = normalized
                return self._stable_count >= self.stability_windows

        self._last_candidate = normalized
        self._stable_count = 1
        return False

    def reset(self) -> None:
        self._last_candidate = ""
        self._stable_count = 0


class TranscriptStabilizer:
    """Convert raw rolling-window hypotheses into clean stable final sentences."""

    def __init__(
        self,
        min_new_chars: int = 8,
        emit_partials: bool = False,
        require_sentence_end: bool = True,
        dedupe_min_overlap_chars: int = 20,
        stability_windows: int = 2,
        stability_similarity_threshold: float = 0.86,
        recent_duplicate_threshold: float = 0.82,
        min_final_words: int = 3,
        min_final_chars: int = 12,
        allow_short_utterances: bool = True,
        debug: bool = False,
    ) -> None:
        self.emit_partials = bool(emit_partials)
        self.require_sentence_end = bool(require_sentence_end)
        self.min_new_chars = int(min_new_chars)
        self.debug = bool(debug)
        self.quality_filter = SentenceQualityFilter(
            min_words=min_final_words,
            min_chars=min_final_chars,
            allow_short_utterances=allow_short_utterances,
        )
        self.stability_tracker = HypothesisStabilityTracker(
            stability_windows=stability_windows,
            similarity_threshold=stability_similarity_threshold,
        )
        self.recent_cache = RecentSentenceCache(similarity_threshold=recent_duplicate_threshold)
        self.partial_text = ""

    def process_window_text(self, raw_window_text: str) -> dict:
        cleaned = clean_whisper_artifacts(raw_window_text)
        if not cleaned:
            return self._result(raw_window_text, "", [], "", [], "", [])

        candidate_sentences, partial_tail = split_complete_sentences(cleaned)
        rejected_candidates: list[dict[str, str]] = []
        stable_sentences: list[str] = []

        for candidate in candidate_sentences:
            candidate = clean_whisper_artifacts(candidate)
            if not candidate:
                continue
            if len(candidate) < self.min_new_chars and not self.quality_filter.is_short_allowlisted(candidate):
                rejected_candidates.append({"text": candidate, "reason": "too_short"})
                continue
            if not self.quality_filter.is_good_final_sentence(candidate):
                rejected_candidates.append({"text": candidate, "reason": "quality_filter"})
                continue
            if not self.stability_tracker.observe(candidate):
                rejected_candidates.append({"text": candidate, "reason": "not_stable_yet"})
                continue
            if self.recent_cache.is_duplicate(candidate):
                rejected_candidates.append({"text": candidate, "reason": "recent_duplicate"})
                continue

            self.recent_cache.add(candidate)
            stable_sentences.append(candidate)

        self.partial_text = partial_tail
        if self.emit_partials and partial_tail and not stable_sentences:
            partial_text = partial_tail
        else:
            partial_text = partial_tail if self.emit_partials else ""

        emitted_text = " ".join(stable_sentences).strip()
        return self._result(
            raw_window_text=raw_window_text,
            cleaned_window_text=cleaned,
            candidate_sentences=candidate_sentences,
            partial_text=partial_text,
            stable_sentences=stable_sentences,
            emitted_text=emitted_text,
            rejected_candidates=rejected_candidates,
        )

    def finalize_utterance_text(self, raw_utterance_text: str) -> dict:
        cleaned = clean_whisper_artifacts(raw_utterance_text)
        if not cleaned:
            return self._result(raw_utterance_text, "", [], "", [], "", [])

        complete_sentences, partial_tail = split_complete_sentences(cleaned)
        candidate_sentences = list(complete_sentences)
        if partial_tail:
            candidate_sentences.append(self._final_tail_candidate(partial_tail))
        if not candidate_sentences:
            candidate_sentences = [self._final_tail_candidate(cleaned)]

        rejected_candidates: list[dict[str, str]] = []
        final_sentences: list[str] = []
        for candidate in candidate_sentences:
            candidate = clean_whisper_artifacts(candidate)
            if not candidate:
                continue
            if not self.quality_filter.is_good_final_sentence(candidate):
                rejected_candidates.append({"text": candidate, "reason": "quality_filter"})
                continue

            final_sentences.append(candidate)

        emitted_text = " ".join(final_sentences).strip()
        return self._result(
            raw_window_text=raw_utterance_text,
            cleaned_window_text=cleaned,
            candidate_sentences=candidate_sentences,
            partial_text="",
            stable_sentences=final_sentences,
            emitted_text=emitted_text,
            rejected_candidates=rejected_candidates,
        )

    def reset(self) -> None:
        self.partial_text = ""
        self.stability_tracker.reset()
        self.recent_cache = RecentSentenceCache(similarity_threshold=self.recent_cache.similarity_threshold)

    def _final_tail_candidate(self, text: str) -> str:
        text = clean_whisper_artifacts(text)
        if not text:
            return ""
        if text.endswith(tuple(SENTENCE_ENDINGS)):
            return text
        punctuated = f"{text}."
        if self.quality_filter.is_short_allowlisted(punctuated):
            return punctuated
        return text

    def _result(
        self,
        raw_window_text: str,
        cleaned_window_text: str,
        candidate_sentences: list[str],
        partial_text: str,
        stable_sentences: list[str],
        emitted_text: str,
        rejected_candidates: list[dict[str, str]],
    ) -> dict:
        return {
            "raw_window_text": raw_window_text,
            "cleaned_window_text": cleaned_window_text,
            "candidate_sentences": candidate_sentences,
            "partial_text": partial_text,
            "stable_sentences": stable_sentences,
            "emitted_text": emitted_text,
            "rejected_candidates": rejected_candidates,
            "new_text": emitted_text,
        }
