from __future__ import annotations

import re
from collections import deque
from difflib import SequenceMatcher


class RecentTextDeduper:
    """Conservatively suppress repeated completed transcript lines."""

    def __init__(self, max_items: int = 12, similarity_threshold: float = 0.97) -> None:
        self.max_items = int(max_items)
        self.similarity_threshold = float(similarity_threshold)
        self._recent: deque[str] = deque(maxlen=self.max_items)

    def is_duplicate(self, text: str) -> bool:
        normalized = _normalize(text)
        if not normalized:
            return True

        for previous in self._recent:
            if normalized == previous:
                return True
            if len(normalized) >= 24 and SequenceMatcher(None, normalized, previous).ratio() >= self.similarity_threshold:
                return True

        return False

    def remember(self, text: str) -> None:
        normalized = _normalize(text)
        if normalized:
            self._recent.append(normalized)


def _normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\wäöüß]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()
