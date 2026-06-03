from __future__ import annotations

import re


_GARBAGE_FRAGMENTS = {
    "uf",
    "uh",
    "äh",
    "hm",
    "hmm",
    "mhm",
    "mm",
}

_ALLOWED_SHORT_FRAGMENTS = {
    "ja",
    "nein",
    "ok",
    "okay",
    "gut",
    "so",
    "doch",
    "genau",
}

_HALLUCINATION_MARKERS = (
    "untertitel",
    "subtitles",
    "subtitle",
    "amara.org",
    "swiss txt",
)


def cleanup_transcript_text(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([¿([{])\s+", r"\1", text)
    text = re.sub(r"\.\s+\.\s+\.", "...", text)
    text = re.sub(r"([!?]){3,}", r"\1", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\s+([’'])", r"\1", text)
    return text.strip(" \t\r\n-")


def is_garbage_fragment(text: str, min_text_length: int = 1) -> bool:
    cleaned = cleanup_transcript_text(text)
    if len(cleaned) < int(min_text_length):
        return True

    normalized = _normalize_fragment(cleaned)
    if not normalized:
        return True

    if normalized in _ALLOWED_SHORT_FRAGMENTS:
        return False

    if normalized in _GARBAGE_FRAGMENTS:
        return True

    if len(normalized) <= 2:
        return True

    lowered = cleaned.lower()
    if any(marker in lowered for marker in _HALLUCINATION_MARKERS):
        return True

    return False


def _normalize_fragment(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\wäöüß]+", "", text, flags=re.IGNORECASE)
    return text.strip()
