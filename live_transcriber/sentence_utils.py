from __future__ import annotations

import re


SENTENCE_ENDINGS = ".!?"


def clean_whisper_artifacts(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:\.\s*){3,}", "...", text)
    text = re.sub(r"([!?])\1{2,}", r"\1\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text.strip()


def normalize_text_for_compare(text: str) -> str:
    text = clean_whisper_artifacts(text).casefold()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=\S)", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_complete_sentences(text: str) -> tuple[list[str], str]:
    text = clean_whisper_artifacts(text)
    sentences: list[str] = []
    start = 0

    for index, char in enumerate(text):
        if char not in SENTENCE_ENDINGS:
            continue
        if char == "." and (
            text[index : index + 3] == "..."
            or text[max(0, index - 1) : index + 2] == "..."
            or text[max(0, index - 2) : index + 1] == "..."
        ):
            continue

        sentence = text[start : index + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = index + 1

    return sentences, text[start:].strip()
