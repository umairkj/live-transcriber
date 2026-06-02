from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.deduper import FuzzyTranscriptDeduper
from live_transcriber.sentence_utils import split_complete_sentences


def assert_equal(actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"Expected {expected!r}, got {actual!r}")


def test_lcp_deduper() -> None:
    deduper = FuzzyTranscriptDeduper()
    deduper.committed_text = "Die Bestellung wurde"
    assert_equal(
        deduper.process("Die Bestellung wurde gestern versendet."),
        "gestern versendet.",
    )


def test_overlap_deduper() -> None:
    deduper = FuzzyTranscriptDeduper()
    deduper.committed_text = "Die Bestellung wurde gestern versendet."
    assert_equal(
        deduper.process("wurde gestern versendet. Der Kunde hat die Zahlung bestätigt."),
        "Der Kunde hat die Zahlung bestätigt.",
    )


def test_sentence_splitter() -> None:
    complete, tail = split_complete_sentences("Hallo. Wie geht es dir? Ich warte")
    assert_equal(complete, ["Hallo.", "Wie geht es dir?"])
    assert_equal(tail, "Ich warte")


def main() -> int:
    test_lcp_deduper()
    test_overlap_deduper()
    test_sentence_splitter()
    print("deduper tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
