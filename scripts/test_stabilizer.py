from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.stabilizer import TranscriptStabilizer


def feed_windows(windows: list[str]) -> list[str]:
    stabilizer = TranscriptStabilizer()
    emitted: list[str] = []
    for window in windows:
        result = stabilizer.process_window_text(window)
        emitted.extend(result["stable_sentences"])
    return emitted


def assert_contains(items: list[str], expected: str) -> None:
    if expected not in items:
        raise AssertionError(f"Expected {expected!r} in {items!r}")


def test_order_payment_flow() -> None:
    emitted = feed_windows(
        [
            "Die Bestellung wurde",
            "Die Bestellung wurde gestern versendet.",
            "Die Bestellung wurde gestern versendet.",
            "Bestellung wurde gestern versendet. Der Kunde",
            "Der Kunde hat die Zahlung bestätigt.",
            "Der Kunde hat die Zahlung bestätigt.",
        ]
    )
    assert_contains(emitted, "Die Bestellung wurde gestern versendet.")
    assert_contains(emitted, "Der Kunde hat die Zahlung bestätigt.")


def test_rolling_repetition_reduced() -> None:
    emitted = feed_windows(
        [
            "Dann ist 9 Uhr. Wow, dann stehen Sie schon um 5 Uhr auf. 6 Uhr.",
            "Wow, dann stehen Sie schon um 5 Uhr auf. 6 Uhr gerade, tatsächlich ja.",
            "6 Uhr gerade tatsächlich ja. Aber das ist fantastisch.",
        ]
    )
    assert len(emitted) == len(set(emitted)), emitted
    assert not any(text.startswith("Wow, dann stehen Sie") for text in emitted[2:]), emitted


def test_incomplete_then_complete() -> None:
    stabilizer = TranscriptStabilizer()
    results = [
        stabilizer.process_window_text("Ein Glas Wasser trinken, dann gehe ich mich..."),
        stabilizer.process_window_text("Ein Glas Wasser trinken, dann gehe ich mich waschen..."),
        stabilizer.process_window_text("Ein Glas Wasser trinken, dann gehe ich mich waschen, Zähne putzen."),
        stabilizer.process_window_text("Ein Glas Wasser trinken, dann gehe ich mich waschen, Zähne putzen."),
    ]
    assert results[0]["stable_sentences"] == []
    assert results[1]["stable_sentences"] == []
    assert results[2]["stable_sentences"] == []
    emitted = [
        sentence
        for result in results
        for sentence in result["stable_sentences"]
    ]
    assert emitted == ["Ein Glas Wasser trinken, dann gehe ich mich waschen, Zähne putzen."], emitted


def test_fragment_rejection() -> None:
    emitted = feed_windows(
        [
            "Uhr.",
            "Uhr.",
            "uf.",
            "uf.",
            "gearbeitet.",
            "gearbeitet.",
        ]
    )
    assert emitted == [], emitted


def test_duplicate_window() -> None:
    emitted = feed_windows(
        [
            "Der Kunde hat die Zahlung bestätigt.",
            "Der Kunde hat die Zahlung bestätigt.",
        ]
    )
    assert emitted == ["Der Kunde hat die Zahlung bestätigt."], emitted


def test_finalize_utterance_accepts_pause_bounded_tail() -> None:
    stabilizer = TranscriptStabilizer()
    result = stabilizer.finalize_utterance_text("Mein Nachname ist Schneider")
    assert result["stable_sentences"] == ["Mein Nachname ist Schneider"], result


def test_finalize_utterance_keeps_repeated_real_speech() -> None:
    stabilizer = TranscriptStabilizer()
    first = stabilizer.finalize_utterance_text("Hallo")
    second = stabilizer.finalize_utterance_text("Hallo")
    assert first["stable_sentences"] == ["Hallo."], first
    assert second["stable_sentences"] == ["Hallo."], second


def main() -> int:
    test_order_payment_flow()
    test_rolling_repetition_reduced()
    test_incomplete_then_complete()
    test_fragment_rejection()
    test_duplicate_window()
    test_finalize_utterance_accepts_pause_bounded_tail()
    test_finalize_utterance_keeps_repeated_real_speech()
    print("stabilizer tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
