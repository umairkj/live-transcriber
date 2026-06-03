from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.translations import build_selective_translations


def test_selective_translations_extract_useful_words() -> None:
    hints = build_selective_translations("Ich putze mir die Zaehne und trinke Wasser.")
    pairs = {(hint["source"], hint["translation"]) for hint in hints}
    assert ("putze", "clean") in pairs
    assert ("trinke", "drink") in pairs
    assert ("Wasser", "water") in pairs


def test_selective_translations_are_english_only_for_now() -> None:
    assert build_selective_translations("Ich trinke Wasser.", target_language="fr") == []


def main() -> None:
    test_selective_translations_extract_useful_words()
    test_selective_translations_are_english_only_for_now()
    print("translation helper tests passed")


if __name__ == "__main__":
    main()
