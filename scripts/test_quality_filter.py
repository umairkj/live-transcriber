from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.quality_filter import SentenceQualityFilter


def main() -> int:
    quality = SentenceQualityFilter()
    assert not quality.is_good_final_sentence("Uhr.")
    assert not quality.is_good_final_sentence("uf.")
    assert not quality.is_good_final_sentence("gearbeitet.")
    assert quality.is_good_final_sentence("Genau.")
    assert quality.is_good_final_sentence("Ja.")
    assert quality.is_good_final_sentence("Hallo.")
    assert quality.is_good_final_sentence("Ein Glas Wasser trinken, dann gehe ich mich waschen, Zähne putzen.")
    assert not quality.is_good_final_sentence("ich nehme das weil ich nehme das weil.")
    print("quality filter tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
