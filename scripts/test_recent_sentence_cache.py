from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.deduper import RecentSentenceCache


def main() -> int:
    cache = RecentSentenceCache(similarity_threshold=0.82)
    assert cache.check_and_add("seine Zeit für sich schon genommen.")
    assert not cache.check_and_add("seine Zeit für sich schon genommen.")
    assert cache.check_and_add("Tatsächlich ja.")
    assert not cache.check_and_add("Ja, tatsächlich.")
    assert cache.check_and_add("Der Kunde hat die Zahlung bestätigt.")
    print("recent sentence cache tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
