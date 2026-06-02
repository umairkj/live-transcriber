from __future__ import annotations

from io import StringIO
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.live_transcript import LiveTranscriptDisplay


def main() -> int:
    stream = StringIO()
    display = LiveTranscriptDisplay(enabled=True, stream=stream)
    display.update(" Hallo   Welt ")
    display.update("Hallo Welt")
    display.update("Hallo Welt, wie geht es?")
    display.commit("Hallo Welt, wie geht es dir?", timestamp="2026-06-02 20:38:32")

    output = stream.getvalue().splitlines()
    assert output[0].endswith("] Hallo Welt"), output
    assert output[1].endswith("] Hallo Welt, wie geht es?"), output
    assert output[2] == "[20:38:32] Hallo Welt, wie geht es dir?", output
    print("live transcript tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
