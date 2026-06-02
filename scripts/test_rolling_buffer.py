from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.rolling_buffer import RollingAudioBuffer


def test_ranges_follow_absolute_stream_time() -> None:
    buffer = RollingAudioBuffer(sample_rate=10, max_seconds=1.0)
    first_start, first_end = buffer.append(np.arange(8, dtype=np.float32))
    second_start, second_end = buffer.append(np.arange(8, 16, dtype=np.float32))

    assert first_start == 0
    assert first_end == 0.8
    assert second_start == 0.8
    assert second_end == 1.6
    assert buffer.current_time_seconds() == 1.6

    ranged = buffer.get_range(0.6, 1.2)
    assert ranged.tolist() == [6, 7, 8, 9, 10, 11]

    last_audio, start, end = buffer.get_last_with_range(0.3)
    assert last_audio.tolist() == [13, 14, 15]
    assert round(start, 1) == 1.3
    assert round(end, 1) == 1.6


def main() -> int:
    test_ranges_follow_absolute_stream_time()
    print("rolling buffer tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
