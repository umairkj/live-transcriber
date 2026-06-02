from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from live_transcriber.speech_activity import SpeechActivityDetector


def test_pause_finalizes_utterance() -> None:
    detector = SpeechActivityDetector(
        threshold=0.1,
        min_speech_seconds=0.3,
        pause_finalize_seconds=0.5,
        max_utterance_seconds=0,
    )

    events = []
    events.extend(detector.observe_frame(0.0, 0.2, 0.2))
    events.extend(detector.observe_frame(0.2, 0.4, 0.2))
    events.extend(detector.observe_frame(0.4, 0.6, 0.2))
    events.extend(detector.observe_frame(0.6, 0.8, 0.0))
    events.extend(detector.observe_frame(0.8, 1.0, 0.0))
    events.extend(detector.observe_frame(1.0, 1.2, 0.0))

    event_types = [event.event_type for event in events]
    assert "speech_start" in event_types, event_types
    speech_end = [event for event in events if event.event_type == "speech_end"]
    assert len(speech_end) == 1, events
    assert speech_end[0].start_time == 0.0
    assert round(speech_end[0].end_time, 1) == 0.6
    assert speech_end[0].reason == "pause"


def test_short_noise_does_not_start_utterance() -> None:
    detector = SpeechActivityDetector(
        threshold=0.1,
        min_speech_seconds=0.3,
        pause_finalize_seconds=0.5,
        max_utterance_seconds=0,
    )

    events = []
    events.extend(detector.observe_frame(0.0, 0.1, 0.2))
    events.extend(detector.observe_frame(0.1, 0.2, 0.0))
    events.extend(detector.observe_frame(0.2, 0.8, 0.0))
    assert events == []


def main() -> int:
    test_pause_finalizes_utterance()
    test_short_noise_does_not_start_utterance()
    print("speech activity tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
