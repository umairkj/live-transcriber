from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.audio import AudioFrame
from live_transcriber.device_utils import DeviceInfo
from live_transcriber.session import LiveTranscriberCallbacks, LiveTranscriberConfig, LiveTranscriberSession


class FakeRecorder:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = deque(frames)

    def __enter__(self) -> "FakeRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    def read(self, timeout: float = 0.2) -> AudioFrame | None:
        if not self.frames:
            time.sleep(0.005)
            return None
        time.sleep(0.003)
        return AudioFrame(self.frames.popleft())


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        text = "ich putze"
        if audio.size >= 300:
            text = "ich putze zaehne"
        return {
            "text": text,
            "language": "de",
            "language_probability": 1.0,
            "segments": [],
        }


class FakeWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def frame(value: float, samples: int = 100) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


def test_session_emits_events_and_does_not_save_when_disabled() -> None:
    recorder = FakeRecorder([frame(0.1), frame(0.1), frame(0.1), frame(0.0), frame(0.0)])
    writer_called = False

    def writer_factory(config: LiveTranscriberConfig) -> FakeWriter:
        nonlocal writer_called
        writer_called = True
        return FakeWriter()

    session = LiveTranscriberSession(
        recorder_factory=lambda config: recorder,
        transcriber_factory=lambda config: FakeTranscriber(),
        writer_factory=writer_factory,
        device_provider=lambda: [DeviceInfo(0, "BlackHole 2ch", 2, 48000.0)],
    )
    partials: list[str] = []
    finals: list[str] = []
    statuses: list[str] = []
    errors: list[str] = []

    config = LiveTranscriberConfig(
        device=0,
        save_transcript=False,
        sample_rate=1000,
        pause_seconds=0.2,
        pre_roll_seconds=0.0,
        min_speech_seconds=0.1,
        min_segment_seconds=0.1,
        partial_seconds=0.001,
    )
    callbacks = LiveTranscriberCallbacks(
        on_status=statuses.append,
        on_partial_text=partials.append,
        on_final_text=lambda text, record: finals.append(text),
        on_error=errors.append,
    )

    session.start(config, callbacks)
    deadline = time.time() + 2
    while not finals and time.time() < deadline:
        time.sleep(0.01)
    session.stop()
    session.join(2)

    assert not errors
    assert "Listening" in statuses
    assert partials
    assert finals == ["ich putze zaehne"]
    assert not writer_called


def test_session_saves_final_records_when_enabled() -> None:
    recorder = FakeRecorder([frame(0.1), frame(0.1), frame(0.1), frame(0.0), frame(0.0)])
    writer = FakeWriter()
    session = LiveTranscriberSession(
        recorder_factory=lambda config: recorder,
        transcriber_factory=lambda config: FakeTranscriber(),
        writer_factory=lambda config: writer,
        device_provider=lambda: [DeviceInfo(0, "BlackHole 2ch", 2, 48000.0)],
    )
    finals: list[str] = []

    config = LiveTranscriberConfig(
        device=0,
        save_transcript=True,
        sample_rate=1000,
        pause_seconds=0.2,
        pre_roll_seconds=0.0,
        min_speech_seconds=0.1,
        min_segment_seconds=0.1,
        partial_seconds=0,
    )
    callbacks = LiveTranscriberCallbacks(on_final_text=lambda text, record: finals.append(text))

    session.start(config, callbacks)
    deadline = time.time() + 2
    while not finals and time.time() < deadline:
        time.sleep(0.01)
    session.stop()
    session.join(2)

    assert finals == ["ich putze zaehne"]
    assert len(writer.records) == 1
    assert writer.records[0]["text"] == "ich putze zaehne"


def test_device_list_adapter() -> None:
    session = LiveTranscriberSession(device_provider=lambda: [DeviceInfo(7, "Example Mic", 1, 44100.0)])
    devices = session.list_devices()
    assert devices == [DeviceInfo(7, "Example Mic", 1, 44100.0)]


def main() -> None:
    test_session_emits_events_and_does_not_save_when_disabled()
    test_session_saves_final_records_when_enabled()
    test_device_list_adapter()
    print("session controller tests passed")


if __name__ == "__main__":
    main()
