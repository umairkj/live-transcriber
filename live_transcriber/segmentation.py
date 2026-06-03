from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechSegment:
    """A completed speech segment ready for Whisper."""

    utterance_id: int
    audio: np.ndarray
    duration_seconds: float
    speech_seconds: float
    rms_peak: float
    closed_by: str


@dataclass(frozen=True)
class SpeechSnapshot:
    """A copy of the active speech segment for provisional transcription."""

    utterance_id: int
    audio: np.ndarray
    duration_seconds: float
    speech_seconds: float
    rms_peak: float


@dataclass(frozen=True)
class _BufferedFrame:
    audio: np.ndarray
    is_speech: bool
    rms: float


class SpeechSegmenter:
    """Turn continuous audio frames into pause-finalized speech segments."""

    def __init__(
        self,
        sample_rate: int,
        speech_threshold: float,
        pause_seconds: float,
        pre_roll_seconds: float,
        min_speech_seconds: float,
        min_segment_seconds: float,
        max_segment_seconds: float,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.speech_threshold = float(speech_threshold)
        self.pause_samples = int(float(pause_seconds) * self.sample_rate)
        self.pre_roll_samples = int(float(pre_roll_seconds) * self.sample_rate)
        self.min_speech_samples = int(float(min_speech_seconds) * self.sample_rate)
        self.min_segment_samples = int(float(min_segment_seconds) * self.sample_rate)
        self.max_segment_samples = int(float(max_segment_seconds) * self.sample_rate)

        self._pre_roll: deque[_BufferedFrame] = deque()
        self._pre_roll_sample_count = 0
        self._current_frames: list[np.ndarray] = []
        self._current_sample_count = 0
        self._speech_sample_count = 0
        self._speech_run_samples = 0
        self._silence_sample_count = 0
        self._rms_peak = 0.0
        self._in_speech = False
        self._next_utterance_id = 0
        self._current_utterance_id: int | None = None

    def process(self, audio: np.ndarray) -> list[SpeechSegment]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []

        rms = _rms(audio)
        is_speech = rms >= self.speech_threshold

        if not self._in_speech:
            self._remember_pre_roll(audio, is_speech=is_speech, rms=rms)
            if is_speech:
                self._speech_run_samples += int(audio.size)
                if self._speech_run_samples >= self.min_speech_samples:
                    self._start_segment_from_pre_roll()
            else:
                self._speech_run_samples = 0
            return []

        self._append_current(audio, is_speech=is_speech, rms=rms)
        if is_speech:
            self._silence_sample_count = 0
        else:
            self._silence_sample_count += int(audio.size)

        if self.pause_samples > 0 and self._silence_sample_count >= self.pause_samples:
            segment = self._close_segment("pause")
            return [segment] if segment is not None else []

        if self.max_segment_samples > 0 and self._current_sample_count >= self.max_segment_samples:
            segment = self._close_segment("max_duration")
            return [segment] if segment is not None else []

        return []

    def flush(self) -> SpeechSegment | None:
        return self._close_segment("stop") if self._in_speech else None

    @property
    def current_utterance_id(self) -> int | None:
        return self._current_utterance_id if self._in_speech else None

    def active_snapshot(self) -> SpeechSnapshot | None:
        if not self._in_speech or self._current_utterance_id is None or not self._current_frames:
            return None

        audio = np.concatenate(self._current_frames).astype(np.float32, copy=True)
        if audio.size < self.min_segment_samples or self._speech_sample_count < self.min_speech_samples:
            return None

        return SpeechSnapshot(
            utterance_id=self._current_utterance_id,
            audio=audio,
            duration_seconds=round(float(audio.size) / float(self.sample_rate), 3),
            speech_seconds=round(float(self._speech_sample_count) / float(self.sample_rate), 3),
            rms_peak=round(float(self._rms_peak), 6),
        )

    def _remember_pre_roll(self, audio: np.ndarray, is_speech: bool, rms: float) -> None:
        buffered = _BufferedFrame(audio=audio.copy(), is_speech=is_speech, rms=rms)
        self._pre_roll.append(buffered)
        self._pre_roll_sample_count += int(buffered.audio.size)

        while self.pre_roll_samples > 0 and self._pre_roll_sample_count > self.pre_roll_samples:
            removed = self._pre_roll.popleft()
            self._pre_roll_sample_count -= int(removed.audio.size)

        if self.pre_roll_samples <= 0:
            self._pre_roll.clear()
            self._pre_roll_sample_count = 0

    def _start_segment_from_pre_roll(self) -> None:
        self._next_utterance_id += 1
        self._current_utterance_id = self._next_utterance_id
        self._in_speech = True
        self._current_frames = []
        self._current_sample_count = 0
        self._speech_sample_count = 0
        self._silence_sample_count = 0
        self._rms_peak = 0.0

        for frame in self._pre_roll:
            self._append_current(frame.audio, is_speech=frame.is_speech, rms=frame.rms)

        self._pre_roll.clear()
        self._pre_roll_sample_count = 0

    def _append_current(self, audio: np.ndarray, is_speech: bool, rms: float) -> None:
        self._current_frames.append(audio.copy())
        self._current_sample_count += int(audio.size)
        if is_speech:
            self._speech_sample_count += int(audio.size)
        self._rms_peak = max(self._rms_peak, float(rms))

    def _close_segment(self, closed_by: str) -> SpeechSegment | None:
        if not self._current_frames:
            self._reset()
            return None

        audio = np.concatenate(self._current_frames).astype(np.float32, copy=False)
        speech_samples = self._speech_sample_count
        rms_peak = self._rms_peak
        utterance_id = self._current_utterance_id
        self._reset()

        if utterance_id is None or audio.size < self.min_segment_samples or speech_samples < self.min_speech_samples:
            return None

        return SpeechSegment(
            utterance_id=utterance_id,
            audio=audio,
            duration_seconds=round(float(audio.size) / float(self.sample_rate), 3),
            speech_seconds=round(float(speech_samples) / float(self.sample_rate), 3),
            rms_peak=round(float(rms_peak), 6),
            closed_by=closed_by,
        )

    def _reset(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_sample_count = 0
        self._current_frames = []
        self._current_sample_count = 0
        self._speech_sample_count = 0
        self._speech_run_samples = 0
        self._silence_sample_count = 0
        self._rms_peak = 0.0
        self._in_speech = False
        self._current_utterance_id = None


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    audio = audio.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(np.square(audio))))
