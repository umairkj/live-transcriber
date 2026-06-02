from __future__ import annotations

import threading

import numpy as np


class RollingAudioBuffer:
    """Thread-safe rolling mono float32 audio buffer."""

    def __init__(self, sample_rate: int, max_seconds: float = 30.0) -> None:
        self.sample_rate = int(sample_rate)
        self.max_seconds = float(max_seconds)
        self._max_samples = max(1, int(self.sample_rate * self.max_seconds))
        self._audio = np.array([], dtype=np.float32)
        self._start_sample = 0
        self._total_samples = 0
        self._lock = threading.Lock()

    def append(self, audio: np.ndarray) -> tuple[float, float]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            current_time = self.current_time_seconds()
            return current_time, current_time

        with self._lock:
            start_sample = self._total_samples
            self._total_samples += int(audio.size)
            if self._audio.size == 0:
                self._audio = audio[-self._max_samples :].copy()
            else:
                self._audio = np.concatenate([self._audio, audio])[-self._max_samples :].astype(
                    np.float32,
                    copy=False,
                )
            self._start_sample = max(0, self._total_samples - int(self._audio.size))
            return start_sample / self.sample_rate, self._total_samples / self.sample_rate

    def get_last(self, seconds: float) -> np.ndarray:
        sample_count = max(0, int(self.sample_rate * float(seconds)))
        if sample_count <= 0:
            return np.array([], dtype=np.float32)

        with self._lock:
            return self._audio[-sample_count:].copy()

    def get_last_with_range(self, seconds: float) -> tuple[np.ndarray, float, float]:
        sample_count = max(0, int(self.sample_rate * float(seconds)))
        if sample_count <= 0:
            current_time = self.current_time_seconds()
            return np.array([], dtype=np.float32), current_time, current_time

        with self._lock:
            audio = self._audio[-sample_count:].copy()
            end_sample = self._total_samples
            start_sample = max(self._start_sample, end_sample - int(audio.size))
            return audio, start_sample / self.sample_rate, end_sample / self.sample_rate

    def get_range(self, start_seconds: float, end_seconds: float) -> np.ndarray:
        with self._lock:
            start_sample = max(self._start_sample, int(float(start_seconds) * self.sample_rate))
            end_sample = min(self._total_samples, int(float(end_seconds) * self.sample_rate))
            if end_sample <= start_sample:
                return np.array([], dtype=np.float32)

            local_start = start_sample - self._start_sample
            local_end = end_sample - self._start_sample
            return self._audio[local_start:local_end].copy()

    def duration_seconds(self) -> float:
        with self._lock:
            return float(self._audio.size) / float(self.sample_rate)

    def current_time_seconds(self) -> float:
        with self._lock:
            return float(self._total_samples) / float(self.sample_rate)

    def clear(self) -> None:
        with self._lock:
            self._audio = np.array([], dtype=np.float32)
            self._start_sample = self._total_samples
