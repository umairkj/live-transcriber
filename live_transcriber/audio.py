from __future__ import annotations

import logging
import queue
from collections import deque

import numpy as np


logger = logging.getLogger(__name__)


def calculate_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class AudioRecorder:
    """Continuously capture mono audio and return fixed-length chunks.

    The sounddevice input stream runs in the background, so audio keeps being
    buffered while Whisper is transcribing the previous chunk.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | None = None,
        silence_threshold: float = 0.003,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.device = device
        self.silence_threshold = float(silence_threshold)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._pending_blocks: deque[np.ndarray] = deque()
        self._pending_samples = 0
        self._stream = None
        self._stream_error: str | None = None

    def start(self) -> None:
        if self._stream is not None:
            return

        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise RuntimeError("sounddevice is not installed. Run ./setup.sh or pip install -r requirements.txt.") from exc

        def callback(indata, frames, time_info, status) -> None:
            if status:
                self._stream_error = str(status)
                logger.debug("Audio stream status: %s", status)

            audio = np.asarray(indata, dtype=np.float32)
            if audio.ndim == 2 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            self._audio_queue.put(audio.reshape(-1).copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise RuntimeError(f"Could not start audio input stream: {exc}") from exc

    def stop(self) -> None:
        if self._stream is None:
            return

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def __enter__(self) -> "AudioRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def record_chunk(self, seconds: int) -> np.ndarray:
        self.start()

        target_samples = int(self.sample_rate * seconds)
        if target_samples <= 0:
            return np.array([], dtype=np.float32)

        timeout_seconds = max(5.0, float(seconds) + 2.0)
        while self._pending_samples < target_samples:
            try:
                block = self._audio_queue.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                if self._stream_error:
                    raise RuntimeError(f"Audio input stream stopped providing data: {self._stream_error}") from exc
                raise RuntimeError("Timed out waiting for audio input data.") from exc

            block = np.asarray(block, dtype=np.float32).reshape(-1)
            if block.size == 0:
                continue
            self._pending_blocks.append(block)
            self._pending_samples += int(block.size)

        return self._pop_samples(target_samples)

    def _pop_samples(self, sample_count: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        remaining = int(sample_count)

        while remaining > 0 and self._pending_blocks:
            block = self._pending_blocks.popleft()
            if block.size <= remaining:
                chunks.append(block)
                remaining -= int(block.size)
                self._pending_samples -= int(block.size)
                continue

            chunks.append(block[:remaining])
            self._pending_blocks.appendleft(block[remaining:])
            self._pending_samples -= remaining
            remaining = 0

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def is_silent(self, audio: np.ndarray) -> bool:
        return calculate_rms(audio) < self.silence_threshold
