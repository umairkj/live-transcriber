from __future__ import annotations

import logging

import numpy as np

from live_transcriber.rolling_buffer import RollingAudioBuffer


logger = logging.getLogger(__name__)


class ContinuousAudioStream:
    """Continuously append sounddevice input callback audio to a rolling buffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | None = None,
        block_duration_ms: int = 100,
        buffer_max_seconds: float = 30.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.device = device
        self.block_duration_ms = int(block_duration_ms)
        self.buffer = RollingAudioBuffer(sample_rate=self.sample_rate, max_seconds=buffer_max_seconds)
        self._stream = None
        self._last_status: str | None = None

    def start(self) -> None:
        if self._stream is not None:
            return

        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise RuntimeError("sounddevice is not installed. Run ./setup.sh or pip install -r requirements.txt.") from exc

        blocksize = max(1, int(self.sample_rate * (self.block_duration_ms / 1000.0)))

        def callback(indata, frames, time_info, status) -> None:
            if status:
                self._last_status = str(status)
                logger.debug("Audio stream status: %s", status)

            audio = np.asarray(indata, dtype=np.float32)
            if audio.ndim == 2 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            self.buffer.append(audio.reshape(-1))

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                blocksize=blocksize,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise RuntimeError(f"Could not start continuous audio stream: {exc}") from exc

    def stop(self) -> None:
        if self._stream is None:
            return

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def get_last(self, seconds: float) -> np.ndarray:
        return self.buffer.get_last(seconds)

    def get_last_with_range(self, seconds: float) -> tuple[np.ndarray, float, float]:
        return self.buffer.get_last_with_range(seconds)

    def get_range(self, start_seconds: float, end_seconds: float) -> np.ndarray:
        return self.buffer.get_range(start_seconds, end_seconds)

    def duration_seconds(self) -> float:
        return self.buffer.duration_seconds()

    def current_time_seconds(self) -> float:
        return self.buffer.current_time_seconds()

    @property
    def last_status(self) -> str | None:
        return self._last_status
