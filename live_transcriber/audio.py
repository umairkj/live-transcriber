from __future__ import annotations

import math
from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """One mono audio block captured by the sounddevice callback."""

    audio: np.ndarray
    status: str | None = None


def audio_signal_level(audio: np.ndarray, active_threshold: float = 0.008) -> tuple[float, bool]:
    """Return a display-friendly 0..1 audio level and whether it looks active."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0, False

    rms = float(np.sqrt(np.mean(np.square(np.clip(audio, -1.0, 1.0)))))
    db = 20.0 * math.log10(max(rms, 1e-9))
    level = max(0.0, min(1.0, (db + 60.0) / 54.0))
    return level, rms >= active_threshold


class AudioRecorder:
    """Record fixed-length mono audio chunks with sounddevice."""

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

    def record_chunk(self, seconds: int) -> np.ndarray:
        frames = int(self.sample_rate * seconds)
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise RuntimeError("sounddevice is not installed. Run ./setup.sh or pip install -r requirements.txt.") from exc

        try:
            recording = sd.rec(
                frames=frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
            )
            sd.wait()
        except Exception as exc:
            raise RuntimeError(f"Could not record audio chunk: {exc}") from exc

        audio = np.asarray(recording, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        return audio.reshape(-1)

    def is_silent(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True

        audio = np.asarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < self.silence_threshold


class ContinuousAudioRecorder:
    """Continuously capture mono audio blocks without doing heavy callback work."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | None = None,
        frame_duration_ms: int = 100,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.device = device
        self.frame_duration_ms = int(frame_duration_ms)
        self.blocksize = max(1, int(self.sample_rate * self.frame_duration_ms / 1000))
        self._frames: Queue[AudioFrame] = Queue()
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:
            raise RuntimeError("sounddevice is not installed. Run ./setup.sh or pip install -r requirements.txt.") from exc

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            audio = np.asarray(indata, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

            status_text = str(status) if status else None
            self._frames.put(AudioFrame(audio=audio.reshape(-1).copy(), status=status_text))

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                blocksize=self.blocksize,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"Could not start continuous audio stream: {exc}") from exc

    def read(self, timeout: float = 0.2) -> AudioFrame | None:
        try:
            return self._frames.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        if self._stream is None:
            return

        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def __enter__(self) -> "ContinuousAudioRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.stop()
