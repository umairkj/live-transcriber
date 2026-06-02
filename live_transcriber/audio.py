from __future__ import annotations

import numpy as np


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
