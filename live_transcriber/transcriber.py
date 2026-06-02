from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """Small wrapper around faster-whisper for chunk transcription."""

    def __init__(
        self,
        model_name: str = "base",
        device_type: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_type = device_type
        self.compute_type = compute_type
        self.language = language

        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("faster-whisper is not installed. Run ./setup.sh or pip install -r requirements.txt.") from exc

        try:
            self.model = WhisperModel(model_name, device=device_type, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load faster-whisper model '{model_name}' "
                f"on device '{device_type}' with compute type '{compute_type}': {exc}"
            ) from exc

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {
                "text": "",
                "language": None,
                "language_probability": None,
                "segments": [],
            }

        if sample_rate == 16000:
            try:
                return self._transcribe_source(audio)
            except Exception as exc:
                logger.warning("Direct numpy transcription failed; retrying with temporary WAV: %s", exc)

        wav_path = self._write_temp_wav(audio, sample_rate)
        try:
            return self._transcribe_source(str(wav_path))
        except Exception as exc:
            raise RuntimeError(f"Could not transcribe audio chunk: {exc}") from exc
        finally:
            wav_path.unlink(missing_ok=True)

    def _transcribe_source(self, source: str | np.ndarray) -> dict[str, Any]:
        segments_iter, info = self.model.transcribe(
            source,
            language=self.language,
            beam_size=1,
        )

        segments: list[dict[str, float | str]] = []
        text_parts: list[str] = []
        for segment in segments_iter:
            segment_text = segment.text.strip()
            if segment_text:
                text_parts.append(segment_text)
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment_text,
                }
            )

        language_probability = getattr(info, "language_probability", None)
        if language_probability is not None:
            language_probability = float(language_probability)

        return {
            "text": " ".join(text_parts).strip(),
            "language": getattr(info, "language", None),
            "language_probability": language_probability,
            "segments": segments,
        }

    def _write_temp_wav(self, audio: np.ndarray, sample_rate: int) -> Path:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm_audio = (clipped * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = Path(temp_file.name)

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm_audio.tobytes())

        return wav_path
