from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


DEFAULT_SPEAKER_BACKEND = "local"
SPEAKER_BACKENDS = ("local", "pyannote")
DEFAULT_SPEAKER_SIMILARITY_THRESHOLD = 0.72
DEFAULT_MIN_NEW_SPEAKER_SECONDS = 1.0


@dataclass(frozen=True)
class SpeakerInfo:
    speaker_label: str
    confidence: float | None
    backend: str


class SpeakerLabeler(Protocol):
    def label(self, audio: np.ndarray, sample_rate: int, utterance_id: int) -> SpeakerInfo: ...
    def reset(self) -> None: ...


class SpeakerBackendUnavailable(RuntimeError):
    """Raised when an optional speaker-labeling backend cannot be loaded."""


@dataclass
class _SpeakerProfile:
    label: str
    embedding: np.ndarray
    updates: int = 1


class EmbeddingSpeakerLabeler:
    """Assign anonymous speaker labels by comparing utterance embeddings."""

    def __init__(
        self,
        embedding_func: Callable[[np.ndarray, int], np.ndarray],
        backend: str,
        similarity_threshold: float = DEFAULT_SPEAKER_SIMILARITY_THRESHOLD,
        min_new_speaker_seconds: float = DEFAULT_MIN_NEW_SPEAKER_SECONDS,
    ) -> None:
        self._embedding_func = embedding_func
        self._backend = backend
        self._similarity_threshold = float(similarity_threshold)
        self._min_new_speaker_seconds = float(min_new_speaker_seconds)
        self._profiles: list[_SpeakerProfile] = []
        self._previous_label: str | None = None

    def label(self, audio: np.ndarray, sample_rate: int, utterance_id: int) -> SpeakerInfo:
        del utterance_id
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_seconds = float(audio.size) / float(sample_rate)

        if duration_seconds < self._min_new_speaker_seconds:
            label = self._previous_label or "?"
            return SpeakerInfo(speaker_label=label, confidence=None, backend=self._backend)

        embedding = self._normalize_embedding(self._embedding_func(audio, sample_rate))
        if embedding is None:
            label = self._previous_label or "?"
            return SpeakerInfo(speaker_label=label, confidence=None, backend=self._backend)

        best_profile: _SpeakerProfile | None = None
        best_similarity = -1.0
        for profile in self._profiles:
            similarity = _cosine_similarity(embedding, profile.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_profile = profile

        if best_profile is not None and best_similarity >= self._similarity_threshold:
            label = best_profile.label
            updated_embedding = self._normalize_embedding(
                (best_profile.embedding * best_profile.updates + embedding) / float(best_profile.updates + 1)
            )
            if updated_embedding is not None:
                best_profile.embedding = updated_embedding
            best_profile.updates += 1
            confidence = round(float(best_similarity), 4)
        else:
            label = _label_for_index(len(self._profiles))
            self._profiles.append(_SpeakerProfile(label=label, embedding=embedding))
            confidence = None

        self._previous_label = label
        return SpeakerInfo(speaker_label=label, confidence=confidence, backend=self._backend)

    def reset(self) -> None:
        self._profiles.clear()
        self._previous_label = None

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray | None:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size == 0 or norm <= 0:
            return None
        return embedding / norm


class SpeechBrainSpeakerLabeler(EmbeddingSpeakerLabeler):
    """SpeechBrain ECAPA speaker embeddings with local profile matching."""

    def __init__(self) -> None:
        self._torch, self._classifier = self._load_classifier()
        super().__init__(embedding_func=self._encode, backend="local")

    def _load_classifier(self):  # noqa: ANN202
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ModuleNotFoundError as exc:
            raise SpeakerBackendUnavailable(
                "Speaker labels need optional local speaker dependencies. "
                "Install them with: venv/bin/python -m pip install speechbrain torch torchaudio"
            ) from exc

        try:
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not load SpeechBrain speaker model: {exc}") from exc

        return torch, classifier

    def _encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = _resample_audio(audio, sample_rate, target_rate=16000)
        audio = _peak_normalize(audio)
        waveform = self._torch.from_numpy(audio).float().unsqueeze(0)
        with self._torch.no_grad():
            embedding = self._classifier.encode_batch(waveform)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)


def create_speaker_labeler(config) -> SpeakerLabeler | None:  # noqa: ANN001
    if not getattr(config, "speaker_labels", False):
        return None

    backend = str(getattr(config, "speaker_backend", DEFAULT_SPEAKER_BACKEND)).casefold()
    if backend == "local":
        return SpeechBrainSpeakerLabeler()
    if backend == "pyannote":
        raise SpeakerBackendUnavailable(
            "The pyannote speaker backend is planned as an optional advanced backend, "
            "but it is not wired into this app yet. Use --speaker-backend local."
        )
    raise ValueError(f"Unsupported speaker backend: {backend}")


def format_speaker_text(text: str, speaker_label: str | None) -> str:
    if not speaker_label:
        return text
    return f"{speaker_label}: {text}"


def _label_for_index(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, len(alphabet))
        label = alphabet[remainder] + label
    return label


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return -1.0
    return float(np.dot(left, right) / denominator)


def _peak_normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def _resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or sample_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    duration_seconds = float(audio.size) / float(sample_rate)
    target_size = max(1, int(round(duration_seconds * float(target_rate))))
    source_positions = np.linspace(0.0, duration_seconds, num=audio.size, endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)
