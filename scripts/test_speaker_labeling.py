from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.speakers import EmbeddingSpeakerLabeler, format_speaker_text


def fake_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    del sample_rate
    if float(np.mean(audio)) < 0.5:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


def test_same_and_different_fake_voices_get_stable_labels() -> None:
    labeler = EmbeddingSpeakerLabeler(fake_embedding, backend="fake", similarity_threshold=0.8)

    voice_a = np.full(16000, 0.1, dtype=np.float32)
    voice_a_again = np.full(16000, 0.2, dtype=np.float32)
    voice_b = np.full(16000, 0.9, dtype=np.float32)

    assert labeler.label(voice_a, 16000, utterance_id=1).speaker_label == "A"
    assert labeler.label(voice_a_again, 16000, utterance_id=2).speaker_label == "A"
    assert labeler.label(voice_b, 16000, utterance_id=3).speaker_label == "B"


def test_short_segments_do_not_create_new_speakers() -> None:
    fresh_labeler = EmbeddingSpeakerLabeler(fake_embedding, backend="fake", min_new_speaker_seconds=1.0)
    short_voice = np.full(4000, 0.9, dtype=np.float32)
    assert fresh_labeler.label(short_voice, 16000, utterance_id=1).speaker_label == "?"

    labeler = EmbeddingSpeakerLabeler(fake_embedding, backend="fake", min_new_speaker_seconds=1.0)
    voice_a = np.full(16000, 0.1, dtype=np.float32)
    assert labeler.label(voice_a, 16000, utterance_id=1).speaker_label == "A"
    assert labeler.label(short_voice, 16000, utterance_id=2).speaker_label == "A"


def test_format_speaker_text() -> None:
    assert format_speaker_text("Hallo", "A") == "A: Hallo"
    assert format_speaker_text("Hallo", None) == "Hallo"


def main() -> None:
    test_same_and_different_fake_voices_get_stable_labels()
    test_short_segments_do_not_create_new_speakers()
    test_format_speaker_text()
    print("speaker labeling tests passed")


if __name__ == "__main__":
    main()
