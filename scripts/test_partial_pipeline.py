from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.jobs import TranscriptionJob, TranscriptionJobQueue
from live_transcriber.partials import PartialTextTracker
from live_transcriber.segmentation import SpeechSegmenter


def frame(value: float, samples: int = 100) -> np.ndarray:
    return np.full(samples, value, dtype=np.float32)


def make_job(kind: str, utterance_id: int) -> TranscriptionJob:
    return TranscriptionJob(
        kind=kind,  # type: ignore[arg-type]
        utterance_id=utterance_id,
        audio=np.zeros(100, dtype=np.float32),
        duration_seconds=0.1,
        speech_seconds=0.1,
        rms_peak=0.1,
        closed_by="pause" if kind == "final" else None,
    )


def test_active_snapshot_and_final_segment() -> None:
    segmenter = SpeechSegmenter(
        sample_rate=1000,
        speech_threshold=0.01,
        pause_seconds=0.2,
        pre_roll_seconds=0.2,
        min_speech_seconds=0.1,
        min_segment_seconds=0.1,
        max_segment_seconds=5.0,
    )

    assert segmenter.process(frame(0.1)) == []
    snapshot = segmenter.active_snapshot()
    assert snapshot is not None
    assert snapshot.utterance_id == 1
    assert snapshot.duration_seconds == 0.1

    segments = []
    segments.extend(segmenter.process(frame(0.1)))
    segments.extend(segmenter.process(frame(0.0)))
    segments.extend(segmenter.process(frame(0.0)))

    assert len(segments) == 1
    assert segments[0].utterance_id == snapshot.utterance_id
    assert segments[0].closed_by == "pause"


def test_job_queue_prioritizes_finals_and_limits_partials() -> None:
    queue = TranscriptionJobQueue()
    partial = make_job("partial", utterance_id=1)
    second_partial = make_job("partial", utterance_id=1)
    final = make_job("final", utterance_id=1)

    assert queue.try_put_partial(partial)
    assert not queue.try_put_partial(second_partial)

    queue.put_final(final)
    assert queue.is_completed(1)
    assert not queue.try_put_partial(second_partial)

    first_job = queue.get()
    assert first_job is not None
    assert first_job.kind == "final"
    queue.task_done(first_job)

    second_job = queue.get()
    assert second_job is not None
    assert second_job.kind == "partial"
    queue.task_done(second_job)

    queue.put_stop()
    stop_job = queue.get()
    assert stop_job is None
    queue.task_done(stop_job)


def test_partial_tracker_emits_suffixes_and_suppresses_rewrites() -> None:
    tracker = PartialTextTracker()

    assert tracker.update(1, "Ich glaube das ist") == "Ich glaube das ist"
    assert tracker.update(1, "Ich glaube das ist eine gute Frage") == "eine gute Frage"
    assert tracker.update(1, "Das ist jetzt komplett anders") is None
    assert tracker.update(1, "Das ist jetzt komplett anders heute") == "heute"

    tracker.finish(1)
    assert tracker.update(1, "Neue Aussage") == "Neue Aussage"


def main() -> None:
    test_active_snapshot_and_final_segment()
    test_job_queue_prioritizes_finals_and_limits_partials()
    test_partial_tracker_emits_suffixes_and_suppresses_rewrites()
    print("partial pipeline tests passed")


if __name__ == "__main__":
    main()
