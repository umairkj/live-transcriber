from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from queue import PriorityQueue
from threading import Lock
from typing import Literal

import numpy as np


JobKind = Literal["partial", "final"]


@dataclass(frozen=True)
class TranscriptionJob:
    """A Whisper job for either provisional or final transcript output."""

    kind: JobKind
    utterance_id: int
    audio: np.ndarray
    duration_seconds: float
    speech_seconds: float
    rms_peak: float
    closed_by: str | None = None


class TranscriptionJobQueue:
    """Priority queue that keeps final jobs ahead of provisional jobs."""

    def __init__(self) -> None:
        self._queue: PriorityQueue[tuple[int, int, TranscriptionJob | None]] = PriorityQueue()
        self._sequence = count()
        self._lock = Lock()
        self._outstanding_partials = 0
        self._outstanding_finals = 0
        self._completed_utterance_ids: set[int] = set()

    def put_final(self, job: TranscriptionJob) -> None:
        if job.kind != "final":
            raise ValueError("put_final requires a final job")

        with self._lock:
            self._completed_utterance_ids.add(job.utterance_id)
            self._outstanding_finals += 1
            self._put(priority=0, job=job)

    def try_put_partial(self, job: TranscriptionJob) -> bool:
        if job.kind != "partial":
            raise ValueError("try_put_partial requires a partial job")

        with self._lock:
            if (
                self._outstanding_partials > 0
                or self._outstanding_finals > 0
                or job.utterance_id in self._completed_utterance_ids
            ):
                return False

            self._outstanding_partials += 1
            self._put(priority=1, job=job)
            return True

    def can_accept_partial(self, utterance_id: int) -> bool:
        with self._lock:
            return (
                self._outstanding_partials == 0
                and self._outstanding_finals == 0
                and utterance_id not in self._completed_utterance_ids
            )

    def put_stop(self) -> None:
        self._put(priority=2, job=None)

    def get(self) -> TranscriptionJob | None:
        _priority, _sequence, job = self._queue.get()
        return job

    def task_done(self, job: TranscriptionJob | None) -> None:
        if job is not None:
            with self._lock:
                if job.kind == "partial":
                    self._outstanding_partials = max(0, self._outstanding_partials - 1)
                elif job.kind == "final":
                    self._outstanding_finals = max(0, self._outstanding_finals - 1)

        self._queue.task_done()

    def is_completed(self, utterance_id: int) -> bool:
        with self._lock:
            return utterance_id in self._completed_utterance_ids

    def _put(self, priority: int, job: TranscriptionJob | None) -> None:
        self._queue.put((priority, next(self._sequence), job))
