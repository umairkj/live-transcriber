from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SpeechActivityEventType = Literal["speech_start", "speech", "speech_end"]


@dataclass(frozen=True)
class SpeechActivityEvent:
    event_type: SpeechActivityEventType
    start_time: float
    end_time: float
    rms: float
    reason: str = ""


class SpeechActivityDetector:
    """Stateful RMS-based utterance boundary detector."""

    def __init__(
        self,
        threshold: float,
        min_speech_seconds: float,
        pause_finalize_seconds: float,
        max_utterance_seconds: float,
    ) -> None:
        self.threshold = float(threshold)
        self.min_speech_seconds = max(0.0, float(min_speech_seconds))
        self.pause_finalize_seconds = max(0.0, float(pause_finalize_seconds))
        self.max_utterance_seconds = max(0.0, float(max_utterance_seconds))
        self.in_speech = False
        self.candidate_start_time: float | None = None
        self.utterance_start_time: float | None = None
        self.last_speech_end_time: float | None = None

    def observe_frame(self, start_time: float, end_time: float, rms: float) -> list[SpeechActivityEvent]:
        start_time = float(start_time)
        end_time = max(start_time, float(end_time))
        rms = float(rms)
        if end_time <= start_time:
            return []

        if rms >= self.threshold:
            return self._observe_speech(start_time, end_time, rms)

        return self._observe_silence(start_time, end_time, rms)

    def flush(self, end_time: float, rms: float = 0.0) -> list[SpeechActivityEvent]:
        if not self.in_speech or self.utterance_start_time is None:
            self._reset_candidate()
            return []

        final_end_time = self.last_speech_end_time or float(end_time)
        event = SpeechActivityEvent(
            event_type="speech_end",
            start_time=self.utterance_start_time,
            end_time=final_end_time,
            rms=float(rms),
            reason="flush",
        )
        self.reset()
        return [event]

    def reset(self) -> None:
        self.in_speech = False
        self._reset_candidate()
        self.utterance_start_time = None
        self.last_speech_end_time = None

    def _observe_speech(self, start_time: float, end_time: float, rms: float) -> list[SpeechActivityEvent]:
        events: list[SpeechActivityEvent] = []
        if self.candidate_start_time is None:
            self.candidate_start_time = start_time

        self.last_speech_end_time = end_time

        if not self.in_speech:
            candidate_duration = end_time - self.candidate_start_time
            if candidate_duration >= self.min_speech_seconds:
                self.in_speech = True
                self.utterance_start_time = self.candidate_start_time
                events.append(
                    SpeechActivityEvent(
                        event_type="speech_start",
                        start_time=self.utterance_start_time,
                        end_time=end_time,
                        rms=rms,
                    )
                )

        if self.in_speech and self.utterance_start_time is not None:
            events.append(
                SpeechActivityEvent(
                    event_type="speech",
                    start_time=self.utterance_start_time,
                    end_time=end_time,
                    rms=rms,
                )
            )
            if self._reached_max_utterance(end_time):
                events.append(
                    SpeechActivityEvent(
                        event_type="speech_end",
                        start_time=self.utterance_start_time,
                        end_time=end_time,
                        rms=rms,
                        reason="max_utterance_seconds",
                    )
                )
                self.reset()

        return events

    def _observe_silence(self, start_time: float, end_time: float, rms: float) -> list[SpeechActivityEvent]:
        if not self.in_speech:
            self._reset_candidate()
            return []

        if self.utterance_start_time is None or self.last_speech_end_time is None:
            self.reset()
            return []

        if end_time - self.last_speech_end_time < self.pause_finalize_seconds:
            return []

        event = SpeechActivityEvent(
            event_type="speech_end",
            start_time=self.utterance_start_time,
            end_time=self.last_speech_end_time,
            rms=rms,
            reason="pause",
        )
        self.reset()
        return [event]

    def _reached_max_utterance(self, end_time: float) -> bool:
        if self.max_utterance_seconds <= 0 or self.utterance_start_time is None:
            return False
        return end_time - self.utterance_start_time >= self.max_utterance_seconds

    def _reset_candidate(self) -> None:
        self.candidate_start_time = None
