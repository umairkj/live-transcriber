from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4


UtteranceStatus = Literal["hypothesis", "final", "corrected"]
TranscriptEventType = Literal["hypothesis", "final", "correction"]


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class UtteranceState:
    start_time: float
    end_time: float
    utterance_id: str = field(default_factory=lambda: str(uuid4()))
    hypothesis_text: str = ""
    final_text: str = ""
    final_timestamp: str = ""
    updated_timestamp: str = ""
    revision: int = 0
    is_final: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "hypothesis_text": self.hypothesis_text,
            "final_text": self.final_text,
            "final_timestamp": self.final_timestamp,
            "updated_timestamp": self.updated_timestamp,
            "revision": self.revision,
            "is_final": self.is_final,
        }


@dataclass
class TranscriptEvent:
    event_type: TranscriptEventType
    utterance: UtteranceState
    text: str
    timestamp: str = field(default_factory=current_timestamp)
    previous_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {
            "timestamp": self.timestamp,
            "type": self.event_type,
            "text": self.text,
            "previous_text": self.previous_text,
            **self.utterance.to_record(),
        }
        record.update(self.metadata)
        return record


class TranscriptSession:
    """Mutable transcript state keyed by utterance for final transcript rewrites."""

    def __init__(self) -> None:
        self.utterances: list[UtteranceState] = []

    def upsert_final(self, utterance: UtteranceState) -> TranscriptEvent:
        timestamp = current_timestamp()
        previous = self._find(utterance.utterance_id)
        if previous is None:
            utterance.is_final = True
            utterance.final_timestamp = timestamp
            utterance.updated_timestamp = timestamp
            self.utterances.append(utterance)
            return TranscriptEvent(
                event_type="final",
                utterance=utterance,
                text=utterance.final_text,
                timestamp=timestamp,
            )

        previous_text = previous.final_text
        previous.end_time = utterance.end_time
        previous.hypothesis_text = utterance.hypothesis_text
        previous.final_text = utterance.final_text
        previous.revision += 1
        previous.is_final = True
        if not previous.final_timestamp:
            previous.final_timestamp = timestamp
        previous.updated_timestamp = timestamp
        event_type: TranscriptEventType = "correction" if previous_text != previous.final_text else "final"
        return TranscriptEvent(
            event_type=event_type,
            utterance=previous,
            text=previous.final_text,
            previous_text=previous_text,
            timestamp=timestamp,
        )

    def final_lines(self) -> list[tuple[str, str]]:
        return [
            (utterance.final_timestamp or current_timestamp(), utterance.final_text)
            for utterance in self.utterances
            if utterance.is_final and utterance.final_text.strip()
        ]

    def _find(self, utterance_id: str) -> UtteranceState | None:
        for utterance in self.utterances:
            if utterance.utterance_id == utterance_id:
                return utterance
        return None
