from __future__ import annotations

import sys
from datetime import datetime
from typing import TextIO

from live_transcriber.sentence_utils import clean_whisper_artifacts


class LiveTranscriptDisplay:
    """Render a live-updating transcript line, then commit it after a pause."""

    def __init__(
        self,
        enabled: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.stream = stream or sys.stdout
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._line_open = False
        self._current_text = ""
        self._current_timestamp = ""

    def update(self, text: str) -> None:
        if not self.enabled:
            return

        text = clean_whisper_artifacts(text)
        if not text or text == self._current_text:
            return

        if not self._current_timestamp:
            self._current_timestamp = self._timestamp()
        self._current_text = text

        line = self._format_line(self._current_timestamp, text)
        if not self._is_tty:
            self.stream.write(f"{line}\n")
            self.stream.flush()
            return

        self.stream.write(f"\r\033[2K{line}")
        self.stream.flush()
        self._line_open = True

    def commit(self, text: str, timestamp: str | None = None) -> None:
        if not self.enabled:
            return

        text = clean_whisper_artifacts(text) or self._current_text
        if not text:
            self.finish()
            return

        display_timestamp = self._display_timestamp(timestamp) or self._current_timestamp or self._timestamp()
        line = self._format_line(display_timestamp, text)
        if self._is_tty:
            self.stream.write(f"\r\033[2K{line}\n")
        elif text != self._current_text:
            self.stream.write(f"{line}\n")
        self.stream.flush()
        self._reset_line()

    def finish(self) -> None:
        if self.enabled and self._is_tty and self._line_open:
            self.stream.write("\n")
            self.stream.flush()
        self._reset_line()

    def _reset_line(self) -> None:
        self._line_open = False
        self._current_text = ""
        self._current_timestamp = ""

    def _format_line(self, timestamp: str, text: str) -> str:
        return f"[{timestamp}] {text}"

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _display_timestamp(self, timestamp: str | None) -> str:
        if not timestamp:
            return ""
        return timestamp.split()[-1]
