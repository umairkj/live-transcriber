from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TranscriptWriter:
    """Append transcript text and JSONL records with immediate flushes."""

    def __init__(
        self,
        text_path: Path,
        jsonl_path: Path | None = None,
        overwrite: bool = False,
    ) -> None:
        self.text_path = Path(text_path)
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.initial_text_lines: list[str] = []

        self.text_path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and self.text_path.exists():
            self.initial_text_lines = self.text_path.read_text(encoding="utf-8").splitlines()
        self.text_path.touch(exist_ok=True)
        if overwrite:
            self.text_path.write_text("", encoding="utf-8")

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.touch(exist_ok=True)
            if overwrite:
                self.jsonl_path.write_text("", encoding="utf-8")

    def write_text(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        timestamp = metadata.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.text_path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {text.strip()}\n")
            file.flush()

    def write_jsonl(self, record: dict[str, Any]) -> None:
        if self.jsonl_path is None:
            return

        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

    def rewrite_text_lines(self, records: list[tuple[str, str]]) -> None:
        lines = list(self.initial_text_lines)
        for timestamp, text in records:
            clean_text = str(text).strip()
            if clean_text:
                lines.append(f"[{timestamp}] {clean_text}")

        temp_path = self.text_path.with_suffix(f"{self.text_path.suffix}.tmp")
        temp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        temp_path.replace(self.text_path)

    def write_record(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.write_text(str(record.get("text", "")), metadata=record)
        self.write_jsonl(record)
