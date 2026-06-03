from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from live_transcriber.device_utils import DeviceInfo

import ui_app


class FakeWorker:
    def __init__(self) -> None:
        self.reset_called = False
        self.stop_called = False

    def reset_speakers(self) -> None:
        self.reset_called = True

    def stop(self) -> None:
        self.stop_called = True


class FakeMinutesClient:
    def __init__(self, calls: list[tuple[str, int]]) -> None:
        self._calls = calls

    def summarize(self, existing_minutes: str, events: list[dict[str, object]]) -> str:
        self._calls.append((existing_minutes, len(events)))
        return (
            "Decisions\n"
            "- [2026-06-03 09:00] Keep the morning routine.\n"
            "Action Items\n"
            "- [2026-06-03 09:00] Practice new vocabulary.\n"
            "Open Questions\n"
            "- [2026-06-03 09:00] None yet.\n"
            "Important Points\n"
            "- [2026-06-03 09:00] Teeth and water were discussed."
        )


class FailingMinutesClient:
    def summarize(self, existing_minutes: str, events: list[dict[str, object]]) -> str:
        del existing_minutes, events
        raise ui_app.MinutesError("Ollama is not running. Start it with: ollama serve")


def wait_for(app: ui_app.QApplication, condition, timeout: float = 3.0) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached in time")


def main() -> None:
    app = ui_app.QApplication.instance() or ui_app.QApplication([])
    ui_app.get_input_devices = lambda: [DeviceInfo(0, "BlackHole 2ch", 2, 48000.0)]

    minutes_calls: list[tuple[str, int]] = []
    window = ui_app.MainWindow(minutes_client_factory=lambda: FakeMinutesClient(minutes_calls))
    assert window.start_button.text() == "Start"
    assert window.tabs.count() == 1
    assert window._current_document().splitter.count() == 3
    assert window.device_combo.currentData() == 0
    assert window.model_combo.currentText() == "base"
    assert window.language_combo.currentText() == "de"
    assert not window.speaker_labels_check.isChecked()
    assert window.full_translation_check.isChecked()
    assert window.selective_translation_check.isChecked()

    window.speaker_labels_check.setChecked(True)
    config = window._build_config()
    assert config.speaker_labels is True
    assert config.speaker_backend == "local"
    assert config.full_translation is True
    assert config.selective_translation is True
    assert config.selective_translation_backend == "ollama"
    assert config.translation_target_language == "en"

    window._set_running(True)
    assert window.start_button.text() == "Stop"
    window._set_running(False)
    assert window.start_button.text() == "Start"

    window.append_partial("ich putze")
    partial_rendered = window.transcript.toHtml()
    assert "live" in partial_rendered
    assert "clean" in partial_rendered
    assert "putze" in partial_rendered
    assert "putze" not in window.vocabulary.toHtml()
    assert minutes_calls == []

    window.append_final(
        "A: Ich putze die Zaehne.",
        {
            "text": "Ich putze die Zaehne.",
            "speaker_label": "A",
            "translation_text": "I brush my teeth.",
            "translation_target_language": "en",
            "selective_translations": [
                {"source": "putze", "translation": "clean", "kind": "verb"},
                {"source": "Zaehne", "translation": "teeth", "kind": "noun", "article": "die"},
            ],
        },
    )
    rendered = window.transcript.toHtml()
    assert "live" in rendered
    assert "A:" in rendered
    assert "Ich" in rendered
    assert "putze" in rendered
    assert "Zaehne." in rendered
    assert "I brush my teeth." in rendered
    assert "clean" in rendered
    assert "teeth" in rendered
    vocabulary_rendered = window.vocabulary.toHtml()
    assert "putze" in vocabulary_rendered
    assert "Zaehne" in vocabulary_rendered
    assert "die Zaehne" in vocabulary_rendered
    assert "clean" in vocabulary_rendered
    assert minutes_calls == []

    window.append_final(
        "A: Ich trinke Wasser.",
        {
            "timestamp": "2026-06-03 09:00:01",
            "text": "Ich trinke Wasser.",
            "speaker_label": "A",
            "translation_text": "I drink water.",
            "translation_target_language": "en",
        },
    )
    assert minutes_calls == []

    window.append_final(
        "B: Dann gehe ich zur Arbeit.",
        {
            "timestamp": "2026-06-03 09:00:02",
            "text": "Dann gehe ich zur Arbeit.",
            "speaker_label": "B",
            "translation_text": "Then I go to work.",
            "translation_target_language": "en",
        },
    )
    wait_for(app, lambda: len(minutes_calls) == 1 and window._minutes_thread is None)
    assert minutes_calls[0][1] == 3
    assert "Decisions" in window.minutes.toPlainText()
    assert "Keep the morning routine" in window.minutes.toPlainText()
    assert window._current_document().last_minutes_event_index == 3

    with tempfile.TemporaryDirectory() as temp_dir:
        document = window._current_document()
        document.title = "Morning Routine"
        document.path = Path(temp_dir) / "Morning-Routine.trans"
        window._save_document(document)
        assert document.path.exists()

        loaded = window._load_document(document.path)
        assert loaded.title == "Morning Routine"
        assert len(loaded.events) == 4
        assert "zaehne" in loaded.vocabulary_items
        assert loaded.vocabulary_items["zaehne"]["article"] == "die"
        assert "die Zaehne" in loaded.vocabulary.toHtml()
        assert loaded.last_minutes_event_index == 3
        assert "Keep the morning routine" in loaded.minutes.toPlainText()

    document = window._current_document()
    document.minutes_text = (
        "*Meeting Minutes**\n"
        "**Important Points:**\n"
        "+ The E-Podcast Ausgabe N193 is a summer interview.\n"
        "**Decisions:** None\n"
        "**Action Items:** None\n"
        "**Open Questions:** None"
    )
    window._render_minutes(document)
    rendered_minutes = window.minutes.toPlainText()
    assert "Meeting Minutes" in rendered_minutes
    assert "Important Points" in rendered_minutes
    assert "The E-Podcast Ausgabe N193" in rendered_minutes
    assert "**" not in rendered_minutes
    assert "+ The" not in rendered_minutes

    window.new_transcript()
    assert window.tabs.count() == 2

    window._minutes_client_factory = FailingMinutesClient
    window.append_final("Test one.", {"timestamp": "2026-06-03 09:01:00", "text": "Test one."})
    window.append_final("Test two.", {"timestamp": "2026-06-03 09:01:01", "text": "Test two."})
    window.append_final("Test three.", {"timestamp": "2026-06-03 09:01:02", "text": "Test three."})
    failure_document = window._current_document()
    wait_for(app, lambda: window._minutes_thread is None and "Ollama unavailable" in failure_document.minutes_status.text())
    assert "Start it with: ollama serve" in failure_document.minutes_status.text()
    failure_document.dirty = False

    fake_worker = FakeWorker()
    window._worker = fake_worker
    window.stop_transcription()
    assert fake_worker.stop_called
    window._set_running(False)
    window.reset_speakers()
    assert fake_worker.reset_called

    window.close()
    app.quit()
    print("ui smoke tests passed")


if __name__ == "__main__":
    main()
