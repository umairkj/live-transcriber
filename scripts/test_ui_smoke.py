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
from live_transcriber.model_assets import AssetStatus


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
    settings_dir = tempfile.TemporaryDirectory()
    os.environ["LIVE_TRANSCRIBER_SETTINGS_FILE"] = str(Path(settings_dir.name) / "settings.ini")
    app = ui_app.QApplication.instance() or ui_app.QApplication([])
    ui_app.get_input_devices = lambda: [DeviceInfo(0, "BlackHole 2ch", 2, 48000.0)]

    minutes_calls: list[tuple[str, int]] = []
    window = ui_app.MainWindow(minutes_client_factory=lambda: FakeMinutesClient(minutes_calls))
    assert window.start_button.text() == "Start"
    assert not window.windowIcon().isNull()
    assert window.app_tabs.count() == 2
    assert window.app_tabs.tabText(0) == "Live"
    assert window.app_tabs.tabText(1) == "Settings"
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
    assert config.word_hint_model
    assert config.whisper_download_root == ui_app.whisper_download_root(window._app_data_dir)
    assert config.translation_target_language == "en"

    window._refresh_model_use_buttons(
        whisper_statuses={
            "base": AssetStatus("base", True, "Ready in app cache"),
            "small": AssetStatus("small", True, "Ready in app cache"),
            "medium": AssetStatus("medium", False, "Not downloaded"),
        },
        ollama_statuses={
            "llama3.1:latest": AssetStatus("llama3.1:latest", True, "Available"),
            "llama3.2:3b": AssetStatus("llama3.2:3b", True, "Available"),
            "mistral:7b": AssetStatus("mistral:7b", False, "Not pulled"),
        },
    )
    assert not window._model_use_buttons[("transcription", "base")].isEnabled()
    assert window._model_use_buttons[("transcription", "small")].isEnabled()
    assert not window._model_use_buttons[("transcription", "medium")].isEnabled()
    assert not window._model_use_buttons[("minutes", "llama3.1:latest")].isEnabled()
    assert window._model_use_buttons[("minutes", "llama3.2:3b")].isEnabled()
    assert not window._model_use_buttons[("word_hints", "mistral:7b")].isEnabled()

    task_key = ("whisper", "base")
    task_button = window._asset_buttons[task_key]
    assert task_button.text() == "Download"
    window._set_asset_button_loading(*task_key)
    assert not task_button.isEnabled()
    assert task_button.text() == "Downloading..."
    assert not task_button.icon().isNull()
    window._advance_spinner_icons()
    window._restore_asset_button(*task_key)
    assert task_button.isEnabled()
    assert task_button.text() == "Download"

    with tempfile.TemporaryDirectory() as temp_dir:
        spacy_worker = ui_app.AssetInstallWorker("spacy", "de_core_news_sm", Path(temp_dir))
        commands: list[list[str]] = []
        spacy_worker._run_subprocess = lambda command: commands.append(command)  # type: ignore[method-assign]
        spacy_worker._install_spacy()
        assert commands == [
            [sys.executable, "-m", "pip", "install", "spacy"],
            [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
        ]

    window._set_transcription_model("small")
    assert window.model_combo.currentText() == "small"
    assert window.settings.value("models/transcription") == "small"
    window._set_minutes_model("llama3.2:3b")
    assert window._selected_minutes_model() == "llama3.2:3b"
    window._set_word_hints_model("mistral:7b")
    assert window._selected_word_hints_model() == "mistral:7b"
    window._set_transcription_model("base")

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
    os.environ.pop("LIVE_TRANSCRIBER_SETTINGS_FILE", None)
    settings_dir.cleanup()
    print("ui smoke tests passed")


if __name__ == "__main__":
    main()
