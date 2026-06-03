from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from live_transcriber.device_utils import DeviceInfo

import ui_app


class FakeWorker:
    def __init__(self) -> None:
        self.reset_called = False

    def reset_speakers(self) -> None:
        self.reset_called = True


def main() -> None:
    app = ui_app.QApplication.instance() or ui_app.QApplication([])
    ui_app.get_input_devices = lambda: [DeviceInfo(0, "BlackHole 2ch", 2, 48000.0)]

    window = ui_app.MainWindow()
    assert window.start_button.text() == "Start"
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

    window.append_final(
        "A: Ich putze die Zaehne.",
        {
            "text": "Ich putze die Zaehne.",
            "speaker_label": "A",
            "translation_text": "I brush my teeth.",
            "translation_target_language": "en",
            "selective_translations": [
                {"source": "putze", "translation": "clean", "kind": "verb"},
                {"source": "Zaehne", "translation": "teeth", "kind": "noun"},
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
    assert "clean" in vocabulary_rendered

    fake_worker = FakeWorker()
    window._worker = fake_worker
    window.reset_speakers()
    assert fake_worker.reset_called

    window.close()
    app.quit()
    print("ui smoke tests passed")


if __name__ == "__main__":
    main()
