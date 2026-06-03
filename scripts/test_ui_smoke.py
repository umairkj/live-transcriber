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

    window.speaker_labels_check.setChecked(True)
    config = window._build_config()
    assert config.speaker_labels is True
    assert config.speaker_backend == "local"

    window._set_running(True)
    assert window.start_button.text() == "Stop"
    window._set_running(False)
    assert window.start_button.text() == "Start"

    window.append_partial("ich putze")
    window.append_final("A: Ich putze die Zaehne.", {"text": "Ich putze die Zaehne.", "speaker_label": "A"})
    rendered = window.transcript.toHtml()
    assert "partial" in rendered
    assert "A:" in rendered
    assert "Ich putze die Zaehne." in rendered

    fake_worker = FakeWorker()
    window._worker = fake_worker
    window.reset_speakers()
    assert fake_worker.reset_called

    window.close()
    app.quit()
    print("ui smoke tests passed")


if __name__ == "__main__":
    main()
