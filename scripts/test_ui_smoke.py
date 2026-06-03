from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from live_transcriber.device_utils import DeviceInfo

import ui_app


def main() -> None:
    app = ui_app.QApplication.instance() or ui_app.QApplication([])
    ui_app.get_input_devices = lambda: [DeviceInfo(0, "BlackHole 2ch", 2, 48000.0)]

    window = ui_app.MainWindow()
    assert window.start_button.text() == "Start"
    assert window.device_combo.currentData() == 0
    assert window.model_combo.currentText() == "base"
    assert window.language_combo.currentText() == "de"

    window._set_running(True)
    assert window.start_button.text() == "Stop"
    window._set_running(False)
    assert window.start_button.text() == "Start"

    window.append_partial("ich putze")
    window.append_final("Ich putze die Zaehne.", {})
    rendered = window.transcript.toHtml()
    assert "partial" in rendered
    assert "final" in rendered
    assert "Ich putze die Zaehne." in rendered

    window.close()
    app.quit()
    print("ui smoke tests passed")


if __name__ == "__main__":
    main()
