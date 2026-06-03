from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

from live_transcriber.config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_JSONL_OUTPUT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_PARTIAL_SECONDS,
    DEFAULT_PAUSE_SECONDS,
)
from live_transcriber.device_utils import DeviceInfo, get_input_devices
from live_transcriber.session import LiveTranscriberCallbacks, LiveTranscriberConfig, LiveTranscriberSession

try:
    from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStyle,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit("PySide6 is not installed. Run: venv/bin/python -m pip install -r requirements.txt") from exc


class TranscriptionWorker(QObject):
    status_changed = Signal(str)
    partial_text = Signal(str)
    final_text = Signal(str, object)
    error = Signal(str)
    finished = Signal(int)

    def __init__(self, config: LiveTranscriberConfig) -> None:
        super().__init__()
        self._config = config
        self._session: LiveTranscriberSession | None = None
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        self._session = LiveTranscriberSession()
        if self._stop_requested:
            self._session.stop()

        callbacks = LiveTranscriberCallbacks(
            on_status=self.status_changed.emit,
            on_partial_text=self.partial_text.emit,
            on_final_text=lambda text, record: self.final_text.emit(text, record),
            on_error=self.error.emit,
        )
        exit_code = self._session.run_blocking(self._config, callbacks)
        self.finished.emit(exit_code)

    @Slot()
    def stop(self) -> None:
        self._stop_requested = True
        if self._session is not None:
            self._session.stop()


class MainWindow(QMainWindow):
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live Transcriber")
        self.resize(980, 680)

        self._thread: QThread | None = None
        self._worker: TranscriptionWorker | None = None
        self._is_running = False

        self._build_ui()
        self.refresh_devices()
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.clicked.connect(self.toggle_transcription)
        top_row.addWidget(self.start_button)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(320)
        top_row.addWidget(QLabel("Input"))
        top_row.addWidget(self.device_combo, stretch=1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_devices)
        top_row.addWidget(self.refresh_button)
        layout.addLayout(top_row)

        options_grid = QGridLayout()
        options_grid.setHorizontalSpacing(16)
        options_grid.setVerticalSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium"])
        self.model_combo.setCurrentText(DEFAULT_MODEL)
        options_grid.addWidget(QLabel("Model"), 0, 0)
        options_grid.addWidget(self.model_combo, 0, 1)

        self.language_combo = QComboBox()
        self.language_combo.setEditable(True)
        self.language_combo.addItem("de")
        self.language_combo.addItem("en")
        self.language_combo.addItem("auto")
        self.language_combo.setCurrentText("de")
        options_grid.addWidget(QLabel("Language"), 0, 2)
        options_grid.addWidget(self.language_combo, 0, 3)

        self.show_partials_check = QCheckBox("Show partials")
        self.show_partials_check.setChecked(True)
        self.save_transcript_check = QCheckBox("Save transcript")
        self.save_transcript_check.setChecked(True)
        self.clear_on_start_check = QCheckBox("Clear transcript on start")
        self.always_on_top_check = QCheckBox("Always on top")
        self.always_on_top_check.toggled.connect(self.set_always_on_top)
        options_grid.addWidget(self.show_partials_check, 1, 0)
        options_grid.addWidget(self.save_transcript_check, 1, 1)
        options_grid.addWidget(self.clear_on_start_check, 1, 2)
        options_grid.addWidget(self.always_on_top_check, 1, 3)

        timing_form = QFormLayout()
        self.partial_seconds_spin = QDoubleSpinBox()
        self.partial_seconds_spin.setRange(0.0, 10.0)
        self.partial_seconds_spin.setSingleStep(0.25)
        self.partial_seconds_spin.setDecimals(2)
        self.partial_seconds_spin.setValue(DEFAULT_PARTIAL_SECONDS)
        timing_form.addRow("Partial seconds", self.partial_seconds_spin)

        self.pause_seconds_spin = QDoubleSpinBox()
        self.pause_seconds_spin.setRange(0.2, 5.0)
        self.pause_seconds_spin.setSingleStep(0.05)
        self.pause_seconds_spin.setDecimals(2)
        self.pause_seconds_spin.setValue(DEFAULT_PAUSE_SECONDS)
        timing_form.addRow("Pause seconds", self.pause_seconds_spin)
        options_grid.addLayout(timing_form, 2, 0, 1, 4)
        layout.addLayout(options_grid)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Transcript appears here when listening starts.")
        self.transcript.setStyleSheet(
            "QTextEdit { font-size: 15px; line-height: 1.35; background: #fbfbfb; border: 1px solid #d7d7d7; }"
        )
        layout.addWidget(self.transcript, stretch=1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Idle")
        footer.addWidget(self.status_label, stretch=1)
        self.output_label = QLabel(str(DEFAULT_OUTPUT))
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(self.output_label)
        layout.addLayout(footer)

        self.setCentralWidget(root)

    @Slot()
    def refresh_devices(self) -> None:
        current_device = self.device_combo.currentData()
        self.device_combo.clear()
        self.device_combo.addItem("System Default Input", None)

        try:
            devices = get_input_devices()
        except RuntimeError as exc:
            self.set_status(str(exc))
            return

        blackhole_index: int | None = None
        for device in devices:
            label = self._device_label(device)
            self.device_combo.addItem(label, device.id)
            if blackhole_index is None and "blackhole" in device.name.casefold():
                blackhole_index = self.device_combo.count() - 1

        if current_device is not None:
            index = self.device_combo.findData(current_device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
                return

        if blackhole_index is not None:
            self.device_combo.setCurrentIndex(blackhole_index)

    @Slot()
    def toggle_transcription(self) -> None:
        if self._is_running:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self) -> None:
        config = self._build_config()
        self.transcript.clear()
        self.output_label.setText(str(config.text_path or "Transcript saving disabled"))

        self._thread = QThread(self)
        self._worker = TranscriptionWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self.stop_requested.connect(self._worker.stop)
        self._worker.status_changed.connect(self.set_status)
        self._worker.partial_text.connect(self.append_partial)
        self._worker.final_text.connect(self.append_final)
        self._worker.error.connect(self.show_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._set_running(True)
        self._thread.start()

    def stop_transcription(self) -> None:
        self.set_status("Stopping")
        self.stop_requested.emit()
        self.start_button.setEnabled(False)

    @Slot(str)
    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.set_status(message)
        QMessageBox.warning(self, "Live Transcriber", message)

    @Slot(str)
    def append_partial(self, text: str) -> None:
        if not self.show_partials_check.isChecked():
            return
        self._append_html(
            f'<p style="margin: 6px 0; color: #707070;"><span style="font-weight: 600;">partial</span> '
            f"{html.escape(text)}</p>"
        )

    @Slot(str, object)
    def append_final(self, text: str, record: object | None = None) -> None:
        self._append_html(
            f'<p style="margin: 9px 0; color: #111111;"><span style="font-weight: 700;">final</span> '
            f"{html.escape(text)}</p>"
        )

    @Slot(int)
    def _on_worker_finished(self, exit_code: int) -> None:
        if self._worker is not None:
            try:
                self.stop_requested.disconnect(self._worker.stop)
            except (RuntimeError, TypeError):
                pass
        if exit_code == 0:
            self.set_status("Idle")
        self._set_running(False)
        self._worker = None
        self._thread = None

    def set_always_on_top(self, enabled: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.show()

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._is_running:
            self.stop_transcription()
            if self._thread is not None:
                if not self._thread.wait(3000):
                    self.set_status("Still stopping...")
                    event.ignore()
                    return
        super().closeEvent(event)

    def _build_config(self) -> LiveTranscriberConfig:
        language = self.language_combo.currentText().strip() or None
        if language == "auto":
            language = None

        save_transcript = self.save_transcript_check.isChecked()
        partial_seconds = self.partial_seconds_spin.value() if self.show_partials_check.isChecked() else 0.0

        return LiveTranscriberConfig(
            model=self.model_combo.currentText(),
            device=self.device_combo.currentData(),
            language=language,
            text_path=Path(DEFAULT_OUTPUT) if save_transcript else None,
            jsonl_path=Path(DEFAULT_JSONL_OUTPUT) if save_transcript else None,
            save_transcript=save_transcript,
            overwrite=self.clear_on_start_check.isChecked(),
            pause_seconds=self.pause_seconds_spin.value(),
            partial_seconds=partial_seconds,
            beam_size=DEFAULT_BEAM_SIZE,
        )

    def _set_running(self, is_running: bool) -> None:
        self._is_running = is_running
        self.start_button.setEnabled(True)
        if is_running:
            self.start_button.setText("Stop")
            self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        else:
            self.start_button.setText("Start")
            self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        for widget in (
            self.device_combo,
            self.refresh_button,
            self.model_combo,
            self.language_combo,
            self.show_partials_check,
            self.save_transcript_check,
            self.clear_on_start_check,
            self.partial_seconds_spin,
            self.pause_seconds_spin,
        ):
            widget.setEnabled(not is_running)

    def _append_html(self, value: str) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(value)
        cursor.insertBlock()
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def _device_label(self, device: DeviceInfo) -> str:
        rate = "unknown"
        if device.default_sample_rate is not None:
            rate = f"{device.default_sample_rate:.0f} Hz"
        return f"{device.id}: {device.name} ({device.input_channels} ch, {rate})"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Live Transcriber")
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
