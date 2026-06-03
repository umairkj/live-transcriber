from __future__ import annotations

import html
import re
import sys
import unicodedata
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
from live_transcriber.speakers import DEFAULT_SPEAKER_BACKEND, SPEAKER_BACKENDS
from live_transcriber.translations import DEFAULT_TRANSLATION_TARGET_LANGUAGE, build_selective_translations

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
        QSplitter,
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

    @Slot()
    def reset_speakers(self) -> None:
        if self._session is not None:
            self._session.reset_speakers()


class MainWindow(QMainWindow):
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live Transcriber")
        self.resize(980, 680)

        self._thread: QThread | None = None
        self._worker: TranscriptionWorker | None = None
        self._is_running = False
        self._last_final_speaker: str | None = None
        self._vocabulary_items: dict[str, dict[str, str]] = {}

        self._build_ui()
        self.refresh_devices()
        self._set_running(False)
        self._update_speaker_controls()
        self._update_translation_controls()

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

        self.speaker_labels_check = QCheckBox("Speaker labels")
        self.speaker_labels_check.toggled.connect(self._update_speaker_controls)
        options_grid.addWidget(self.speaker_labels_check, 2, 0)

        self.reset_speakers_button = QPushButton("Reset speakers")
        self.reset_speakers_button.clicked.connect(self.reset_speakers)
        options_grid.addWidget(self.reset_speakers_button, 2, 1)

        self.speaker_backend_label = QLabel("Speaker backend")
        self.speaker_backend_combo = QComboBox()
        self.speaker_backend_combo.addItems(SPEAKER_BACKENDS)
        self.speaker_backend_combo.setCurrentText(DEFAULT_SPEAKER_BACKEND)
        options_grid.addWidget(self.speaker_backend_label, 2, 2)
        options_grid.addWidget(self.speaker_backend_combo, 2, 3)

        self.full_translation_check = QCheckBox("Full translation")
        self.full_translation_check.setChecked(True)
        self.full_translation_check.toggled.connect(self._update_translation_controls)
        options_grid.addWidget(self.full_translation_check, 3, 0)

        self.selective_translation_check = QCheckBox("Selective word hints")
        self.selective_translation_check.setChecked(True)
        self.selective_translation_check.toggled.connect(self._update_translation_controls)
        options_grid.addWidget(self.selective_translation_check, 3, 1)

        self.translation_target_label = QLabel("Translate to")
        self.translation_target_combo = QComboBox()
        self.translation_target_combo.addItem(DEFAULT_TRANSLATION_TARGET_LANGUAGE)
        self.translation_target_combo.setCurrentText(DEFAULT_TRANSLATION_TARGET_LANGUAGE)
        options_grid.addWidget(self.translation_target_label, 3, 2)
        options_grid.addWidget(self.translation_target_combo, 3, 3)

        timing_form = QFormLayout()
        timing_form.setHorizontalSpacing(10)
        timing_form.setVerticalSpacing(6)
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
        options_grid.addLayout(timing_form, 4, 0, 1, 4)
        layout.addLayout(options_grid)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Transcript appears here when listening starts.")
        self.transcript.setStyleSheet(
            """
            QTextEdit {
                font-size: 15px;
                line-height: 1.4;
                background: #f4f6f7;
                border: 1px solid #cfd6da;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #b9d7ff;
            }
            """
        )

        self.vocabulary = QTextEdit()
        self.vocabulary.setReadOnly(True)
        self.vocabulary.setPlaceholderText("New words appear here.")
        self.vocabulary.setMinimumWidth(230)
        self.vocabulary.setStyleSheet(
            """
            QTextEdit {
                font-size: 14px;
                background: #fffdf8;
                border: 1px solid #d9ccb4;
                border-radius: 8px;
                padding: 9px;
                selection-background-color: #ead7a8;
            }
            """
        )

        self.transcript_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.transcript_splitter.addWidget(self.transcript)
        vocab_column = QWidget()
        vocab_layout = QVBoxLayout(vocab_column)
        vocab_layout.setContentsMargins(0, 0, 0, 0)
        vocab_layout.setSpacing(6)
        vocab_title = QLabel("Vocabulary")
        vocab_title.setStyleSheet("font-weight: 700; color: #3d3321;")
        vocab_layout.addWidget(vocab_title)
        vocab_layout.addWidget(self.vocabulary, stretch=1)
        self.transcript_splitter.addWidget(vocab_column)
        self.transcript_splitter.setSizes([760, 260])
        layout.addWidget(self.transcript_splitter, stretch=1)

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
        if self.clear_on_start_check.isChecked():
            self.transcript.clear()
            self.vocabulary.clear()
            self._vocabulary_items.clear()
            self._last_final_speaker = None
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
        partial_text = self._render_partial_text(text)
        self._append_html(
            '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 6px 0 8px 0;">'
            "<tr>"
            '<td width="24%"></td>'
            '<td style="background-color: #e6f2ff; border: 1px solid #a8cfee; padding: 8px 10px;">'
            '<div style="color: #2f6f9f; font-size: 11px; font-weight: 700; letter-spacing: 0; margin-bottom: 3px;">'
            "live"
            "</div>"
            f"{partial_text}"
            "</td>"
            "</tr>"
            "</table>"
        )

    @Slot(str, object)
    def append_final(self, text: str, record: object | None = None) -> None:
        record_data = record if isinstance(record, dict) else {}
        raw_text = str(record_data.get("text") or text)
        speaker_label = record_data.get("speaker_label")
        timestamp = str(record_data.get("timestamp") or "")
        selective_translations = record_data.get("selective_translations") or []
        translation_text = str(record_data.get("translation_text") or "")
        margin_top = "3px" if speaker_label and speaker_label == self._last_final_speaker else "10px"
        self._last_final_speaker = str(speaker_label) if speaker_label else None

        if speaker_label:
            marker = (
                '<td width="34" align="center" valign="top" '
                'style="background-color: #1f4d3a; color: #ffffff; font-size: 14px; '
                'font-weight: 800; padding: 8px 6px;">'
                f"{html.escape(str(speaker_label))}:"
                "</td>"
            )
        else:
            marker = (
                '<td width="6" style="background-color: #3f5f6f; padding: 0;">'
                "&nbsp;"
                "</td>"
            )

        timestamp_html = ""
        if timestamp:
            timestamp_html = (
                '<div style="color: #8a969c; font-size: 11px; margin-bottom: 3px;">'
                f"{html.escape(timestamp)}"
                "</div>"
            )

        word_hints_html = self._render_word_hints(raw_text, selective_translations)
        self._remember_vocabulary(selective_translations)
        translation_html = self._render_full_translation(
            translation_text,
            str(record_data.get("translation_target_language") or DEFAULT_TRANSLATION_TARGET_LANGUAGE),
        )

        self._append_html(
            f'<table width="100%" cellspacing="0" cellpadding="0" style="margin: {margin_top} 0 8px 0;">'
            "<tr>"
            f"{marker}"
            '<td style="background-color: #ffffff; border: 1px solid #d8dee2; '
            'border-left: 0; padding: 8px 10px;">'
            f"{timestamp_html}"
            f"{word_hints_html}"
            f"{translation_html}"
            "</td>"
            "</tr>"
            "</table>"
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

    @Slot()
    def reset_speakers(self) -> None:
        if self._worker is not None:
            self._worker.reset_speakers()
            self.set_status("Speaker profiles reset")
        else:
            self.set_status("No active speaker profiles")

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
            speaker_labels=self.speaker_labels_check.isChecked(),
            speaker_backend=self.speaker_backend_combo.currentText(),
            full_translation=self.full_translation_check.isChecked(),
            selective_translation=self.selective_translation_check.isChecked(),
            translation_target_language=self.translation_target_combo.currentText(),
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
            self.speaker_labels_check,
            self.speaker_backend_combo,
            self.full_translation_check,
            self.selective_translation_check,
            self.translation_target_combo,
            self.partial_seconds_spin,
            self.pause_seconds_spin,
        ):
            widget.setEnabled(not is_running)
        self.reset_speakers_button.setEnabled(self.speaker_labels_check.isChecked())

    @Slot(bool)
    def _update_speaker_controls(self, checked: bool | None = None) -> None:
        del checked
        enabled = self.speaker_labels_check.isChecked()
        self.speaker_backend_label.setVisible(enabled)
        self.speaker_backend_combo.setVisible(enabled)
        self.reset_speakers_button.setEnabled(enabled)

    @Slot(bool)
    def _update_translation_controls(self, checked: bool | None = None) -> None:
        del checked
        enabled = self.full_translation_check.isChecked() or self.selective_translation_check.isChecked()
        self.translation_target_label.setVisible(enabled)
        self.translation_target_combo.setVisible(enabled)

    def _render_partial_text(self, text: str) -> str:
        if not self.selective_translation_check.isChecked():
            return self._render_plain_transcript_text(text, text_color="#2e5268")

        hints = build_selective_translations(
            text,
            target_language=self.translation_target_combo.currentText(),
        )
        return self._render_word_hints(text, hints, text_color="#244f68")

    def _render_word_hints(self, text: str, items: object, text_color: str = "#111820") -> str:
        if not isinstance(items, list):
            return self._render_plain_transcript_text(text, text_color=text_color)

        hints_by_source: dict[str, dict[str, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            translation = str(item.get("translation") or "").strip()
            kind = str(item.get("kind") or "phrase")
            if kind not in {"noun", "verb"} or not source or not translation:
                continue
            hints_by_source.setdefault(self._translation_key(source), {"translation": translation, "kind": kind})

        words = self._word_cells_from_text(text)
        if not words or not hints_by_source:
            return self._render_plain_transcript_text(text, text_color=text_color)

        rows: list[str] = []
        for start in range(0, len(words), 8):
            chunk = words[start : start + 8]
            gloss_cells: list[str] = []
            word_cells: list[str] = []
            for word, display in chunk:
                hint = hints_by_source.get(self._translation_key(word))
                if hint is None:
                    gloss_cells.append('<td style="padding: 0 8px 1px 0;">&nbsp;</td>')
                else:
                    background, _border, hint_text_color = self._word_hint_colors(hint["kind"])
                    gloss_cells.append(
                        '<td align="center" style="padding: 0 8px 1px 0; white-space: nowrap;">'
                        f'<span style="background-color: {background}; color: {hint_text_color}; '
                        'font-size: 11px; font-weight: 800; padding: 1px 4px;">'
                        f"{html.escape(hint['translation'])}"
                        "</span>"
                        "</td>"
                    )
                word_cells.append(
                    '<td align="center" style="padding: 0 8px 7px 0; white-space: nowrap; '
                    f"color: {text_color}; font-size: 15px;\">"
                    f"{html.escape(display)}"
                    "</td>"
                )
            rows.append("<tr>" + "".join(gloss_cells) + "</tr><tr>" + "".join(word_cells) + "</tr>")

        return '<table cellspacing="0" cellpadding="0" style="margin: 0 0 3px 0;">' + "".join(rows) + "</table>"

    def _render_plain_transcript_text(self, text: str, text_color: str = "#111820") -> str:
        return (
            f'<div style="color: {text_color}; font-size: 15px; line-height: 1.45;">'
            f"{html.escape(text)}"
            "</div>"
        )

    def _render_full_translation(self, text: str, target_language: str) -> str:
        if not text:
            return ""

        return (
            '<div style="margin-top: 8px; background-color: #f7f1df; border: 1px solid #e0c98d; '
            'border-left: 4px solid #b9821f; padding: 7px 9px;">'
            f'<span style="color: #7b5716; font-size: 11px; font-weight: 800;">{html.escape(target_language.upper())}</span>'
            f'<span style="color: #3c2f16;"> {html.escape(text)}</span>'
            "</div>"
        )

    def _word_hint_colors(self, kind: str) -> tuple[str, str, str]:
        if kind == "verb":
            return "#e8f0ff", "#b7c9ed", "#214f8f"
        if kind == "noun":
            return "#f8ecd1", "#dfc17f", "#604715"
        if kind == "adjective":
            return "#e7f3e8", "#b9d5bd", "#295a31"
        if kind == "adverb":
            return "#f0e9f7", "#cdb8df", "#5a3f70"
        return "#eef0f2", "#cbd0d5", "#42484d"

    def _word_cells_from_text(self, text: str) -> list[tuple[str, str]]:
        cells: list[list[str]] = []
        for token in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
            if re.match(r"\w+", token, flags=re.UNICODE):
                cells.append([token, token])
            elif cells:
                cells[-1][1] += token

        return [(word, display) for word, display in cells]

    def _remember_vocabulary(self, items: object) -> None:
        if not isinstance(items, list):
            return

        changed = False
        for item in items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            translation = str(item.get("translation") or "").strip()
            kind = str(item.get("kind") or "phrase")
            if kind not in {"noun", "verb"} or not source or not translation:
                continue

            key = self._translation_key(source)
            if key in self._vocabulary_items:
                continue

            self._vocabulary_items[key] = {
                "source": source,
                "translation": translation,
                "kind": kind,
            }
            changed = True

        if changed:
            self._render_vocabulary()

    def _render_vocabulary(self) -> None:
        parts = ['<table width="100%" cellspacing="0" cellpadding="5">']
        for item in self._vocabulary_items.values():
            background, border, text_color = self._word_hint_colors(item["kind"])
            parts.append(
                "<tr>"
                f'<td style="background-color: {background}; border: 1px solid {border};">'
                f'<span style="color: {text_color}; font-size: 16px; font-weight: 800;">'
                f"{html.escape(item['source'])}"
                "</span><br>"
                f'<span style="color: #2e2a22; font-size: 13px;">{html.escape(item["translation"])}</span><br>'
                f'<span style="color: #81745e; font-size: 10px;">{html.escape(item["kind"])}</span>'
                "</td>"
                "</tr>"
            )
        parts.append("</table>")

        self.vocabulary.setHtml("".join(parts))
        cursor = self.vocabulary.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.vocabulary.setTextCursor(cursor)

    def _translation_key(self, word: str) -> str:
        word = word.casefold().replace("\u00df", "ss")
        normalized = unicodedata.normalize("NFKD", word)
        return "".join(char for char in normalized if not unicodedata.combining(char))

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
