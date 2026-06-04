from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable
from urllib import error, request

from live_transcriber.audio import audio_signal_level
from live_transcriber.config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_JSONL_OUTPUT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_PARTIAL_SECONDS,
    DEFAULT_PAUSE_SECONDS,
)
from live_transcriber.device_utils import DeviceInfo, get_input_devices
from live_transcriber.minutes import DEFAULT_MINUTES_MODEL, MINUTES_UPDATE_FINAL_COUNT, MinutesError, OllamaMinutesClient
from live_transcriber.model_assets import (
    OLLAMA_MODELS,
    WHISPER_MODELS,
    AssetStatus,
    blackhole_status,
    default_app_data_dir,
    dictionary_path,
    dictionary_status,
    ensure_app_dirs,
    homebrew_path,
    ollama_model_names,
    ollama_status,
    spacy_model_status,
    whisper_download_root,
    whisper_model_status,
)
from live_transcriber.session import LiveTranscriberCallbacks, LiveTranscriberConfig, LiveTranscriberSession
from live_transcriber.speakers import DEFAULT_SPEAKER_BACKEND, SPEAKER_BACKENDS
from live_transcriber.translations import (
    DEFAULT_WORD_HINT_MODEL,
    DEFAULT_TRANSLATION_TARGET_LANGUAGE,
    build_selective_translations,
)

try:
    from PySide6.QtCore import QObject, QSettings, QSize, QStandardPaths, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QStyle,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit("PySide6 is not installed. Run: venv/bin/python -m pip install -r requirements.txt") from exc


TRANSCRIPT_SCHEMA_VERSION = 2
TRANSCRIPT_FILE_SUFFIX = ".trans"
SUPPORTED_TRANSCRIPT_SCHEMA_VERSIONS = {1, 2}
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app_icon.png"


@dataclass
class TranscriptDocument:
    title: str
    widget: QWidget
    transcript: QTextEdit
    minutes: QTextEdit
    minutes_status: QLabel
    vocabulary: QTextEdit
    splitter: QSplitter
    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    vocabulary_items: dict[str, dict[str, str]] = field(default_factory=dict)
    minutes_text: str = ""
    minutes_status_text: str = "Minutes: idle"
    last_minutes_event_index: int = 0
    minutes_update_pending: bool = False
    minutes_running: bool = False
    last_final_speaker: str | None = None
    dirty: bool = False


class TranscriptionWorker(QObject):
    status_changed = Signal(str)
    partial_text = Signal(str)
    final_text = Signal(str, object)
    audio_level = Signal(float, bool, str)
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
            on_audio_level=self.audio_level.emit,
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


class MinutesWorker(QObject):
    finished = Signal(str, int)
    error = Signal(str, int)

    def __init__(
        self,
        events: list[dict[str, Any]],
        existing_minutes: str,
        target_event_index: int,
        client_factory: Callable[[], OllamaMinutesClient],
    ) -> None:
        super().__init__()
        self._events = events
        self._existing_minutes = existing_minutes
        self._target_event_index = target_event_index
        self._client_factory = client_factory

    @Slot()
    def run(self) -> None:
        try:
            minutes = self._client_factory().summarize(self._existing_minutes, self._events)
        except MinutesError as exc:
            self.error.emit(str(exc), self._target_event_index)
        except RuntimeError as exc:
            self.error.emit(str(exc), self._target_event_index)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Minutes update failed: {exc}", self._target_event_index)
        else:
            self.finished.emit(minutes, self._target_event_index)


class AssetInstallWorker(QObject):
    status = Signal(str)
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, task: str, asset_id: str, app_data_dir: Path) -> None:
        super().__init__()
        self._task = task
        self._asset_id = asset_id
        self._app_data_dir = app_data_dir
        self._last_command: list[str] | None = None
        self._last_output: list[str] = []

    @Slot()
    def run(self) -> None:
        try:
            if self._task == "whisper":
                self._download_whisper()
            elif self._task == "ollama":
                self._pull_ollama()
            elif self._task == "spacy":
                self._install_spacy()
            elif self._task == "dictionary":
                self._build_dictionary()
            elif self._task == "blackhole":
                self._install_blackhole()
            else:
                raise RuntimeError(f"Unknown setup task: {self._task}")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(self._format_failure(str(exc)))
        else:
            self.finished.emit(self._asset_id)

    def _download_whisper(self) -> None:
        self.status.emit(f"Downloading Whisper model: {self._asset_id}")
        self.progress.emit(0)
        root = whisper_download_root(self._app_data_dir)
        root.mkdir(parents=True, exist_ok=True)
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("faster-whisper is not installed. Run ./setup.sh or install requirements.") from exc

        WhisperModel(self._asset_id, device="cpu", compute_type="int8", download_root=str(root))
        self.progress.emit(100)
        self.status.emit(f"Whisper model ready: {self._asset_id}")

    def _pull_ollama(self) -> None:
        self.status.emit(f"Pulling Ollama model: {self._asset_id}")
        self.progress.emit(0)
        payload = json.dumps({"name": self._asset_id, "stream": True}).encode("utf-8")
        req = request.Request(
            "http://localhost:11434/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    update = json.loads(raw_line.decode("utf-8"))
                    status = str(update.get("status") or "").strip()
                    if status:
                        self.status.emit(status)
                    completed = update.get("completed")
                    total = update.get("total")
                    if isinstance(completed, int) and isinstance(total, int) and total > 0:
                        self.progress.emit(max(0, min(100, int(completed * 100 / total))))
        except error.URLError as exc:
            if shutil.which("ollama") is None:
                raise RuntimeError(
                    "Ollama is not installed or is not on PATH. Install it from https://ollama.com/download "
                    "or with: brew install ollama"
                ) from exc
            raise RuntimeError("Ollama is not running. Start the Ollama app or run: ollama serve") from exc
        except TimeoutError as exc:
            raise RuntimeError("Timed out connecting to Ollama. Start the Ollama app or run: ollama serve") from exc
        self.progress.emit(100)
        self.status.emit(f"Ollama model ready: {self._asset_id}")

    def _install_spacy(self) -> None:
        self.status.emit("Installing spaCy")
        self.progress.emit(0)
        self._run_subprocess([sys.executable, "-m", "pip", "install", "spacy"])
        self.progress.emit(50)
        self.status.emit("Installing German spaCy model")
        self._run_subprocess([sys.executable, "-m", "spacy", "download", "de_core_news_sm"])
        self.progress.emit(100)
        self.status.emit("spaCy and German model ready")

    def _build_dictionary(self) -> None:
        self.status.emit("Building German-English dictionary")
        self.progress.emit(0)
        db_path = dictionary_path(self._app_data_dir)
        from scripts import build_dictionary as dictionary_builder

        source = dictionary_builder._download_if_needed(dictionary_builder.DEFAULT_FREEDICT_URL, db_path.parent)
        self.status.emit("Indexing dictionary entries")
        count = dictionary_builder.build_dictionary(source, db_path)
        self.progress.emit(100)
        self.status.emit(f"Dictionary ready: {db_path} ({count} rows)")

    def _install_blackhole(self) -> None:
        brew = homebrew_path()
        if brew is None:
            raise RuntimeError("Homebrew was not found. Install BlackHole manually with: brew install --cask blackhole-2ch")
        self.status.emit("Installing BlackHole 2ch")
        self.progress.emit(0)
        self._run_subprocess([brew, "install", "--cask", "blackhole-2ch"])
        self.progress.emit(100)
        self.status.emit("BlackHole 2ch installed")

    def _run_subprocess(self, command: list[str]) -> None:
        self._last_command = command
        self._last_output = []
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            message = line.strip()
            if message:
                self._last_output.append(message)
                self._last_output = self._last_output[-12:]
                self.status.emit(message)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed ({return_code}): {' '.join(command)}")

    def _format_failure(self, message: str) -> str:
        parts = [message]
        if self._last_command is not None:
            command = " ".join(self._last_command)
            parts.append(f"Command: {command}")
            parts.append(f"Manual command: {command}")
        if self._last_output:
            parts.append("Last output:")
            parts.extend(self._last_output)
        return "\n".join(parts)


class AudioProbeWorker(QObject):
    level_changed = Signal(float, bool, str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, device: int | None, sample_rate: int = 16000) -> None:
        super().__init__()
        self._device = device
        self._sample_rate = int(sample_rate)
        self._stop_event = Event()
        self._stream = None

    @Slot()
    def run(self) -> None:
        try:
            import sounddevice as sd
        except ModuleNotFoundError:
            self.error.emit("Input meter unavailable: sounddevice is not installed")
            self.finished.emit()
            return

        last_emit_at = 0.0

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info
            nonlocal last_emit_at
            now = time.monotonic()
            if now - last_emit_at < 0.08:
                return
            last_emit_at = now
            level, active = audio_signal_level(indata)
            self.level_changed.emit(level, active, str(status) if status else "")

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
                blocksize=max(1, int(self._sample_rate * 0.08)),
                callback=callback,
            )
            self._stream.start()
            while not self._stop_event.wait(0.1):
                pass
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Input meter unavailable: {exc}")
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop_event.set()


class AudioLevelBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._level = 0.0
        self._active = False
        self.setMinimumWidth(180)
        self.setFixedHeight(32)
        self.setToolTip("Shows whether the selected input device is receiving audio.")

    def set_level(self, level: float, active: bool) -> None:
        self._level = max(0.0, min(1.0, float(level)))
        self._active = bool(active)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 4, -1, -4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#edf3f4"))
        painter.drawRoundedRect(rect, 8, 8)

        bars = 22
        gap = 3
        available_width = max(1, rect.width() - gap * (bars + 1))
        bar_width = max(2, available_width / bars)
        active_bars = int(round(self._level * bars))
        center_y = rect.center().y()
        max_height = max(6, rect.height() - 8)

        for index in range(bars):
            envelope = 0.25 + 0.75 * abs((index - (bars - 1) / 2) / ((bars - 1) / 2))
            envelope = 1.0 - min(0.72, envelope * 0.46)
            live_strength = self._level if self._active else min(self._level, 0.2)
            height = max(4, int(max_height * (0.16 + live_strength * envelope)))
            x = rect.left() + gap + index * (bar_width + gap)
            y = int(center_y - height / 2)

            if self._active and index < active_bars:
                color = QColor("#1f7f78" if self._level < 0.78 else "#bf7a14")
            else:
                color = QColor("#c9d5d8")
            painter.setBrush(color)
            painter.drawRoundedRect(int(x), y, int(bar_width), height, 2, 2)

        painter.end()


class MainWindow(QMainWindow):
    asset_task_finished = Signal(str, str, str)
    asset_task_failed = Signal(str, str, str)

    def __init__(self, minutes_client_factory: Callable[[], OllamaMinutesClient] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Live Transcriber")
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(980, 680)

        app = QApplication.instance()
        if app is not None:
            app.setOrganizationName("LiveTranscriber")
            app.setApplicationName("Live Transcriber")
            if APP_ICON_PATH.exists():
                app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        settings_file = os.environ.get("LIVE_TRANSCRIBER_SETTINGS_FILE")
        if settings_file:
            self.settings = QSettings(settings_file, QSettings.Format.IniFormat)
        else:
            self.settings = QSettings("LiveTranscriber", "LiveTranscriber")
        qt_app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        self._app_data_dir = Path(qt_app_data) if qt_app_data else default_app_data_dir()
        ensure_app_dirs(self._app_data_dir)
        os.environ.setdefault("LIVE_TRANSCRIBER_DICTIONARY_PATH", str(dictionary_path(self._app_data_dir)))

        self._thread: QThread | None = None
        self._worker: TranscriptionWorker | None = None
        self._minutes_thread: QThread | None = None
        self._minutes_worker: MinutesWorker | None = None
        self._minutes_document: TranscriptDocument | None = None
        self._minutes_client_factory = minutes_client_factory or self._create_minutes_client
        self._is_running = False
        self._audio_probe_thread: QThread | None = None
        self._audio_probe_worker: AudioProbeWorker | None = None
        self._audio_probe_enabled = (
            os.environ.get("LIVE_TRANSCRIBER_DISABLE_AUDIO_PROBE") != "1"
            and os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        )
        self._documents: dict[QWidget, TranscriptDocument] = {}
        self._recording_document: TranscriptDocument | None = None
        self._untitled_count = 0
        self._asset_threads: list[tuple[QThread, AssetInstallWorker]] = []
        self._asset_buttons: dict[tuple[str, str], QPushButton] = {}
        self._asset_button_defaults: dict[tuple[str, str], tuple[str, QIcon]] = {}
        self._model_use_buttons: dict[tuple[str, str], QPushButton] = {}
        self._active_asset_tasks: set[tuple[str, str]] = set()
        self._spinner_frame = 0
        self._spinner_icons = self._create_spinner_icons()
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(90)
        self._spinner_timer.timeout.connect(self._advance_spinner_icons)
        self._asset_status_label: QLabel | None = None
        self._asset_progress: QProgressBar | None = None
        self._whisper_status_labels: dict[str, QLabel] = {}
        self._ollama_status_labels: dict[str, QLabel] = {}
        self._spacy_status_label: QLabel | None = None
        self._dictionary_status_label: QLabel | None = None
        self._blackhole_status_label: QLabel | None = None
        self._selected_transcription_model_label: QLabel | None = None
        self._selected_minutes_model_label: QLabel | None = None
        self._selected_word_hints_model_label: QLabel | None = None

        self.asset_task_finished.connect(self._on_asset_task_finished, Qt.ConnectionType.QueuedConnection)
        self.asset_task_failed.connect(self._on_asset_task_error, Qt.ConnectionType.QueuedConnection)

        self._build_ui()
        self.new_transcript()
        self.refresh_devices()
        self._refresh_asset_statuses()
        self._set_running(False)
        self._update_speaker_controls()
        self._update_translation_controls()

    def _build_ui(self) -> None:
        live_page = QWidget()
        layout = QVBoxLayout(live_page)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.clicked.connect(self.toggle_transcription)
        top_row.addWidget(self.start_button)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(320)
        self.device_combo.currentIndexChanged.connect(self._on_input_device_changed)
        top_row.addWidget(QLabel("Input"))
        top_row.addWidget(self.device_combo, stretch=1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_devices)
        top_row.addWidget(self.refresh_button)
        layout.addLayout(top_row)

        meter_row = QHBoxLayout()
        meter_row.setSpacing(10)
        meter_label = QLabel("Signal")
        meter_label.setStyleSheet("color: #52616d; font-weight: 700;")
        meter_row.addWidget(meter_label)
        self.audio_level_bar = AudioLevelBar()
        meter_row.addWidget(self.audio_level_bar, stretch=1)
        self.audio_signal_label = QLabel("Checking input...")
        self.audio_signal_label.setMinimumWidth(155)
        self.audio_signal_label.setStyleSheet("color: #60707a;")
        meter_row.addWidget(self.audio_signal_label)
        layout.addLayout(meter_row)

        options_grid = QGridLayout()
        options_grid.setHorizontalSpacing(16)
        options_grid.setVerticalSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.addItems([asset.asset_id for asset in WHISPER_MODELS])
        self.model_combo.setCurrentText(str(self.settings.value("models/transcription", DEFAULT_MODEL)))
        self.model_combo.currentTextChanged.connect(self._on_transcription_model_changed)
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

        document_row = QHBoxLayout()
        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.new_transcript)
        document_row.addWidget(self.new_button)

        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_transcript)
        document_row.addWidget(self.open_button)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_current_transcript)
        document_row.addWidget(self.save_button)

        self.save_as_button = QPushButton("Save As")
        self.save_as_button.clicked.connect(self.save_current_transcript_as)
        document_row.addWidget(self.save_as_button)

        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.rename_current_transcript)
        document_row.addWidget(self.rename_button)

        self.close_tab_button = QPushButton("Close Tab")
        self.close_tab_button.clicked.connect(self.close_current_transcript)
        document_row.addWidget(self.close_tab_button)

        document_row.addStretch(1)
        layout.addLayout(document_row)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_transcript_at)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)
        layout.addWidget(self.tabs, stretch=1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Idle")
        footer.addWidget(self.status_label, stretch=1)
        self.output_label = QLabel(str(DEFAULT_OUTPUT))
        self.output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(self.output_label)
        layout.addLayout(footer)

        self.app_tabs = QTabWidget()
        self.app_tabs.addTab(live_page, "Live")
        self.app_tabs.addTab(self._build_settings_page(), "Settings")

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.app_tabs)
        self.setCentralWidget(root)

    def _build_settings_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        heading = QLabel("Settings")
        heading.setStyleSheet("font-size: 22px; font-weight: 800; color: #14222c;")
        layout.addWidget(heading)

        intro = QLabel(
            "Choose local models, pre-download assets, and check whether audio capture support is ready."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #52616d;")
        layout.addWidget(intro)

        selection_group = QGroupBox("Selected Defaults")
        selection_grid = QGridLayout(selection_group)
        selection_grid.setHorizontalSpacing(14)
        selection_grid.setVerticalSpacing(8)
        self._selected_minutes_model_label = QLabel()
        self._selected_word_hints_model_label = QLabel()
        self._selected_transcription_model_label = QLabel(self.model_combo.currentText())
        selection_grid.addWidget(QLabel("Transcription"), 0, 0)
        selection_grid.addWidget(self._selected_transcription_model_label, 0, 1)
        selection_grid.addWidget(QLabel("Meeting minutes"), 1, 0)
        selection_grid.addWidget(self._selected_minutes_model_label, 1, 1)
        selection_grid.addWidget(QLabel("Final word hints"), 2, 0)
        selection_grid.addWidget(self._selected_word_hints_model_label, 2, 1)
        layout.addWidget(selection_group)

        whisper_group = QGroupBox("Whisper Transcription Models")
        whisper_grid = QGridLayout(whisper_group)
        whisper_grid.setHorizontalSpacing(10)
        whisper_grid.setVerticalSpacing(8)
        whisper_grid.addWidget(QLabel("Model"), 0, 0)
        whisper_grid.addWidget(QLabel("Use"), 0, 1)
        whisper_grid.addWidget(QLabel("Download"), 0, 2)
        whisper_grid.addWidget(QLabel("Status"), 0, 3)
        for row, asset in enumerate(WHISPER_MODELS, start=1):
            label = QLabel(
                f"<b>{html.escape(asset.label)}</b> "
                f"<span style='color:#6b7882'>{html.escape(asset.size_hint)}</span><br>"
                f"<span style='color:#53616b'>{html.escape(asset.description)}</span>"
            )
            label.setWordWrap(True)
            whisper_grid.addWidget(label, row, 0)

            use_button = QPushButton("Use")
            use_button.clicked.connect(lambda _checked=False, model=asset.asset_id: self._set_transcription_model(model))
            self._register_model_use_button("transcription", asset.asset_id, use_button)
            whisper_grid.addWidget(use_button, row, 1)

            download_button = QPushButton("Download")
            download_button.clicked.connect(
                lambda _checked=False, model=asset.asset_id: self._start_asset_task("whisper", model)
            )
            self._register_asset_button("whisper", asset.asset_id, download_button)
            whisper_grid.addWidget(download_button, row, 2)

            status_label = self._new_status_label()
            self._whisper_status_labels[asset.asset_id] = status_label
            whisper_grid.addWidget(status_label, row, 3)
        whisper_grid.setColumnStretch(0, 1)
        layout.addWidget(whisper_group)

        ollama_group = QGroupBox("Local LLMs for Minutes and Word Hints")
        ollama_grid = QGridLayout(ollama_group)
        ollama_grid.setHorizontalSpacing(10)
        ollama_grid.setVerticalSpacing(8)
        ollama_grid.addWidget(QLabel("Model"), 0, 0)
        ollama_grid.addWidget(QLabel("Use"), 0, 1)
        ollama_grid.addWidget(QLabel("Pull"), 0, 2)
        ollama_grid.addWidget(QLabel("Status"), 0, 3)
        for row, asset in enumerate(OLLAMA_MODELS, start=1):
            label = QLabel(
                f"<b>{html.escape(asset.label)}</b> "
                f"<span style='color:#6b7882'>{html.escape(asset.size_hint)}</span><br>"
                f"<span style='color:#53616b'>{html.escape(asset.description)}</span>"
            )
            label.setWordWrap(True)
            ollama_grid.addWidget(label, row, 0)

            use_buttons = QVBoxLayout()
            minutes_button = QPushButton("Minutes")
            minutes_button.clicked.connect(
                lambda _checked=False, model=asset.asset_id: self._set_minutes_model(model)
            )
            self._register_model_use_button("minutes", asset.asset_id, minutes_button)
            word_hints_button = QPushButton("Word Hints")
            word_hints_button.clicked.connect(
                lambda _checked=False, model=asset.asset_id: self._set_word_hints_model(model)
            )
            self._register_model_use_button("word_hints", asset.asset_id, word_hints_button)
            use_buttons.addWidget(minutes_button)
            use_buttons.addWidget(word_hints_button)
            ollama_grid.addLayout(use_buttons, row, 1)

            pull_button = QPushButton("Pull")
            pull_button.clicked.connect(
                lambda _checked=False, model=asset.asset_id: self._start_asset_task("ollama", model)
            )
            self._register_asset_button("ollama", asset.asset_id, pull_button)
            ollama_grid.addWidget(pull_button, row, 2)

            status_label = self._new_status_label()
            self._ollama_status_labels[asset.asset_id] = status_label
            ollama_grid.addWidget(status_label, row, 3)
        ollama_grid.setColumnStretch(0, 1)
        layout.addWidget(ollama_group)

        hints_group = QGroupBox("Live Word Hints")
        hints_grid = QGridLayout(hints_group)
        hints_grid.setHorizontalSpacing(10)
        hints_grid.setVerticalSpacing(8)
        self._spacy_status_label = self._new_status_label()
        self._dictionary_status_label = self._new_status_label()
        hints_grid.addWidget(QLabel("<b>spaCy + German model</b><br><span style='color:#53616b'>Better noun and verb detection.</span>"), 0, 0)
        install_spacy = QPushButton("Install spaCy")
        install_spacy.clicked.connect(lambda: self._start_asset_task("spacy", "de_core_news_sm"))
        self._register_asset_button("spacy", "de_core_news_sm", install_spacy)
        hints_grid.addWidget(install_spacy, 0, 1)
        hints_grid.addWidget(self._spacy_status_label, 0, 2)
        hints_grid.addWidget(QLabel("<b>German-English dictionary</b><br><span style='color:#53616b'>Fast offline translations for live partials.</span>"), 1, 0)
        build_dictionary = QPushButton("Build")
        build_dictionary.clicked.connect(lambda: self._start_asset_task("dictionary", "de_en.sqlite"))
        self._register_asset_button("dictionary", "de_en.sqlite", build_dictionary)
        hints_grid.addWidget(build_dictionary, 1, 1)
        hints_grid.addWidget(self._dictionary_status_label, 1, 2)
        hints_grid.setColumnStretch(0, 1)
        layout.addWidget(hints_group)

        audio_group = QGroupBox("Audio Capture")
        audio_grid = QGridLayout(audio_group)
        audio_grid.setHorizontalSpacing(10)
        audio_grid.setVerticalSpacing(8)
        self._blackhole_status_label = self._new_status_label()
        audio_grid.addWidget(
            QLabel(
                "<b>BlackHole 2ch</b><br>"
                "<span style='color:#53616b'>Virtual input for YouTube, meetings, and other Mac system audio.</span>"
            ),
            0,
            0,
        )
        install_blackhole = QPushButton("Install")
        install_blackhole.clicked.connect(self._confirm_blackhole_install)
        self._register_asset_button("blackhole", "blackhole-2ch", install_blackhole)
        audio_grid.addWidget(install_blackhole, 0, 1)
        refresh_audio = QPushButton("Refresh")
        refresh_audio.clicked.connect(self._refresh_asset_statuses)
        audio_grid.addWidget(refresh_audio, 0, 2)
        audio_grid.addWidget(self._blackhole_status_label, 0, 3)
        audio_grid.setColumnStretch(0, 1)
        layout.addWidget(audio_group)

        storage_group = QGroupBox("Storage")
        storage_grid = QGridLayout(storage_group)
        storage_grid.setHorizontalSpacing(10)
        storage_grid.setVerticalSpacing(8)
        storage_path = QLabel(str(self._app_data_dir))
        storage_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        storage_grid.addWidget(QLabel("Application data"), 0, 0)
        storage_grid.addWidget(storage_path, 0, 1)
        open_storage = QPushButton("Open")
        open_storage.clicked.connect(self._open_app_data_dir)
        storage_grid.addWidget(open_storage, 0, 2)
        storage_grid.setColumnStretch(1, 1)
        layout.addWidget(storage_group)

        setup_group = QGroupBox("Setup Activity")
        setup_layout = QVBoxLayout(setup_group)
        self._asset_status_label = QLabel("Ready")
        self._asset_status_label.setWordWrap(True)
        self._asset_progress = QProgressBar()
        self._asset_progress.setRange(0, 100)
        self._asset_progress.setValue(0)
        setup_layout.addWidget(self._asset_status_label)
        setup_layout.addWidget(self._asset_progress)
        layout.addWidget(setup_group)

        refresh_all = QPushButton("Refresh Status")
        refresh_all.clicked.connect(self._refresh_asset_statuses)
        layout.addWidget(refresh_all)
        layout.addStretch(1)

        scroll.setWidget(page)
        return scroll

    def _new_status_label(self) -> QLabel:
        label = QLabel("Checking...")
        label.setWordWrap(True)
        label.setStyleSheet("color: #4f5d67;")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _register_asset_button(self, task: str, asset_id: str, button: QPushButton) -> None:
        key = (task, asset_id)
        self._asset_buttons[key] = button
        self._asset_button_defaults[key] = (button.text(), button.icon())

    def _register_model_use_button(self, role: str, model_id: str, button: QPushButton) -> None:
        self._model_use_buttons[(role, model_id)] = button

    def _create_spinner_icons(self) -> list[QIcon]:
        icons: list[QIcon] = []
        size = 18
        center = size / 2
        for frame in range(12):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for index in range(12):
                alpha = int(40 + 215 * ((index + frame) % 12) / 11)
                pen = QPen(QColor(35, 83, 91, alpha), 2.2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.save()
                painter.translate(center, center)
                painter.rotate(index * 30)
                painter.drawLine(0, -6.5, 0, -8.0)
                painter.restore()
            painter.end()
            icons.append(QIcon(pixmap))
        return icons

    def _task_loading_text(self, task: str) -> str:
        return {
            "whisper": "Downloading...",
            "ollama": "Pulling...",
            "dictionary": "Building...",
            "spacy": "Installing...",
            "blackhole": "Installing...",
        }.get(task, "Working...")

    def _set_asset_button_loading(self, task: str, asset_id: str) -> None:
        key = (task, asset_id)
        button = self._asset_buttons.get(key)
        if button is None:
            return
        self._active_asset_tasks.add(key)
        button.setEnabled(False)
        button.setText(self._task_loading_text(task))
        button.setIcon(self._spinner_icons[self._spinner_frame])
        button.setIconSize(QSize(18, 18))
        if not self._spinner_timer.isActive():
            self._spinner_timer.start()

    def _restore_asset_button(self, task: str, asset_id: str) -> None:
        key = (task, asset_id)
        self._active_asset_tasks.discard(key)
        button = self._asset_buttons.get(key)
        defaults = self._asset_button_defaults.get(key)
        if button is not None and defaults is not None:
            text, icon = defaults
            button.setText(text)
            button.setIcon(icon)
            button.setEnabled(True)
        if not self._active_asset_tasks:
            self._spinner_timer.stop()

    @Slot()
    def _advance_spinner_icons(self) -> None:
        if not self._active_asset_tasks:
            self._spinner_timer.stop()
            return
        self._spinner_frame = (self._spinner_frame + 1) % len(self._spinner_icons)
        icon = self._spinner_icons[self._spinner_frame]
        for key in self._active_asset_tasks:
            button = self._asset_buttons.get(key)
            if button is not None:
                button.setIcon(icon)

    def _create_minutes_client(self) -> OllamaMinutesClient:
        return OllamaMinutesClient(model=self._selected_minutes_model())

    def _selected_minutes_model(self) -> str:
        return str(self.settings.value("models/minutes", DEFAULT_MINUTES_MODEL))

    def _selected_word_hints_model(self) -> str:
        return str(self.settings.value("models/word_hints", DEFAULT_WORD_HINT_MODEL))

    @Slot(str)
    def _on_transcription_model_changed(self, model: str) -> None:
        self.settings.setValue("models/transcription", model)
        if self._selected_transcription_model_label is not None:
            self._selected_transcription_model_label.setText(model)
        status = whisper_model_status(model, self._app_data_dir)
        if not status.installed:
            self.set_status(f"{model} will download on first use, or use Settings to download it now.")
        if self._model_use_buttons:
            self._refresh_asset_statuses()

    def _set_transcription_model(self, model: str) -> None:
        self.model_combo.setCurrentText(model)
        self.settings.setValue("models/transcription", model)
        self._refresh_asset_statuses()

    def _set_minutes_model(self, model: str) -> None:
        self.settings.setValue("models/minutes", model)
        self.set_status(f"Minutes model set to {model}")
        self._refresh_asset_statuses()

    def _set_word_hints_model(self, model: str) -> None:
        self.settings.setValue("models/word_hints", model)
        self.set_status(f"Word-hint model set to {model}")
        self._refresh_asset_statuses()

    @Slot()
    def _refresh_asset_statuses(self) -> None:
        whisper_statuses: dict[str, AssetStatus] = {}
        for asset in WHISPER_MODELS:
            status = whisper_model_status(asset.asset_id, self._app_data_dir)
            whisper_statuses[asset.asset_id] = status
            label = self._whisper_status_labels.get(asset.asset_id)
            if label is None:
                continue
            self._apply_status_label(label, status)

        ollama_names = ollama_model_names()
        ollama_statuses: dict[str, AssetStatus] = {}
        for asset in OLLAMA_MODELS:
            status = ollama_status(asset.asset_id, ollama_names)
            ollama_statuses[asset.asset_id] = status
            label = self._ollama_status_labels.get(asset.asset_id)
            if label is None:
                continue
            self._apply_status_label(label, status)

        if self._spacy_status_label is not None:
            self._apply_status_label(self._spacy_status_label, spacy_model_status())
        if self._dictionary_status_label is not None:
            self._apply_status_label(self._dictionary_status_label, dictionary_status(self._app_data_dir))
        if self._blackhole_status_label is not None:
            device_names = [self.device_combo.itemText(index) for index in range(self.device_combo.count())]
            self._apply_status_label(self._blackhole_status_label, blackhole_status(device_names))

        if self._selected_transcription_model_label is not None:
            self._selected_transcription_model_label.setText(self.model_combo.currentText())
        if self._selected_minutes_model_label is not None:
            self._selected_minutes_model_label.setText(self._selected_minutes_model())
        if self._selected_word_hints_model_label is not None:
            self._selected_word_hints_model_label.setText(self._selected_word_hints_model())
        self._refresh_model_use_buttons(whisper_statuses, ollama_statuses)

    def _apply_status_label(self, label: QLabel, status: AssetStatus) -> None:
        color = "#25714f" if status.installed else "#9a5a17"
        text = html.escape(status.detail)
        if status.path:
            text += f"<br><span style='color:#697782'>{html.escape(status.path)}</span>"
        label.setText(f"<span style='color:{color}; font-weight:700'>{text}</span>")

    def _refresh_model_use_buttons(
        self,
        whisper_statuses: dict[str, AssetStatus],
        ollama_statuses: dict[str, AssetStatus],
    ) -> None:
        selected_transcription = self.model_combo.currentText()
        selected_minutes = self._selected_minutes_model()
        selected_word_hints = self._selected_word_hints_model()

        for asset in WHISPER_MODELS:
            status = whisper_statuses.get(asset.asset_id)
            self._set_model_use_button_state(
                role="transcription",
                model_id=asset.asset_id,
                selected=asset.asset_id == selected_transcription,
                installed=bool(status and status.installed),
            )

        for asset in OLLAMA_MODELS:
            status = ollama_statuses.get(asset.asset_id)
            installed = bool(status and status.installed)
            self._set_model_use_button_state(
                role="minutes",
                model_id=asset.asset_id,
                selected=asset.asset_id == selected_minutes,
                installed=installed,
            )
            self._set_model_use_button_state(
                role="word_hints",
                model_id=asset.asset_id,
                selected=asset.asset_id == selected_word_hints,
                installed=installed,
            )

    def _set_model_use_button_state(
        self,
        role: str,
        model_id: str,
        selected: bool,
        installed: bool,
    ) -> None:
        button = self._model_use_buttons.get((role, model_id))
        if button is None:
            return

        button.setEnabled(installed and not selected)
        if selected:
            button.setToolTip("Already selected in Selected Defaults.")
        elif not installed:
            button.setToolTip("Download or pull this model before using it.")
        else:
            button.setToolTip("")

    def _start_asset_task(self, task: str, asset_id: str) -> None:
        key = (task, asset_id)
        if key in self._active_asset_tasks:
            return
        if self._asset_status_label is not None:
            self._asset_status_label.setText(f"Starting {asset_id}...")
        if self._asset_progress is not None:
            self._asset_progress.setValue(0)
        self._set_asset_button_loading(task, asset_id)

        thread = QThread(self)
        worker = AssetInstallWorker(task, asset_id, self._app_data_dir)
        worker.moveToThread(thread)
        self._asset_threads.append((thread, worker))

        thread.started.connect(worker.run)
        worker.status.connect(self._on_asset_task_status)
        worker.progress.connect(self._on_asset_task_progress)
        worker.finished.connect(
            lambda done_asset_id, task=task, asset_id=asset_id: self.asset_task_finished.emit(
                task,
                asset_id,
                done_asset_id,
            )
        )
        worker.finished.connect(thread.quit)
        worker.error.connect(
            lambda message, task=task, asset_id=asset_id: self.asset_task_failed.emit(task, asset_id, message)
        )
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda task_thread=thread, task_worker=worker: self._cleanup_asset_thread(task_thread, task_worker))
        thread.start()

    @Slot(str)
    def _on_asset_task_status(self, message: str) -> None:
        if self._asset_status_label is not None:
            self._asset_status_label.setText(message)

    @Slot(int)
    def _on_asset_task_progress(self, value: int) -> None:
        if self._asset_progress is not None:
            self._asset_progress.setValue(max(0, min(100, value)))

    @Slot(str, str, str)
    def _on_asset_task_finished(self, task: str, asset_id: str, done_asset_id: str) -> None:
        self._restore_asset_button(task, asset_id)
        if self._asset_status_label is not None:
            self._asset_status_label.setText(f"Ready: {done_asset_id}")
        if self._asset_progress is not None:
            self._asset_progress.setValue(100)
        self._refresh_asset_statuses()

    @Slot(str, str, str)
    def _on_asset_task_error(self, task: str, asset_id: str, message: str) -> None:
        self._restore_asset_button(task, asset_id)
        if self._asset_status_label is not None:
            self._asset_status_label.setText(message)
        QMessageBox.warning(self, "Live Transcriber Setup", message)
        self._refresh_asset_statuses()

    def _cleanup_asset_thread(self, thread: QThread, worker: AssetInstallWorker) -> None:
        self._asset_threads = [
            (task_thread, task_worker)
            for task_thread, task_worker in self._asset_threads
            if not (task_thread is thread and task_worker is worker)
        ]

    @Slot()
    def _confirm_blackhole_install(self) -> None:
        choice = QMessageBox.question(
            self,
            "Install BlackHole 2ch",
            "Install BlackHole 2ch with Homebrew? You may need to restart audio apps afterwards.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if choice == QMessageBox.StandardButton.Ok:
            self._start_asset_task("blackhole", "blackhole-2ch")

    @Slot()
    def _open_app_data_dir(self) -> None:
        self._app_data_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(self._app_data_dir)])
        else:
            QMessageBox.information(self, "Live Transcriber", str(self._app_data_dir))

    @property
    def transcript(self) -> QTextEdit:
        return self._current_document().transcript

    @property
    def minutes(self) -> QTextEdit:
        return self._current_document().minutes

    @property
    def vocabulary(self) -> QTextEdit:
        return self._current_document().vocabulary

    @Slot()
    def new_transcript(self) -> None:
        self._untitled_count += 1
        title = "Untitled" if self._untitled_count == 1 else f"Untitled {self._untitled_count}"
        document = self._create_document(title=title)
        index = self.tabs.addTab(document.widget, document.title)
        self.tabs.setCurrentIndex(index)

    @Slot()
    def open_transcript(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Transcript",
            str(Path("transcripts").resolve()),
            f"Live Transcriber (*{TRANSCRIPT_FILE_SUFFIX})",
        )
        if not path:
            return

        try:
            document = self._load_document(Path(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Live Transcriber", f"Could not open transcript: {exc}")
            return

        index = self.tabs.addTab(document.widget, document.title)
        self.tabs.setCurrentIndex(index)

    @Slot()
    def save_current_transcript(self) -> None:
        document = self._current_document()
        if document.path is None:
            self.save_current_transcript_as()
            return
        self._save_document(document)

    @Slot()
    def save_current_transcript_as(self) -> None:
        document = self._current_document()
        default_path = Path("transcripts") / f"{self._filename_from_title(document.title)}{TRANSCRIPT_FILE_SUFFIX}"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            str(default_path.resolve()),
            f"Live Transcriber (*{TRANSCRIPT_FILE_SUFFIX})",
        )
        if not path:
            return

        chosen_path = self._with_trans_suffix(Path(path))
        document.path = chosen_path
        document.title = chosen_path.stem
        self._save_document(document)
        self._refresh_tab_title(document)

    @Slot()
    def rename_current_transcript(self) -> None:
        document = self._current_document()
        new_title, ok = QInputDialog.getText(self, "Rename Transcript", "Name:", text=document.title)
        if not ok:
            return

        new_title = new_title.strip()
        if not new_title:
            return

        old_path = document.path
        document.title = new_title
        if old_path is not None:
            new_path = old_path.with_name(f"{self._filename_from_title(new_title)}{TRANSCRIPT_FILE_SUFFIX}")
            if new_path != old_path:
                if new_path.exists():
                    QMessageBox.warning(self, "Live Transcriber", f"File already exists: {new_path.name}")
                    document.title = old_path.stem
                    return
                try:
                    old_path.rename(new_path)
                except OSError as exc:
                    QMessageBox.warning(self, "Live Transcriber", f"Could not rename file: {exc}")
                    document.title = old_path.stem
                    return
                document.path = new_path

        document.dirty = True
        self._save_document(document) if document.path is not None else self._mark_dirty(document)
        self._refresh_tab_title(document)

    @Slot()
    def close_current_transcript(self) -> None:
        self.close_transcript_at(self.tabs.currentIndex())

    @Slot(int)
    def close_transcript_at(self, index: int) -> None:
        if index < 0:
            return
        widget = self.tabs.widget(index)
        document = self._documents.get(widget)
        if document is None:
            return
        if self._is_running and document is self._recording_document:
            QMessageBox.warning(self, "Live Transcriber", "Stop listening before closing the active transcript.")
            return
        if not self._confirm_close_document(document):
            return

        self.tabs.removeTab(index)
        self._documents.pop(widget, None)
        widget.deleteLater()
        if self.tabs.count() == 0:
            self.new_transcript()

    def _create_document(self, title: str, path: Path | None = None) -> TranscriptDocument:
        transcript = QTextEdit()
        transcript.setReadOnly(True)
        transcript.setPlaceholderText("Transcript appears here when listening starts.")
        transcript.setStyleSheet(self._transcript_style())

        minutes = QTextEdit()
        minutes.setReadOnly(True)
        minutes.setPlaceholderText("English meeting minutes appear after 3 final transcript lines.")
        minutes.setMinimumWidth(260)
        minutes.setStyleSheet(self._minutes_style())

        minutes_status = QLabel("Minutes: idle")
        minutes_status.setStyleSheet("color: #5d6670; font-size: 11px;")
        minutes_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        vocabulary = QTextEdit()
        vocabulary.setReadOnly(True)
        vocabulary.setPlaceholderText("New words appear here.")
        vocabulary.setMinimumWidth(230)
        vocabulary.setStyleSheet(self._vocabulary_style())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(transcript)

        minutes_column = QWidget()
        minutes_layout = QVBoxLayout(minutes_column)
        minutes_layout.setContentsMargins(0, 0, 0, 0)
        minutes_layout.setSpacing(6)
        minutes_header = QHBoxLayout()
        minutes_title = QLabel("Minutes")
        minutes_title.setStyleSheet("font-weight: 700; color: #253b4a;")
        minutes_header.addWidget(minutes_title)
        minutes_header.addStretch(1)
        minutes_header.addWidget(minutes_status)
        minutes_layout.addLayout(minutes_header)
        minutes_layout.addWidget(minutes, stretch=1)
        splitter.addWidget(minutes_column)

        vocab_column = QWidget()
        vocab_layout = QVBoxLayout(vocab_column)
        vocab_layout.setContentsMargins(0, 0, 0, 0)
        vocab_layout.setSpacing(6)
        vocab_title = QLabel("Vocabulary")
        vocab_title.setStyleSheet("font-weight: 700; color: #3d3321;")
        vocab_layout.addWidget(vocab_title)
        vocab_layout.addWidget(vocabulary, stretch=1)
        splitter.addWidget(vocab_column)
        splitter.setSizes([560, 300, 260])

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(splitter)

        document = TranscriptDocument(
            title=title,
            path=path,
            widget=wrapper,
            transcript=transcript,
            minutes=minutes,
            minutes_status=minutes_status,
            vocabulary=vocabulary,
            splitter=splitter,
        )
        self._documents[wrapper] = document
        return document

    def _load_document(self, path: Path) -> TranscriptDocument:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_version = int(payload.get("schema_version", 0))
        if schema_version not in SUPPORTED_TRANSCRIPT_SCHEMA_VERSIONS:
            raise ValueError("Unsupported transcript file version.")

        title = str(payload.get("title") or path.stem)
        document = self._create_document(title=title, path=path)
        events = payload.get("events") or []
        if not isinstance(events, list):
            raise ValueError("Transcript file has invalid events.")

        document.events = []
        document.vocabulary_items = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            kind = event.get("kind")
            if kind == "partial":
                self._append_partial_to_document(document, str(event.get("text") or ""), save_event=False)
                document.events.append({"kind": "partial", "text": str(event.get("text") or "")})
            elif kind == "final":
                record = event.get("record") if isinstance(event.get("record"), dict) else {}
                display_text = str(event.get("display_text") or record.get("text") or "")
                self._append_final_to_document(document, display_text, record, save_event=False)
                document.events.append({"kind": "final", "display_text": display_text, "record": dict(record)})

        stored_vocabulary = payload.get("vocabulary_items")
        if isinstance(stored_vocabulary, dict):
            document.vocabulary_items = {
                str(key): value
                for key, value in stored_vocabulary.items()
                if isinstance(value, dict)
            }
            self._render_vocabulary(document)

        if schema_version >= 2:
            document.minutes_text = str(payload.get("minutes_text") or "")
            document.last_minutes_event_index = int(payload.get("last_minutes_event_index") or 0)
            status = str(payload.get("minutes_status") or "Minutes: idle")
            document.minutes_status_text = self._restored_minutes_status(status)
            self._render_minutes(document)
            self._set_minutes_status(document, document.minutes_status_text)

        splitter_sizes = payload.get("splitter_sizes")
        if isinstance(splitter_sizes, list) and all(isinstance(value, int) for value in splitter_sizes):
            if len(splitter_sizes) == 2:
                splitter_sizes = [splitter_sizes[0], 300, splitter_sizes[1]]
            document.splitter.setSizes(splitter_sizes)

        transcript_scroll = payload.get("transcript_scroll_position")
        if isinstance(transcript_scroll, int):
            document.transcript.verticalScrollBar().setValue(transcript_scroll)
        minutes_scroll = payload.get("minutes_scroll_position")
        if isinstance(minutes_scroll, int):
            document.minutes.verticalScrollBar().setValue(minutes_scroll)
        vocabulary_scroll = payload.get("vocabulary_scroll_position")
        if isinstance(vocabulary_scroll, int):
            document.vocabulary.verticalScrollBar().setValue(vocabulary_scroll)

        document.dirty = False
        return document

    def _save_document(self, document: TranscriptDocument) -> None:
        if document.path is None:
            return

        payload = self._document_payload(document)
        document.path.parent.mkdir(parents=True, exist_ok=True)
        document.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        document.dirty = False
        self._refresh_tab_title(document)

    def _document_payload(self, document: TranscriptDocument) -> dict[str, Any]:
        return {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "title": document.title,
            "events": document.events,
            "vocabulary_items": document.vocabulary_items,
            "minutes_text": document.minutes_text,
            "minutes_status": document.minutes_status_text,
            "last_minutes_event_index": document.last_minutes_event_index,
            "splitter_sizes": document.splitter.sizes(),
            "transcript_scroll_position": document.transcript.verticalScrollBar().value(),
            "minutes_scroll_position": document.minutes.verticalScrollBar().value(),
            "vocabulary_scroll_position": document.vocabulary.verticalScrollBar().value(),
        }

    def _append_partial_to_document(self, document: TranscriptDocument, text: str, save_event: bool = True) -> None:
        if not text:
            return

        partial_text = self._render_partial_text(text)
        self._append_html_to_document(
            document,
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
            "</table>",
        )
        if save_event:
            document.events.append({"kind": "partial", "text": text})
            self._mark_dirty(document)
            self._autosave_document(document)

    def _append_final_to_document(
        self,
        document: TranscriptDocument,
        text: str,
        record: dict[str, Any],
        save_event: bool = True,
    ) -> None:
        raw_text = str(record.get("text") or text)
        speaker_label = record.get("speaker_label")
        timestamp = str(record.get("timestamp") or "")
        selective_translations = record.get("selective_translations") or []
        translation_text = str(record.get("translation_text") or "")
        margin_top = "3px" if speaker_label and speaker_label == document.last_final_speaker else "10px"
        document.last_final_speaker = str(speaker_label) if speaker_label else None

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
        self._remember_vocabulary(document, selective_translations)
        translation_html = self._render_full_translation(
            translation_text,
            str(record.get("translation_target_language") or DEFAULT_TRANSLATION_TARGET_LANGUAGE),
        )

        self._append_html_to_document(
            document,
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
            "</table>",
        )

        if save_event:
            document.events.append({"kind": "final", "display_text": text, "record": dict(record)})
            self._mark_dirty(document)
            self._autosave_document(document)
            self._maybe_schedule_minutes(document)

    def _append_html_to_document(self, document: TranscriptDocument, value: str) -> None:
        cursor = document.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(value)
        cursor.insertBlock()
        document.transcript.setTextCursor(cursor)
        document.transcript.ensureCursorVisible()

    def _maybe_schedule_minutes(self, document: TranscriptDocument) -> None:
        final_count = len(self._final_events(document))
        if final_count - document.last_minutes_event_index < MINUTES_UPDATE_FINAL_COUNT:
            return

        if document.minutes_running or self._minutes_thread is not None:
            document.minutes_update_pending = True
            return

        self._start_minutes_worker(document, final_count)

    def _start_minutes_worker(self, document: TranscriptDocument, target_event_index: int) -> None:
        final_events = self._final_events(document)
        document.minutes_running = True
        document.minutes_update_pending = False
        self._set_minutes_status(document, "Minutes: updating")

        self._minutes_thread = QThread(self)
        self._minutes_worker = MinutesWorker(
            events=final_events,
            existing_minutes=document.minutes_text,
            target_event_index=target_event_index,
            client_factory=self._minutes_client_factory,
        )
        self._minutes_document = document
        self._minutes_worker.moveToThread(self._minutes_thread)

        self._minutes_thread.started.connect(self._minutes_worker.run)
        self._minutes_worker.finished.connect(self._on_minutes_finished)
        self._minutes_worker.finished.connect(self._minutes_thread.quit)
        self._minutes_worker.error.connect(self._on_minutes_error)
        self._minutes_worker.error.connect(self._minutes_thread.quit)
        self._minutes_thread.finished.connect(self._minutes_worker.deleteLater)
        self._minutes_thread.finished.connect(self._minutes_thread.deleteLater)
        self._minutes_thread.finished.connect(self._on_minutes_thread_finished)
        self._minutes_thread.start()

    @Slot(str, int)
    def _on_minutes_finished(self, minutes_text: str, event_index: int) -> None:
        document = self._minutes_document
        if document is None:
            return

        cleaned = minutes_text.strip()
        document.minutes_running = False
        if not cleaned:
            self._set_minutes_status(document, "Minutes: Ollama returned empty output")
            document.last_minutes_event_index = max(document.last_minutes_event_index, event_index)
            return

        document.minutes_text = cleaned
        document.last_minutes_event_index = max(document.last_minutes_event_index, event_index)
        self._render_minutes(document)
        self._set_minutes_status(document, "Minutes: idle")
        self._mark_dirty(document)
        self._autosave_document(document)

    @Slot(str, int)
    def _on_minutes_error(self, message: str, event_index: int) -> None:
        document = self._minutes_document
        if document is None:
            return

        document.minutes_running = False
        document.minutes_update_pending = False
        document.last_minutes_event_index = max(document.last_minutes_event_index, event_index)
        self._set_minutes_status(document, self._minutes_error_status(message))
        self._mark_dirty(document)
        self._autosave_document(document)

    @Slot()
    def _on_minutes_thread_finished(self) -> None:
        document = self._minutes_document
        self._minutes_thread = None
        self._minutes_worker = None
        self._minutes_document = None
        if document is not None and document.minutes_update_pending:
            document.minutes_update_pending = False
            self._maybe_schedule_minutes(document)

    def _final_events(self, document: TranscriptDocument) -> list[dict[str, Any]]:
        return [
            event
            for event in document.events
            if isinstance(event, dict) and event.get("kind") == "final"
        ]

    def _render_minutes(self, document: TranscriptDocument) -> None:
        text = document.minutes_text.strip()
        if not text:
            document.minutes.clear()
            return

        section_names = {"decisions", "action items", "open questions", "important points"}
        parts = ['<div style="font-size: 13px; color: #172530;">']
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                parts.append('<div style="height: 6px;"></div>')
                continue

            cleaned_line = self._strip_minutes_markup(line)
            section_name, inline_content = self._minutes_section_line(cleaned_line, section_names)
            if section_name:
                parts.append(
                    '<div style="margin: 12px 0 6px 0; padding-bottom: 3px; '
                    'border-bottom: 1px solid #c9d8de; color: #18445a; '
                    'font-size: 14px; font-weight: 800;">'
                    f"{html.escape(section_name)}"
                    "</div>"
                )
                if inline_content:
                    parts.append(self._render_minutes_item(inline_content))
            elif cleaned_line.casefold() == "meeting minutes":
                parts.append(
                    '<div style="margin: 0 0 10px 0; color: #102f3f; '
                    'font-size: 16px; font-weight: 800;">'
                    f"{html.escape(cleaned_line)}"
                    "</div>"
                )
            elif line.startswith(("-", "+", "•")) or (line.startswith("*") and not line.startswith("**")):
                item = self._strip_minutes_markup(line[1:].strip())
                parts.append(self._render_minutes_item(item))
            else:
                parts.append(
                    '<div style="margin: 4px 0; color: #26333b;">'
                    f"{html.escape(cleaned_line)}"
                    "</div>"
                )
        parts.append("</div>")

        document.minutes.setHtml("".join(parts))
        cursor = document.minutes.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        document.minutes.setTextCursor(cursor)

    def _render_minutes_item(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        is_none = text.casefold() in {"none", "none yet", "n/a"}
        if is_none:
            return (
                '<div style="margin: 4px 0; padding: 5px 8px; color: #6f7b83; '
                'background-color: #f7fafb; border: 1px solid #dbe5e9;">'
                f"{html.escape(text)}"
                "</div>"
            )

        return (
            '<div style="margin: 4px 0; padding: 7px 9px; '
            'background-color: #ffffff; border: 1px solid #d6e0e5;">'
            f"{html.escape(text)}"
            "</div>"
        )

    def _strip_minutes_markup(self, text: str) -> str:
        value = text.strip()
        value = re.sub(r"^#{1,6}\s*", "", value)
        value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
        value = value.strip().strip("*").strip()
        return value

    def _minutes_section_line(self, line: str, section_names: set[str]) -> tuple[str | None, str]:
        candidate, separator, remainder = line.partition(":")
        normalized = candidate.strip().casefold()
        if normalized in section_names:
            return self._format_minutes_section(candidate), remainder.strip() if separator else ""

        normalized_line = line.strip().casefold()
        if normalized_line in section_names:
            return self._format_minutes_section(line), ""

        return None, ""

    def _format_minutes_section(self, section: str) -> str:
        words = section.strip().split()
        small_words = {"and", "or", "of", "the"}
        formatted = []
        for index, word in enumerate(words):
            lowered = word.casefold()
            formatted.append(lowered if index > 0 and lowered in small_words else lowered.capitalize())
        return " ".join(formatted)

    def _set_minutes_status(self, document: TranscriptDocument, status: str) -> None:
        status = status.strip() or "Minutes: idle"
        if not status.startswith("Minutes:"):
            status = f"Minutes: {status}"
        document.minutes_status_text = status
        document.minutes_status.setText(status)

    def _restored_minutes_status(self, status: str) -> str:
        status = status.strip() or "Minutes: idle"
        if "updating" in status.casefold():
            return "Minutes: idle"
        return status if status.startswith("Minutes:") else f"Minutes: {status}"

    def _minutes_error_status(self, message: str) -> str:
        if "ollama is not running" in message.casefold():
            return "Minutes: Ollama unavailable. Start it with: ollama serve"
        return f"Minutes: {message}"

    def _autosave_document(self, document: TranscriptDocument) -> None:
        if document.path is not None:
            self._save_document(document)

    def _mark_dirty(self, document: TranscriptDocument) -> None:
        document.dirty = True
        self._refresh_tab_title(document)

    def _refresh_tab_title(self, document: TranscriptDocument) -> None:
        index = self.tabs.indexOf(document.widget)
        if index < 0:
            return
        suffix = "*" if document.dirty else ""
        self.tabs.setTabText(index, f"{document.title}{suffix}")
        if document is self._documents.get(self.tabs.currentWidget()):
            self.output_label.setText(str(document.path or "Unsaved transcript"))

    def _current_document(self) -> TranscriptDocument:
        widget = self.tabs.currentWidget()
        if widget is not None and widget in self._documents:
            return self._documents[widget]
        if self._documents:
            return next(iter(self._documents.values()))
        raise RuntimeError("No transcript documents are open.")

    def _target_document(self) -> TranscriptDocument:
        return self._recording_document or self._current_document()

    def _confirm_close_document(self, document: TranscriptDocument) -> bool:
        if not document.dirty:
            return True

        choice = QMessageBox.question(
            self,
            "Live Transcriber",
            f"Save changes to {document.title}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            if document.path is None:
                self.tabs.setCurrentWidget(document.widget)
                self.save_current_transcript_as()
                return document.path is not None and not document.dirty
            self._save_document(document)
        return True

    def _with_trans_suffix(self, path: Path) -> Path:
        return path if path.suffix == TRANSCRIPT_FILE_SUFFIX else path.with_suffix(TRANSCRIPT_FILE_SUFFIX)

    def _filename_from_title(self, title: str) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", title.strip()).strip("-._")
        return name or "transcript"

    @Slot(int)
    def _on_current_tab_changed(self, index: int) -> None:
        if index < 0:
            self.output_label.setText("No transcript")
            return
        document = self._documents.get(self.tabs.widget(index))
        self.output_label.setText(str(document.path or "Unsaved transcript") if document is not None else "No transcript")

    def _transcript_style(self) -> str:
        return """
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

    def _minutes_style(self) -> str:
        return """
            QTextEdit {
                font-size: 13px;
                background: #eef5f6;
                border: 1px solid #bfd0d6;
                border-radius: 8px;
                padding: 9px;
                selection-background-color: #bddbe7;
            }
            """

    def _vocabulary_style(self) -> str:
        return """
            QTextEdit {
                font-size: 14px;
                background: #fffdf8;
                border: 1px solid #d9ccb4;
                border-radius: 8px;
                padding: 9px;
                selection-background-color: #ead7a8;
            }
            """

    @Slot()
    def _on_input_device_changed(self) -> None:
        if self._blackhole_status_label is not None:
            self._refresh_asset_statuses()
        if not self._is_running:
            self._restart_audio_probe()

    def _restart_audio_probe(self) -> None:
        self._stop_audio_probe()
        self._set_audio_meter_state(0.0, False, "Checking input...")
        if not self._audio_probe_enabled or self._is_running:
            if not self._audio_probe_enabled:
                self._set_audio_meter_state(0.0, False, "Meter unavailable")
            return

        thread = QThread(self)
        worker = AudioProbeWorker(device=self.device_combo.currentData())
        worker.moveToThread(thread)
        self._audio_probe_thread = thread
        self._audio_probe_worker = worker

        thread.started.connect(worker.run)
        worker.level_changed.connect(self.update_audio_level)
        worker.error.connect(self._on_audio_probe_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda probe_thread=thread, probe_worker=worker: self._cleanup_audio_probe(probe_thread, probe_worker)
        )
        thread.start()

    def _stop_audio_probe(self) -> None:
        worker = self._audio_probe_worker
        thread = self._audio_probe_thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(1000)
        self._audio_probe_worker = None
        self._audio_probe_thread = None

    def _cleanup_audio_probe(self, thread: QThread, worker: AudioProbeWorker) -> None:
        if self._audio_probe_thread is thread and self._audio_probe_worker is worker:
            self._audio_probe_thread = None
            self._audio_probe_worker = None

    @Slot(float, bool, str)
    def update_audio_level(self, level: float, active: bool, status: str = "") -> None:
        if status:
            label = "Input warning"
            color = "#9a5a17"
        elif active:
            label = "Streaming audio"
            color = "#24745d"
        elif level > 0.08:
            label = "Low signal"
            color = "#64737c"
        else:
            label = "No signal"
            color = "#7a8790"
        self._set_audio_meter_state(level, active, label, color=color)

    @Slot(str)
    def _on_audio_probe_error(self, message: str) -> None:
        self._set_audio_meter_state(0.0, False, message, color="#9a5a17")

    def _set_audio_meter_state(
        self,
        level: float,
        active: bool,
        label: str,
        color: str = "#60707a",
    ) -> None:
        self.audio_level_bar.set_level(level, active)
        self.audio_signal_label.setText(html.escape(label))
        self.audio_signal_label.setStyleSheet(f"color: {color}; font-weight: 700;")

    @Slot()
    def refresh_devices(self) -> None:
        current_device = self.device_combo.currentData()
        signals_were_blocked = self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("System Default Input", None)

        try:
            devices = get_input_devices()
        except RuntimeError as exc:
            self.device_combo.blockSignals(signals_were_blocked)
            self.set_status(str(exc))
            return

        blackhole_index: int | None = None
        selected_index: int | None = None
        for device in devices:
            label = self._device_label(device)
            self.device_combo.addItem(label, device.id)
            if blackhole_index is None and "blackhole" in device.name.casefold():
                blackhole_index = self.device_combo.count() - 1

        if current_device is not None:
            index = self.device_combo.findData(current_device)
            if index >= 0:
                selected_index = index

        if selected_index is None and blackhole_index is not None:
            selected_index = blackhole_index
        if selected_index is not None:
            self.device_combo.setCurrentIndex(selected_index)
        self.device_combo.blockSignals(signals_were_blocked)
        self._refresh_asset_statuses()
        if not self._is_running:
            self._restart_audio_probe()

    @Slot()
    def toggle_transcription(self) -> None:
        if self._is_running:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self) -> None:
        self._stop_audio_probe()
        self._set_audio_meter_state(0.0, False, "Listening...")
        config = self._build_config()
        document = self._current_document()
        self._recording_document = document
        if self.clear_on_start_check.isChecked():
            document.transcript.clear()
            document.minutes.clear()
            document.vocabulary.clear()
            document.minutes_text = ""
            document.last_minutes_event_index = 0
            document.minutes_update_pending = False
            self._set_minutes_status(document, "Minutes: idle")
            document.vocabulary_items.clear()
            document.events.clear()
            document.last_final_speaker = None
            self._mark_dirty(document)
            self._autosave_document(document)
        self.output_label.setText(str(document.path or "Unsaved transcript"))

        self._thread = QThread(self)
        self._worker = TranscriptionWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(self.set_status)
        self._worker.partial_text.connect(self.append_partial)
        self._worker.final_text.connect(self.append_final)
        self._worker.audio_level.connect(self.update_audio_level)
        self._worker.error.connect(self.show_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._set_running(True)
        self._thread.start()

    def stop_transcription(self) -> None:
        self.set_status("Stopping")
        self._set_audio_meter_state(0.0, False, "Stopping...")
        if self._worker is not None:
            self._worker.stop()
        self.start_button.setEnabled(False)

    @Slot(str)
    def set_status(self, message: str) -> None:
        self.status_label.setText(message)
        if message == "Listening":
            self._refresh_asset_statuses()

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.set_status(message)
        QMessageBox.warning(self, "Live Transcriber", message)

    @Slot(str)
    def append_partial(self, text: str) -> None:
        if not self.show_partials_check.isChecked():
            return
        self._append_partial_to_document(self._target_document(), text)

    @Slot(str, object)
    def append_final(self, text: str, record: object | None = None) -> None:
        record_data = record if isinstance(record, dict) else {}
        self._append_final_to_document(self._target_document(), text, record_data)

    @Slot(int)
    def _on_worker_finished(self, exit_code: int) -> None:
        if exit_code == 0:
            self.set_status("Idle")
        self._refresh_asset_statuses()
        self._set_running(False)
        self._recording_document = None
        self._worker = None
        self._thread = None
        self._restart_audio_probe()

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
        for document in list(self._documents.values()):
            if not self._confirm_close_document(document):
                if not self._is_running:
                    self._restart_audio_probe()
                event.ignore()
                return
        self._stop_audio_probe()
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
            selective_translation_backend="ollama",
            word_hint_model=self._selected_word_hints_model(),
            translation_target_language=self.translation_target_combo.currentText(),
            whisper_download_root=whisper_download_root(self._app_data_dir),
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
            self.new_button,
            self.open_button,
            self.rename_button,
            self.close_tab_button,
        ):
            widget.setEnabled(not is_running)
        self.save_button.setEnabled(True)
        self.save_as_button.setEnabled(True)
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
            backend="dictionary",
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

    def _remember_vocabulary(self, document: TranscriptDocument, items: object) -> None:
        if not isinstance(items, list):
            return

        changed = False
        for item in items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            translation = str(item.get("translation") or "").strip()
            kind = str(item.get("kind") or "phrase")
            article = str(item.get("article") or "").strip()
            if kind not in {"noun", "verb"} or not source or not translation:
                continue

            key = self._translation_key(source)
            if key in document.vocabulary_items:
                continue

            document.vocabulary_items[key] = {
                "source": source,
                "translation": translation,
                "kind": kind,
                "article": article,
            }
            changed = True

        if changed:
            self._render_vocabulary(document)

    def _render_vocabulary(self, document: TranscriptDocument) -> None:
        parts = ['<table width="100%" cellspacing="0" cellpadding="5">']
        for item in document.vocabulary_items.values():
            background, border, text_color = self._word_hint_colors(item["kind"])
            source = self._vocabulary_source_label(item)
            parts.append(
                "<tr>"
                f'<td style="background-color: {background}; border: 1px solid {border};">'
                f'<span style="color: {text_color}; font-size: 16px; font-weight: 800;">'
                f"{html.escape(source)}"
                "</span><br>"
                f'<span style="color: #2e2a22; font-size: 13px;">{html.escape(item["translation"])}</span><br>'
                f'<span style="color: #81745e; font-size: 10px;">{html.escape(item["kind"])}</span>'
                "</td>"
                "</tr>"
            )
        parts.append("</table>")

        document.vocabulary.setHtml("".join(parts))
        cursor = document.vocabulary.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        document.vocabulary.setTextCursor(cursor)

    def _vocabulary_source_label(self, item: dict[str, str]) -> str:
        source = str(item.get("source") or "")
        article = str(item.get("article") or "")
        if item.get("kind") == "noun" and article:
            return f"{article} {source}"
        return source

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
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
