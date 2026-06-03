from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Protocol

from live_transcriber.config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_FRAME_DURATION_MS,
    DEFAULT_JSONL_OUTPUT,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MIN_SPEECH_SECONDS,
    DEFAULT_MIN_TEXT_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_PAUSE_SECONDS,
    DEFAULT_PARTIAL_SECONDS,
    DEFAULT_PRE_ROLL_SECONDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
)
from live_transcriber.device_utils import DeviceInfo, get_input_devices
from live_transcriber.speakers import DEFAULT_SPEAKER_BACKEND, SPEAKER_BACKENDS, SpeakerLabeler
from live_transcriber.translations import (
    DEFAULT_SELECTIVE_TRANSLATION_BACKEND,
    DEFAULT_TRANSLATION_TARGET_LANGUAGE,
    DEFAULT_WORD_HINT_MODEL,
    SELECTIVE_TRANSLATION_BACKENDS,
)


logger = logging.getLogger(__name__)


class Recorder(Protocol):
    def __enter__(self) -> "Recorder": ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...  # noqa: ANN001
    def read(self, timeout: float = 0.2): ...  # noqa: ANN201


class Transcriber(Protocol):
    def transcribe(self, audio, sample_rate: int) -> dict[str, Any]: ...  # noqa: ANN001
    def translate(self, audio, sample_rate: int) -> dict[str, Any]: ...  # noqa: ANN001


@dataclass(frozen=True)
class LiveTranscriberConfig:
    model: str = DEFAULT_MODEL
    device: int | None = None
    language: str | None = None
    text_path: Path | None = DEFAULT_OUTPUT
    jsonl_path: Path | None = DEFAULT_JSONL_OUTPUT
    save_transcript: bool = True
    overwrite: bool = False
    sample_rate: int = DEFAULT_SAMPLE_RATE
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD
    pause_seconds: float = DEFAULT_PAUSE_SECONDS
    pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS
    max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS
    partial_seconds: float = DEFAULT_PARTIAL_SECONDS
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH
    compute_type: str = DEFAULT_COMPUTE_TYPE
    device_type: str = DEFAULT_DEVICE_TYPE
    beam_size: int = DEFAULT_BEAM_SIZE
    speaker_labels: bool = False
    speaker_backend: str = DEFAULT_SPEAKER_BACKEND
    full_translation: bool = False
    selective_translation: bool = False
    selective_translation_backend: str = DEFAULT_SELECTIVE_TRANSLATION_BACKEND
    word_hint_model: str = DEFAULT_WORD_HINT_MODEL
    translation_target_language: str = DEFAULT_TRANSLATION_TARGET_LANGUAGE
    whisper_download_root: Path | None = None


@dataclass(frozen=True)
class LiveTranscriberCallbacks:
    on_status: Callable[[str], None] | None = None
    on_partial_text: Callable[[str], None] | None = None
    on_final_text: Callable[[str, dict[str, Any]], None] | None = None
    on_error: Callable[[str], None] | None = None


class LiveTranscriberSession:
    """Reusable controller for CLI and UI live transcription."""

    def __init__(
        self,
        recorder_factory: Callable[[LiveTranscriberConfig], Recorder] | None = None,
        transcriber_factory: Callable[[LiveTranscriberConfig], Transcriber] | None = None,
        writer_factory: Callable[[LiveTranscriberConfig], Any] | None = None,
        speaker_labeler_factory: Callable[[LiveTranscriberConfig], SpeakerLabeler | None] | None = None,
        device_provider: Callable[[], list[DeviceInfo]] = get_input_devices,
    ) -> None:
        self._recorder_factory = recorder_factory or self._default_recorder
        self._transcriber_factory = transcriber_factory or self._default_transcriber
        self._writer_factory = writer_factory or self._default_writer
        self._speaker_labeler_factory = speaker_labeler_factory or self._default_speaker_labeler
        self._device_provider = device_provider
        self._stop_event = Event()
        self._speaker_lock = Lock()
        self._speaker_labeler: SpeakerLabeler | None = None
        self._thread: Thread | None = None

    def list_devices(self) -> list[DeviceInfo]:
        return self._device_provider()

    def start(self, config: LiveTranscriberConfig, callbacks: LiveTranscriberCallbacks | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Live transcription is already running.")

        self._stop_event.clear()
        self._thread = Thread(
            target=self.run_blocking,
            args=(config, callbacks or LiveTranscriberCallbacks()),
            name="live-transcriber-session",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def reset_speakers(self) -> None:
        with self._speaker_lock:
            if self._speaker_labeler is not None:
                self._speaker_labeler.reset()

    def run_blocking(
        self,
        config: LiveTranscriberConfig,
        callbacks: LiveTranscriberCallbacks | None = None,
    ) -> int:
        callbacks = callbacks or LiveTranscriberCallbacks()
        self._stop_event.clear()

        try:
            self._validate_config(config)
            self._ensure_device(config.device)
        except (RuntimeError, ValueError) as exc:
            self._error(callbacks, str(exc))
            return 1

        writer = None
        if config.save_transcript:
            try:
                writer = self._writer_factory(config)
            except OSError as exc:
                self._error(callbacks, f"Could not prepare transcript files: {exc}")
                return 1

        self._status(callbacks, f"Loading model: {config.model}")
        try:
            transcriber = self._transcriber_factory(config)
        except RuntimeError as exc:
            self._error(callbacks, str(exc))
            return 1

        speaker_labeler = None
        if config.speaker_labels:
            try:
                speaker_labeler = self._speaker_labeler_factory(config)
            except (RuntimeError, ValueError) as exc:
                self._error(callbacks, f"Speaker labels disabled: {exc}")
                speaker_labeler = None

        with self._speaker_lock:
            self._speaker_labeler = speaker_labeler

        from live_transcriber.deduper import RecentTextDeduper
        from live_transcriber.jobs import TranscriptionJob, TranscriptionJobQueue
        from live_transcriber.partials import PartialTextTracker
        from live_transcriber.segmentation import SpeechSegmenter
        from live_transcriber.speakers import SpeakerInfo, format_speaker_text
        from live_transcriber.text_cleanup import cleanup_transcript_text, is_garbage_fragment

        recorder = self._recorder_factory(config)
        segmenter = SpeechSegmenter(
            sample_rate=config.sample_rate,
            speech_threshold=config.silence_threshold,
            pause_seconds=config.pause_seconds,
            pre_roll_seconds=config.pre_roll_seconds,
            min_speech_seconds=config.min_speech_seconds,
            min_segment_seconds=config.min_segment_seconds,
            max_segment_seconds=config.max_segment_seconds,
        )
        jobs = TranscriptionJobQueue()
        deduper = RecentTextDeduper()
        partial_tracker = PartialTextTracker()
        full_translation_enabled = config.full_translation

        def transcribe_jobs() -> None:
            nonlocal full_translation_enabled

            while True:
                job = jobs.get()
                try:
                    if job is None:
                        return

                    if job.kind == "partial":
                        if jobs.is_completed(job.utterance_id):
                            continue

                        try:
                            result = transcriber.transcribe(job.audio, sample_rate=config.sample_rate)
                        except RuntimeError as exc:
                            self._error(callbacks, f"Partial transcription failed: {exc}")
                            continue

                        if jobs.is_completed(job.utterance_id):
                            continue

                        text = cleanup_transcript_text(str(result.get("text", "")))
                        if is_garbage_fragment(text, config.min_text_length):
                            logger.debug("Skipping empty, too-short, or garbage partial transcription: %r", text)
                            continue

                        new_text = partial_tracker.update(job.utterance_id, text)
                        if new_text and callbacks.on_partial_text is not None:
                            callbacks.on_partial_text(new_text)
                        continue

                    partial_tracker.finish(job.utterance_id)
                    try:
                        result = transcriber.transcribe(job.audio, sample_rate=config.sample_rate)
                    except RuntimeError as exc:
                        self._error(callbacks, f"Transcription failed for this segment: {exc}")
                        continue

                    text = cleanup_transcript_text(str(result.get("text", "")))
                    if is_garbage_fragment(text, config.min_text_length):
                        logger.debug("Skipping empty, too-short, or garbage transcription: %r", text)
                        continue
                    if deduper.is_duplicate(text):
                        logger.debug("Skipping duplicate transcription: %r", text)
                        continue
                    deduper.remember(text)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    record = {
                        "timestamp": timestamp,
                        "text": text,
                        "language": result.get("language"),
                        "language_probability": result.get("language_probability"),
                        "utterance_id": job.utterance_id,
                        "duration_seconds": job.duration_seconds,
                        "speech_seconds": job.speech_seconds,
                        "closed_by": job.closed_by,
                        "rms_peak": job.rms_peak,
                        "model": config.model,
                        "device_id": config.device,
                        "sample_rate": config.sample_rate,
                        "beam_size": config.beam_size,
                        "segments": result.get("segments", []),
                    }
                    if config.speaker_labels:
                        record["speaker_label"] = None
                        record["speaker_confidence"] = None
                        record["speaker_backend"] = config.speaker_backend

                    speaker_info: SpeakerInfo | None = None
                    if config.speaker_labels:
                        speaker_info = self._label_speaker(
                            callbacks=callbacks,
                            audio=job.audio,
                            sample_rate=config.sample_rate,
                            utterance_id=job.utterance_id,
                        )
                        if speaker_info is not None:
                            record["speaker_label"] = speaker_info.speaker_label
                            record["speaker_confidence"] = speaker_info.confidence
                            record["speaker_backend"] = speaker_info.backend

                    if full_translation_enabled:
                        translation_text = self._translate_final(
                            callbacks=callbacks,
                            transcriber=transcriber,
                            audio=job.audio,
                            sample_rate=config.sample_rate,
                        )
                        if translation_text is None:
                            full_translation_enabled = False
                        elif translation_text:
                            record["translation_text"] = translation_text
                            record["translation_target_language"] = config.translation_target_language
                            record["translation_backend"] = "whisper"

                    if config.selective_translation:
                        from live_transcriber.translations import (
                            OllamaWordHintClient,
                            build_selective_translation_result,
                        )

                        selective_client = None
                        if config.selective_translation_backend == "ollama":
                            selective_client = OllamaWordHintClient(model=config.word_hint_model)
                        selective_translations, selective_translation_backend = build_selective_translation_result(
                            text,
                            target_language=config.translation_target_language,
                            backend=config.selective_translation_backend,
                            client=selective_client,
                        )
                        if selective_translations:
                            record["selective_translations"] = selective_translations
                            record["selective_translation_backend"] = selective_translation_backend

                    display_text = format_speaker_text(text, record.get("speaker_label"))
                    if callbacks.on_final_text is not None:
                        callbacks.on_final_text(display_text, record)
                    if writer is not None:
                        try:
                            writer.write_record(record)
                        except OSError as exc:
                            self._error(callbacks, f"Could not write transcript record: {exc}")
                finally:
                    jobs.task_done(job)

        worker = Thread(target=transcribe_jobs, name="whisper-transcriber")
        worker.start()

        def enqueue_final(segment) -> None:  # noqa: ANN001
            jobs.put_final(
                TranscriptionJob(
                    kind="final",
                    utterance_id=segment.utterance_id,
                    audio=segment.audio,
                    duration_seconds=segment.duration_seconds,
                    speech_seconds=segment.speech_seconds,
                    rms_peak=segment.rms_peak,
                    closed_by=segment.closed_by,
                )
            )

        def try_enqueue_partial() -> bool:
            active_utterance_id = segmenter.current_utterance_id
            if active_utterance_id is None or not jobs.can_accept_partial(active_utterance_id):
                return False

            snapshot = segmenter.active_snapshot()
            if snapshot is None:
                return False

            return jobs.try_put_partial(
                TranscriptionJob(
                    kind="partial",
                    utterance_id=snapshot.utterance_id,
                    audio=snapshot.audio,
                    duration_seconds=snapshot.duration_seconds,
                    speech_seconds=snapshot.speech_seconds,
                    rms_peak=snapshot.rms_peak,
                )
            )

        exit_code = 0
        last_partial_at = 0.0
        last_partial_utterance_id: int | None = None
        self._status(callbacks, "Listening")
        try:
            with recorder:
                while not self._stop_event.is_set():
                    frame = recorder.read(timeout=0.2)
                    if frame is None:
                        continue
                    if frame.status:
                        logger.debug("Audio callback status: %s", frame.status)

                    for segment in segmenter.process(frame.audio):
                        enqueue_final(segment)

                    if config.partial_seconds <= 0:
                        continue

                    active_utterance_id = segmenter.current_utterance_id
                    if active_utterance_id is None:
                        last_partial_at = 0.0
                        last_partial_utterance_id = None
                        continue

                    now = time.monotonic()
                    if active_utterance_id != last_partial_utterance_id:
                        last_partial_utterance_id = active_utterance_id
                        last_partial_at = now
                        continue

                    if now - last_partial_at >= config.partial_seconds and try_enqueue_partial():
                        last_partial_at = now
        except KeyboardInterrupt:
            self._status(callbacks, "Stopping")
        except RuntimeError as exc:
            self._error(callbacks, str(exc))
            exit_code = 1
        finally:
            final_segment = segmenter.flush()
            if final_segment is not None:
                enqueue_final(final_segment)
            jobs.put_stop()
            worker.join()
            with self._speaker_lock:
                self._speaker_labeler = None
            self._status(callbacks, "Stopped")

        return exit_code

    def _validate_config(self, config: LiveTranscriberConfig) -> None:
        if config.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if config.silence_threshold < 0:
            raise ValueError("silence_threshold must be 0 or greater")
        if config.pause_seconds <= 0:
            raise ValueError("pause_seconds must be greater than 0")
        if config.pre_roll_seconds < 0:
            raise ValueError("pre_roll_seconds must be 0 or greater")
        if config.min_speech_seconds <= 0:
            raise ValueError("min_speech_seconds must be greater than 0")
        if config.min_segment_seconds <= 0:
            raise ValueError("min_segment_seconds must be greater than 0")
        if config.max_segment_seconds < 0:
            raise ValueError("max_segment_seconds must be 0 or greater")
        if config.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be greater than 0")
        if config.partial_seconds < 0:
            raise ValueError("partial_seconds must be 0 or greater")
        if config.min_text_length < 0:
            raise ValueError("min_text_length must be 0 or greater")
        if config.beam_size <= 0:
            raise ValueError("beam_size must be greater than 0")
        if config.speaker_backend.casefold() not in SPEAKER_BACKENDS:
            raise ValueError(f"speaker_backend must be one of: {', '.join(SPEAKER_BACKENDS)}")
        if config.translation_target_language != "en":
            raise ValueError("translation_target_language must be 'en'")
        if config.selective_translation_backend not in SELECTIVE_TRANSLATION_BACKENDS:
            raise ValueError(
                "selective_translation_backend must be one of: "
                f"{', '.join(SELECTIVE_TRANSLATION_BACKENDS)}"
            )

    def _ensure_device(self, device_id: int | None) -> None:
        devices = self._device_provider()
        if not devices:
            raise RuntimeError("No audio input devices found.")
        if device_id is None:
            return
        if not any(device.id == device_id for device in devices):
            raise ValueError(f"Invalid input device ID {device_id}. Run with --list-devices to see valid IDs.")

    def _default_recorder(self, config: LiveTranscriberConfig) -> Recorder:
        from live_transcriber.audio import ContinuousAudioRecorder

        return ContinuousAudioRecorder(
            sample_rate=config.sample_rate,
            channels=1,
            device=config.device,
            frame_duration_ms=config.frame_duration_ms,
        )

    def _default_transcriber(self, config: LiveTranscriberConfig) -> Transcriber:
        from live_transcriber.transcriber import FasterWhisperTranscriber

        return FasterWhisperTranscriber(
            model_name=config.model,
            device_type=config.device_type,
            compute_type=config.compute_type,
            language=config.language,
            beam_size=config.beam_size,
            download_root=config.whisper_download_root,
        )

    def _default_writer(self, config: LiveTranscriberConfig):
        from live_transcriber.writer import TranscriptWriter

        return TranscriptWriter(
            text_path=config.text_path or DEFAULT_OUTPUT,
            jsonl_path=config.jsonl_path,
            overwrite=config.overwrite,
        )

    def _default_speaker_labeler(self, config: LiveTranscriberConfig) -> SpeakerLabeler | None:
        from live_transcriber.speakers import create_speaker_labeler

        return create_speaker_labeler(config)

    def _label_speaker(
        self,
        callbacks: LiveTranscriberCallbacks,
        audio,  # noqa: ANN001
        sample_rate: int,
        utterance_id: int,
    ):
        with self._speaker_lock:
            if self._speaker_labeler is None:
                return None
            try:
                return self._speaker_labeler.label(audio, sample_rate=sample_rate, utterance_id=utterance_id)
            except Exception as exc:  # noqa: BLE001
                self._speaker_labeler = None
                self._error(callbacks, f"Speaker labels disabled: {exc}")
                return None

    def _translate_final(
        self,
        callbacks: LiveTranscriberCallbacks,
        transcriber: Transcriber,
        audio,  # noqa: ANN001
        sample_rate: int,
    ) -> str | None:
        try:
            result = transcriber.translate(audio, sample_rate=sample_rate)
        except Exception as exc:  # noqa: BLE001
            self._error(callbacks, f"Full translation disabled: {exc}")
            return None

        from live_transcriber.text_cleanup import cleanup_transcript_text

        return cleanup_transcript_text(str(result.get("text", "")))

    def _status(self, callbacks: LiveTranscriberCallbacks, message: str) -> None:
        if callbacks.on_status is not None:
            callbacks.on_status(message)

    def _error(self, callbacks: LiveTranscriberCallbacks, message: str) -> None:
        if callbacks.on_error is not None:
            callbacks.on_error(message)
