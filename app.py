from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from live_transcriber.config import (
    DEFAULT_ACTIVITY_FRAME_SECONDS,
    DEFAULT_ACTIVITY_POLL_SECONDS,
    DEFAULT_BLOCK_DURATION_MS,
    DEFAULT_BUFFER_SECONDS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEDUPE_MIN_OVERLAP_CHARS,
    DEFAULT_DEDUPE_MODE,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_JSONL_OUTPUT,
    DEFAULT_LIVE_TRANSCRIPT,
    DEFAULT_MIN_AUDIO_SECONDS,
    DEFAULT_MIN_FINAL_CHARS,
    DEFAULT_MIN_FINAL_WORDS,
    DEFAULT_MIN_NEW_CHARS,
    DEFAULT_MIN_TEXT_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_MAX_UTTERANCE_SECONDS,
    DEFAULT_MIN_SPEECH_SECONDS,
    DEFAULT_OUTPUT,
    DEFAULT_PAUSE_FINALIZE_SECONDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_STABILITY_SIMILARITY_THRESHOLD,
    DEFAULT_STABILITY_WINDOWS,
    DEFAULT_STEP_SECONDS,
    DEFAULT_UTTERANCE_PADDING_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_RECENT_DUPLICATE_THRESHOLD,
)


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record short audio chunks and transcribe them locally with faster-whisper.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Streaming mode rolling audio window sent to Whisper. Default: {DEFAULT_WINDOW_SECONDS}",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=DEFAULT_STEP_SECONDS,
        help=f"Streaming mode interval between Whisper runs. Default: {DEFAULT_STEP_SECONDS}",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=DEFAULT_BUFFER_SECONDS,
        help=f"Streaming mode rolling buffer length. Default: {DEFAULT_BUFFER_SECONDS}",
    )
    parser.add_argument(
        "--min-audio-seconds",
        type=float,
        default=DEFAULT_MIN_AUDIO_SECONDS,
        help=f"Do not transcribe until this much audio is buffered. Default: {DEFAULT_MIN_AUDIO_SECONDS}",
    )
    parser.add_argument(
        "--block-duration-ms",
        type=int,
        default=DEFAULT_BLOCK_DURATION_MS,
        help=f"Audio callback block size in milliseconds. Default: {DEFAULT_BLOCK_DURATION_MS}",
    )
    parser.add_argument(
        "--activity-poll-seconds",
        type=float,
        default=DEFAULT_ACTIVITY_POLL_SECONDS,
        help=f"How often to scan buffered audio for speech/pause boundaries. Default: {DEFAULT_ACTIVITY_POLL_SECONDS}",
    )
    parser.add_argument(
        "--activity-frame-seconds",
        type=float,
        default=DEFAULT_ACTIVITY_FRAME_SECONDS,
        help=f"Speech activity RMS frame size. Default: {DEFAULT_ACTIVITY_FRAME_SECONDS}",
    )
    parser.add_argument(
        "--pause-finalize-seconds",
        type=float,
        default=DEFAULT_PAUSE_FINALIZE_SECONDS,
        help=f"Finalize an utterance after this much silence. Default: {DEFAULT_PAUSE_FINALIZE_SECONDS}",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=DEFAULT_MIN_SPEECH_SECONDS,
        help=f"Minimum speech duration before opening an utterance. Default: {DEFAULT_MIN_SPEECH_SECONDS}",
    )
    parser.add_argument(
        "--utterance-padding-seconds",
        type=float,
        default=DEFAULT_UTTERANCE_PADDING_SECONDS,
        help=f"Audio padding included before/after pause-finalized utterances. Default: {DEFAULT_UTTERANCE_PADDING_SECONDS}",
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=DEFAULT_MAX_UTTERANCE_SECONDS,
        help=f"Force-finalize very long speech after this duration. Use 0 to disable. Default: {DEFAULT_MAX_UTTERANCE_SECONDS}",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=("none", "lcp", "similarity"),
        default=DEFAULT_DEDUPE_MODE,
        help=f"Deduplication mode for overlapping windows. Default: {DEFAULT_DEDUPE_MODE}",
    )
    parser.add_argument(
        "--min-new-chars",
        type=int,
        default=DEFAULT_MIN_NEW_CHARS,
        help=f"Minimum unique characters before emitting text. Default: {DEFAULT_MIN_NEW_CHARS}",
    )
    parser.add_argument("--emit-partials", action="store_true", help="Print partial text updates without saving them as final transcript.")
    parser.add_argument(
        "--require-sentence-end",
        dest="require_sentence_end",
        action="store_true",
        default=True,
        help="Only save stable text after sentence-ending punctuation. This is the default.",
    )
    parser.add_argument(
        "--no-require-sentence-end",
        dest="require_sentence_end",
        action="store_false",
        help="Allow stable non-sentence text to be emitted.",
    )
    parser.add_argument(
        "--dedupe-min-overlap-chars",
        type=int,
        default=DEFAULT_DEDUPE_MIN_OVERLAP_CHARS,
        help=f"Minimum fuzzy overlap characters for dedupe. Default: {DEFAULT_DEDUPE_MIN_OVERLAP_CHARS}",
    )
    parser.add_argument(
        "--stability-windows",
        type=int,
        default=DEFAULT_STABILITY_WINDOWS,
        help=f"Windows a candidate must appear in before final emission. Default: {DEFAULT_STABILITY_WINDOWS}",
    )
    parser.add_argument(
        "--stability-similarity-threshold",
        type=float,
        default=DEFAULT_STABILITY_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for candidate stability. Default: {DEFAULT_STABILITY_SIMILARITY_THRESHOLD}",
    )
    parser.add_argument(
        "--recent-duplicate-threshold",
        type=float,
        default=DEFAULT_RECENT_DUPLICATE_THRESHOLD,
        help=f"Similarity threshold for recent duplicate suppression. Default: {DEFAULT_RECENT_DUPLICATE_THRESHOLD}",
    )
    parser.add_argument(
        "--min-final-words",
        type=int,
        default=DEFAULT_MIN_FINAL_WORDS,
        help=f"Minimum words for final sentences. Default: {DEFAULT_MIN_FINAL_WORDS}",
    )
    parser.add_argument(
        "--min-final-chars",
        type=int,
        default=DEFAULT_MIN_FINAL_CHARS,
        help=f"Minimum characters for final sentences. Default: {DEFAULT_MIN_FINAL_CHARS}",
    )
    parser.add_argument(
        "--allow-short-utterances",
        dest="allow_short_utterances",
        action="store_true",
        default=True,
        help="Allow common short utterances like Ja. and Genau. This is the default.",
    )
    parser.add_argument(
        "--no-allow-short-utterances",
        dest="allow_short_utterances",
        action="store_false",
        help="Reject common short utterances.",
    )
    parser.add_argument(
        "--debug-raw-windows",
        action="store_true",
        help="Print raw Whisper rolling-window hypotheses with a debug prefix.",
    )
    parser.add_argument("--debug-stabilizer", action="store_true", help="Print stabilizer candidate/rejection debug output.")
    parser.add_argument("--show-live-hypotheses", action="store_true", help="Print live rolling-window hypotheses before pause-finalized text.")
    parser.add_argument(
        "--live-transcript",
        dest="live_transcript",
        action="store_true",
        default=DEFAULT_LIVE_TRANSCRIPT,
        help="Update the current transcript line while speech is still in progress. This is the default.",
    )
    parser.add_argument(
        "--no-live-transcript",
        dest="live_transcript",
        action="store_false",
        help="Disable live line updates and print only pause-finalized transcript lines.",
    )
    parser.add_argument("--device", type=int, default=None, help="Optional sounddevice input device ID.")
    parser.add_argument("--language", default=None, help='Optional language hint, for example "de" or "en".')
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Text transcript path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT),
        help=f"JSONL transcript path. Default: {DEFAULT_JSONL_OUTPUT}",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Audio sample rate. Default: {DEFAULT_SAMPLE_RATE}",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help=f"RMS threshold below which chunks are skipped. Default: {DEFAULT_SILENCE_THRESHOLD}",
    )
    parser.add_argument(
        "--no-silence-skip",
        action="store_true",
        help="Transcribe every captured chunk, even when it looks silent.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=DEFAULT_MIN_TEXT_LENGTH,
        help=f"Ignore transcript text shorter than this. Default: {DEFAULT_MIN_TEXT_LENGTH}",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help=f"faster-whisper compute type. Default: {DEFAULT_COMPUTE_TYPE}",
    )
    parser.add_argument(
        "--device-type",
        default=DEFAULT_DEVICE_TYPE,
        help=f"faster-whisper device type. Default: {DEFAULT_DEVICE_TYPE}",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--no-jsonl", action="store_true", help="Do not write JSONL transcript records.")
    parser.add_argument("--vocab-mode", action="store_true", help="Print vocabulary placeholder output for stable complete sentences.")
    parser.add_argument("--vocab-output", default=None, help="Optional path for vocabulary output lines.")
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to transcript files. This is the default unless --overwrite is set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Clear transcript files at startup, then append new records.")
    parser.add_argument(
        "--rewrite-transcript",
        dest="rewrite_transcript",
        action="store_true",
        default=True,
        help="Rewrite current-run text transcript lines when pause-finalized corrections happen. This is the default.",
    )
    parser.add_argument(
        "--no-rewrite-transcript",
        dest="rewrite_transcript",
        action="store_false",
        help="Append final transcript lines without rewriting current-run text.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not verbose:
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        logging.getLogger("ctranslate2").setLevel(logging.WARNING)


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than 0")
    if args.silence_threshold < 0:
        raise ValueError("--silence-threshold must be 0 or greater")
    if args.min_text_length < 0:
        raise ValueError("--min-text-length must be 0 or greater")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0")
    if args.step_seconds <= 0:
        raise ValueError("--step-seconds must be greater than 0")
    if args.buffer_seconds < args.window_seconds:
        raise ValueError("--buffer-seconds must be greater than or equal to --window-seconds")
    if args.min_audio_seconds <= 0:
        raise ValueError("--min-audio-seconds must be greater than 0")
    if args.min_audio_seconds > args.window_seconds:
        raise ValueError("--min-audio-seconds must be less than or equal to --window-seconds")
    if args.block_duration_ms <= 0:
        raise ValueError("--block-duration-ms must be greater than 0")
    if args.activity_poll_seconds <= 0:
        raise ValueError("--activity-poll-seconds must be greater than 0")
    if args.activity_frame_seconds <= 0:
        raise ValueError("--activity-frame-seconds must be greater than 0")
    if args.pause_finalize_seconds < 0:
        raise ValueError("--pause-finalize-seconds must be 0 or greater")
    if args.min_speech_seconds < 0:
        raise ValueError("--min-speech-seconds must be 0 or greater")
    if args.utterance_padding_seconds < 0:
        raise ValueError("--utterance-padding-seconds must be 0 or greater")
    if args.max_utterance_seconds < 0:
        raise ValueError("--max-utterance-seconds must be 0 or greater")
    if args.min_new_chars < 0:
        raise ValueError("--min-new-chars must be 0 or greater")
    if args.dedupe_min_overlap_chars < 0:
        raise ValueError("--dedupe-min-overlap-chars must be 0 or greater")
    if args.stability_windows <= 0:
        raise ValueError("--stability-windows must be greater than 0")
    if not 0 <= args.stability_similarity_threshold <= 1:
        raise ValueError("--stability-similarity-threshold must be between 0 and 1")
    if not 0 <= args.recent_duplicate_threshold <= 1:
        raise ValueError("--recent-duplicate-threshold must be between 0 and 1")
    if args.min_final_words < 0:
        raise ValueError("--min-final-words must be 0 or greater")
    if args.min_final_chars < 0:
        raise ValueError("--min-final-chars must be 0 or greater")

class StreamingTranscriptionWorker(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        audio_stream,
        transcriber,
        writer,
        stop_event: threading.Event,
        vocab_output_path: Path | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.audio_stream = audio_stream
        self.transcriber = transcriber
        self.writer = writer
        self.stop_event = stop_event
        self.vocab_output_path = vocab_output_path

        from live_transcriber.live_transcript import LiveTranscriptDisplay
        from live_transcriber.models import TranscriptSession
        from live_transcriber.speech_activity import SpeechActivityDetector
        from live_transcriber.stabilizer import TranscriptStabilizer

        self.window_stabilizer = TranscriptStabilizer(
            min_new_chars=args.min_new_chars,
            emit_partials=args.emit_partials,
            require_sentence_end=args.require_sentence_end,
            dedupe_min_overlap_chars=args.dedupe_min_overlap_chars,
            stability_windows=args.stability_windows,
            stability_similarity_threshold=args.stability_similarity_threshold,
            recent_duplicate_threshold=args.recent_duplicate_threshold,
            min_final_words=args.min_final_words,
            min_final_chars=args.min_final_chars,
            allow_short_utterances=args.allow_short_utterances,
            debug=args.debug_stabilizer,
        )
        self.final_stabilizer = TranscriptStabilizer(
            min_new_chars=args.min_new_chars,
            emit_partials=False,
            require_sentence_end=args.require_sentence_end,
            dedupe_min_overlap_chars=args.dedupe_min_overlap_chars,
            stability_windows=args.stability_windows,
            stability_similarity_threshold=args.stability_similarity_threshold,
            recent_duplicate_threshold=args.recent_duplicate_threshold,
            min_final_words=args.min_final_words,
            min_final_chars=args.min_final_chars,
            allow_short_utterances=args.allow_short_utterances,
            debug=args.debug_stabilizer,
        )
        self.activity_detector = SpeechActivityDetector(
            threshold=args.silence_threshold,
            min_speech_seconds=args.min_speech_seconds,
            pause_finalize_seconds=args.pause_finalize_seconds,
            max_utterance_seconds=args.max_utterance_seconds,
        )
        self.transcript_session = TranscriptSession()
        self.current_utterance = None
        self.last_activity_position = 0.0
        self.last_partial_print = ""
        self.last_live_hypothesis_print = ""
        self.live_transcript_display = LiveTranscriptDisplay(enabled=args.live_transcript)
        self.uses_live_windows = (
            args.live_transcript
            or args.show_live_hypotheses
            or args.emit_partials
            or args.debug_raw_windows
            or args.debug_stabilizer
        )

    def run(self) -> None:
        next_live_transcribe_at = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self.process_activity()
            except RuntimeError as exc:
                logger.error("Speech activity processing failed: %s", exc)
            except Exception:
                logger.exception("Unexpected speech activity processing error.")

            if self.uses_live_windows and time.monotonic() >= next_live_transcribe_at:
                started = time.monotonic()
                try:
                    self.process_live_window()
                except RuntimeError as exc:
                    logger.error("Streaming transcription failed for this window: %s", exc)
                except Exception:
                    logger.exception("Unexpected streaming transcription error.")

                elapsed = time.monotonic() - started
                if elapsed > self.args.step_seconds:
                    logger.debug(
                        "Transcription took %.1fs, longer than step interval %.1fs. "
                        "Consider using --model tiny or increasing --step-seconds.",
                        elapsed,
                        self.args.step_seconds,
                    )
                next_live_transcribe_at = max(started + self.args.step_seconds, time.monotonic())

            self.stop_event.wait(self.args.activity_poll_seconds)

        self.flush_current_speech()

    def process_activity(self) -> None:
        from live_transcriber.audio import calculate_rms

        current_time = self.audio_stream.current_time_seconds()
        buffered_start = max(0.0, current_time - self.audio_stream.duration_seconds())
        if self.last_activity_position <= 0:
            self.last_activity_position = buffered_start
        else:
            self.last_activity_position = max(self.last_activity_position, buffered_start)

        if current_time <= self.last_activity_position:
            return

        cursor = self.last_activity_position
        while cursor < current_time:
            frame_end = min(current_time, cursor + self.args.activity_frame_seconds)
            audio = self.audio_stream.get_range(cursor, frame_end)
            if audio.size:
                rms = calculate_rms(audio)
                for event in self.activity_detector.observe_frame(cursor, frame_end, rms):
                    self.handle_activity_event(event)
            cursor = frame_end

        self.last_activity_position = current_time

    def handle_activity_event(self, event: Any) -> None:
        from live_transcriber.models import UtteranceState

        if event.event_type == "speech_start":
            self.current_utterance = UtteranceState(
                start_time=event.start_time,
                end_time=event.end_time,
            )
            return

        if event.event_type == "speech" and self.current_utterance is not None:
            self.current_utterance.end_time = max(self.current_utterance.end_time, event.end_time)
            return

        if event.event_type != "speech_end":
            return

        utterance = self.current_utterance
        if utterance is None:
            utterance = UtteranceState(start_time=event.start_time, end_time=event.end_time)
        utterance.start_time = min(utterance.start_time, event.start_time)
        utterance.end_time = max(utterance.end_time, event.end_time)
        self.current_utterance = None
        self.finalize_utterance(utterance, boundary_reason=event.reason or "pause")

    def process_live_window(self) -> None:
        from live_transcriber.audio import calculate_rms
        from live_transcriber.sentence_utils import clean_whisper_artifacts

        if self.audio_stream.duration_seconds() < self.args.min_audio_seconds:
            return

        if self.args.live_transcript and self.current_utterance is None:
            return

        if self.args.live_transcript and self.current_utterance is not None:
            window_end = self.audio_stream.current_time_seconds()
            utterance_start = max(0.0, self.current_utterance.start_time - self.args.utterance_padding_seconds)
            window_start = max(utterance_start, window_end - self.args.window_seconds)
            audio = self.audio_stream.get_range(window_start, window_end)
        else:
            audio, window_start, window_end = self.audio_stream.get_last_with_range(self.args.window_seconds)

        duration_seconds = round(float(len(audio)) / float(self.args.sample_rate), 3)
        if duration_seconds < self.args.min_audio_seconds:
            return

        rms = calculate_rms(audio)
        if not self.args.live_transcript and not self.args.no_silence_skip and rms < self.args.silence_threshold:
            return

        result = self.transcriber.transcribe(audio, sample_rate=self.args.sample_rate)
        raw_text = str(result.get("text", "")).strip()
        if not raw_text:
            return

        cleaned_text = clean_whisper_artifacts(raw_text)
        if self.current_utterance is not None and cleaned_text:
            self.current_utterance.hypothesis_text = cleaned_text

        stable = self.window_stabilizer.process_window_text(raw_text)
        if self.args.debug_raw_windows:
            print(f"[raw {window_start:.2f}-{window_end:.2f}] {stable['raw_window_text']}", flush=True)
        if self.args.debug_stabilizer:
            print(f"[stabilizer] candidates={stable.get('candidate_sentences', [])}", flush=True)
            for rejected in stable.get("rejected_candidates", []):
                print(f"[stabilizer rejected:{rejected.get('reason')}] {rejected.get('text')}", flush=True)

        partial_text = str(stable.get("partial_text", "")).strip()
        if self.args.emit_partials and partial_text and partial_text != self.last_partial_print:
            self.last_partial_print = partial_text
            print(f"[partial] {partial_text}", flush=True)

        if self.args.live_transcript and self.current_utterance is not None:
            self.live_transcript_display.update(cleaned_text)
        elif self.args.show_live_hypotheses and cleaned_text and cleaned_text != self.last_live_hypothesis_print:
            self.last_live_hypothesis_print = cleaned_text
            print(f"[live] {cleaned_text}", flush=True)

    def finish_live_line(self) -> None:
        self.live_transcript_display.finish()

    def flush_current_speech(self) -> None:
        current_time = self.audio_stream.current_time_seconds()
        for event in self.activity_detector.flush(current_time):
            self.handle_activity_event(event)

    def finalize_utterance(self, utterance: Any, boundary_reason: str) -> None:
        from live_transcriber.audio import calculate_rms
        from live_transcriber.models import TranscriptEvent
        from live_transcriber.sentence_utils import normalize_text_for_compare

        audio_start = max(0.0, utterance.start_time - self.args.utterance_padding_seconds)
        audio_end = utterance.end_time + self.args.utterance_padding_seconds
        audio = self.audio_stream.get_range(audio_start, audio_end)
        duration_seconds = round(float(len(audio)) / float(self.args.sample_rate), 3)
        if duration_seconds <= 0:
            return

        rms = calculate_rms(audio)
        result = self.transcriber.transcribe(audio, sample_rate=self.args.sample_rate)
        raw_text = str(result.get("text", "")).strip()
        if not raw_text:
            self.live_transcript_display.commit(utterance.hypothesis_text)
            return

        stable = self.final_stabilizer.finalize_utterance_text(raw_text)
        if self.args.debug_stabilizer:
            print(f"[utterance raw] {raw_text}", flush=True)
            print(f"[utterance candidates] {stable.get('candidate_sentences', [])}", flush=True)
            for rejected in stable.get("rejected_candidates", []):
                print(f"[utterance rejected:{rejected.get('reason')}] {rejected.get('text')}", flush=True)

        emitted_text = str(stable.get("emitted_text", "")).strip()
        if len(emitted_text) < self.args.min_text_length:
            self.live_transcript_display.commit(utterance.hypothesis_text or raw_text)
            return

        utterance.final_text = emitted_text
        event = self.transcript_session.upsert_final(utterance)
        if utterance.hypothesis_text and normalize_text_for_compare(utterance.hypothesis_text) != normalize_text_for_compare(
            emitted_text
        ):
            event = TranscriptEvent(
                event_type="correction",
                utterance=event.utterance,
                text=event.text,
                previous_text=utterance.hypothesis_text,
                timestamp=event.timestamp,
            )

        event.metadata.update(
            {
                "language": result.get("language"),
                "language_probability": result.get("language_probability"),
                "duration_seconds": duration_seconds,
                "silence_skip_enabled": not self.args.no_silence_skip,
                "model": self.args.model,
                "device_id": self.args.device,
                "segments": result.get("segments", []),
                "streaming_mode": True,
                "pause_finalized": True,
                "boundary_reason": boundary_reason,
                "window_seconds": self.args.window_seconds,
                "step_seconds": self.args.step_seconds,
                "activity_poll_seconds": self.args.activity_poll_seconds,
                "activity_frame_seconds": self.args.activity_frame_seconds,
                "pause_finalize_seconds": self.args.pause_finalize_seconds,
                "min_speech_seconds": self.args.min_speech_seconds,
                "utterance_padding_seconds": self.args.utterance_padding_seconds,
                "utterance_audio_start": round(audio_start, 3),
                "utterance_audio_end": round(audio_end, 3),
                "emitted_text": emitted_text,
                "raw_window_text": stable.get("raw_window_text", ""),
                "cleaned_window_text": stable.get("cleaned_window_text", ""),
                "candidate_sentences": stable.get("candidate_sentences", []),
                "rejected_candidates": stable.get("rejected_candidates", []) if self.args.debug_stabilizer else [],
                "stable_sentences": stable.get("stable_sentences", []),
                "partial_text": "",
                "deduped_new_text": stable.get("new_text", ""),
                "is_partial": False,
                "vocab_line": "",
                "vocab_items": [],
                "rms": rms,
            }
        )
        self.emit_transcript_event(event)

    def emit_transcript_event(self, event: Any) -> None:
        from live_transcriber.vocabulary import create_vocabulary_line

        emitted_text = str(event.text).strip()
        vocab_line = ""
        vocab_items: list[dict[str, Any]] = []
        if self.args.vocab_mode:
            vocab_line = create_vocabulary_line(emitted_text)
            event.metadata["vocab_line"] = vocab_line
            event.metadata["vocab_items"] = vocab_items

        if self.args.vocab_mode and vocab_line:
            self.finish_live_line()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]", flush=True)
            print(vocab_line, flush=True)
            print(emitted_text, flush=True)
        elif self.args.live_transcript:
            self.live_transcript_display.commit(emitted_text, timestamp=event.timestamp)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {emitted_text}", flush=True)

        if vocab_line and self.vocab_output_path is not None:
            self.vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.vocab_output_path.open("a", encoding="utf-8") as file:
                file.write(vocab_line + "\n")
                file.flush()

        if self.args.rewrite_transcript:
            self.writer.rewrite_text_lines(self.transcript_session.final_lines())
        else:
            self.writer.write_text(emitted_text, metadata={"timestamp": event.timestamp})
        self.writer.write_jsonl(event.to_record())


def run_streaming_mode(
    args: argparse.Namespace,
    transcriber,
    writer,
    text_path: Path,
    jsonl_path: Path | None,
) -> int:
    from live_transcriber.stream import ContinuousAudioStream

    vocab_output_path = Path(args.vocab_output) if args.vocab_output else None
    audio_stream = ContinuousAudioStream(
        sample_rate=args.sample_rate,
        channels=1,
        device=args.device,
        block_duration_ms=args.block_duration_ms,
        buffer_max_seconds=args.buffer_seconds,
    )
    stop_event = threading.Event()
    worker = StreamingTranscriptionWorker(
        args=args,
        audio_stream=audio_stream,
        transcriber=transcriber,
        writer=writer,
        stop_event=stop_event,
        vocab_output_path=vocab_output_path,
    )

    print("Starting continuous audio stream...", flush=True)
    audio_stream.start()
    worker.start()

    print("Streaming mode: live transcript", flush=True)
    print(f"Pause finalize seconds: {args.pause_finalize_seconds:g}", flush=True)
    print(f"Activity poll seconds: {args.activity_poll_seconds:g}", flush=True)
    if args.live_transcript or args.show_live_hypotheses or args.emit_partials or args.debug_raw_windows or args.debug_stabilizer:
        print(f"Live window seconds: {args.window_seconds:g}", flush=True)
        print(f"Live step seconds: {args.step_seconds:g}", flush=True)
    print(f"Buffer seconds: {args.buffer_seconds:g}", flush=True)
    print(f"Transcript text file: {text_path}", flush=True)
    if jsonl_path is not None:
        print(f"JSONL file: {jsonl_path}", flush=True)
    if vocab_output_path is not None:
        print(f"Vocabulary file: {vocab_output_path}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print("", flush=True)

    try:
        while not stop_event.wait(0.25):
            pass
    except KeyboardInterrupt:
        stop_event.set()
        worker.finish_live_line()
        print("\nStopping...", flush=True)
    finally:
        stop_event.set()
        worker.finish_live_line()
        worker.join(timeout=max(10.0, args.step_seconds + args.pause_finalize_seconds + 2.0))
        worker.finish_live_line()
        audio_stream.stop()
        print(f"Transcript text file: {text_path}", flush=True)
        if jsonl_path is not None:
            print(f"JSONL file: {jsonl_path}", flush=True)
        if vocab_output_path is not None:
            print(f"Vocabulary file: {vocab_output_path}", flush=True)

    return 0


def run(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)

    if args.list_devices:
        from live_transcriber.device_utils import list_input_devices

        try:
            list_input_devices()
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    from live_transcriber.device_utils import ensure_input_devices_available, validate_input_device

    try:
        validate_args(args)
        ensure_input_devices_available()
        validate_input_device(args.device)
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    text_path = Path(args.output)
    jsonl_path = None if args.no_jsonl else Path(args.jsonl_output)

    from live_transcriber.writer import TranscriptWriter

    try:
        writer = TranscriptWriter(text_path=text_path, jsonl_path=jsonl_path, overwrite=args.overwrite)
    except OSError as exc:
        logger.error("Could not prepare transcript files: %s", exc)
        return 1

    print(f"Loading model: {args.model}", flush=True)
    from live_transcriber.transcriber import FasterWhisperTranscriber

    try:
        transcriber = FasterWhisperTranscriber(
            model_name=args.model,
            device_type=args.device_type,
            compute_type=args.compute_type,
            language=args.language,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    try:
        return run_streaming_mode(args, transcriber, writer, text_path, jsonl_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
