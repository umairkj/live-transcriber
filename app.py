from __future__ import annotations

import argparse
import logging
import time
from threading import Thread
from datetime import datetime
from pathlib import Path

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


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously capture audio, segment speech on pauses, and transcribe locally with faster-whisper.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
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
        help=f"RMS threshold below which audio is treated as silence. Default: {DEFAULT_SILENCE_THRESHOLD}",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help=f"Silence duration that closes a speech segment. Default: {DEFAULT_PAUSE_SECONDS}",
    )
    parser.add_argument(
        "--pre-roll-seconds",
        type=float,
        default=DEFAULT_PRE_ROLL_SECONDS,
        help=f"Audio kept before speech starts so first words are not clipped. Default: {DEFAULT_PRE_ROLL_SECONDS}",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=DEFAULT_MIN_SPEECH_SECONDS,
        help=f"Minimum speech needed before opening a segment. Default: {DEFAULT_MIN_SPEECH_SECONDS}",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SECONDS,
        help=f"Minimum completed segment duration sent to Whisper. Default: {DEFAULT_MIN_SEGMENT_SECONDS}",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=DEFAULT_MAX_SEGMENT_SECONDS,
        help=f"Force-close long speech so output keeps moving. Use 0 to disable. Default: {DEFAULT_MAX_SEGMENT_SECONDS}",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=DEFAULT_FRAME_DURATION_MS,
        help=f"InputStream callback block duration. Default: {DEFAULT_FRAME_DURATION_MS}",
    )
    parser.add_argument(
        "--partial-seconds",
        type=float,
        default=DEFAULT_PARTIAL_SECONDS,
        help=f"Print provisional transcript snapshots every N seconds. Use 0 to disable. Default: {DEFAULT_PARTIAL_SECONDS}",
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
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Whisper beam size. Lower is faster. Default: {DEFAULT_BEAM_SIZE}",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--no-jsonl", action="store_true", help="Do not write JSONL transcript records.")
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to transcript files. This is the default unless --overwrite is set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Clear transcript files at startup, then append new records.")
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
    if args.chunk_seconds is not None:
        if args.chunk_seconds <= 0:
            raise ValueError("--chunk-seconds must be greater than 0")
        args.max_segment_seconds = args.chunk_seconds
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than 0")
    if args.silence_threshold < 0:
        raise ValueError("--silence-threshold must be 0 or greater")
    if args.pause_seconds <= 0:
        raise ValueError("--pause-seconds must be greater than 0")
    if args.pre_roll_seconds < 0:
        raise ValueError("--pre-roll-seconds must be 0 or greater")
    if args.min_speech_seconds <= 0:
        raise ValueError("--min-speech-seconds must be greater than 0")
    if args.min_segment_seconds <= 0:
        raise ValueError("--min-segment-seconds must be greater than 0")
    if args.max_segment_seconds < 0:
        raise ValueError("--max-segment-seconds must be 0 or greater")
    if args.frame_duration_ms <= 0:
        raise ValueError("--frame-duration-ms must be greater than 0")
    if args.partial_seconds < 0:
        raise ValueError("--partial-seconds must be 0 or greater")
    if args.min_text_length < 0:
        raise ValueError("--min-text-length must be 0 or greater")
    if args.beam_size <= 0:
        raise ValueError("--beam-size must be greater than 0")


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
            beam_size=args.beam_size,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    from live_transcriber.audio import ContinuousAudioRecorder
    from live_transcriber.deduper import RecentTextDeduper
    from live_transcriber.jobs import TranscriptionJob, TranscriptionJobQueue
    from live_transcriber.partials import PartialTextTracker
    from live_transcriber.segmentation import SpeechSegmenter
    from live_transcriber.text_cleanup import cleanup_transcript_text, is_garbage_fragment

    recorder = ContinuousAudioRecorder(
        sample_rate=args.sample_rate,
        channels=1,
        device=args.device,
        frame_duration_ms=args.frame_duration_ms,
    )
    segmenter = SpeechSegmenter(
        sample_rate=args.sample_rate,
        speech_threshold=args.silence_threshold,
        pause_seconds=args.pause_seconds,
        pre_roll_seconds=args.pre_roll_seconds,
        min_speech_seconds=args.min_speech_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
    )
    jobs = TranscriptionJobQueue()

    print("Listening...", flush=True)
    print(f"Transcript text file: {text_path}", flush=True)
    if jsonl_path is not None:
        print(f"JSONL file: {jsonl_path}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print("", flush=True)

    deduper = RecentTextDeduper()
    partial_tracker = PartialTextTracker()

    def transcribe_jobs() -> None:
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return

                if job.kind == "partial":
                    if jobs.is_completed(job.utterance_id):
                        continue

                    try:
                        result = transcriber.transcribe(job.audio, sample_rate=args.sample_rate)
                    except RuntimeError as exc:
                        logger.error("Partial transcription failed: %s", exc)
                        continue

                    if jobs.is_completed(job.utterance_id):
                        continue

                    text = cleanup_transcript_text(str(result.get("text", "")))
                    if is_garbage_fragment(text, args.min_text_length):
                        logger.debug("Skipping empty, too-short, or garbage partial transcription: %r", text)
                        continue

                    new_text = partial_tracker.update(job.utterance_id, text)
                    if new_text:
                        print(f"[partial] {new_text}", flush=True)
                    continue

                partial_tracker.finish(job.utterance_id)
                try:
                    result = transcriber.transcribe(job.audio, sample_rate=args.sample_rate)
                except RuntimeError as exc:
                    logger.error("Transcription failed for this segment: %s", exc)
                    continue

                text = cleanup_transcript_text(str(result.get("text", "")))
                if is_garbage_fragment(text, args.min_text_length):
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
                    "model": args.model,
                    "device_id": args.device,
                    "sample_rate": args.sample_rate,
                    "beam_size": args.beam_size,
                    "segments": result.get("segments", []),
                }

                print(text, flush=True)
                try:
                    writer.write_record(record)
                except OSError as exc:
                    logger.error("Could not write transcript record: %s", exc)
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
    try:
        with recorder:
            while True:
                frame = recorder.read(timeout=0.2)
                if frame is None:
                    continue
                if frame.status:
                    logger.debug("Audio callback status: %s", frame.status)

                for segment in segmenter.process(frame.audio):
                    enqueue_final(segment)

                if args.partial_seconds <= 0:
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

                if now - last_partial_at >= args.partial_seconds and try_enqueue_partial():
                    last_partial_at = now
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    except RuntimeError as exc:
        logger.error("%s", exc)
        exit_code = 1
    finally:
        final_segment = segmenter.flush()
        if final_segment is not None:
            enqueue_final(final_segment)
        jobs.put_stop()
        worker.join()
        print(f"Transcript text file: {text_path}", flush=True)
        if jsonl_path is not None:
            print(f"JSONL file: {jsonl_path}", flush=True)

    return exit_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
