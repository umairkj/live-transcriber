from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from live_transcriber.config import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_JSONL_OUTPUT,
    DEFAULT_MIN_TEXT_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
)


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record short audio chunks and transcribe them locally with faster-whisper.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Seconds of audio to record per chunk. Default: {DEFAULT_CHUNK_SECONDS}",
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
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than 0")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than 0")
    if args.silence_threshold < 0:
        raise ValueError("--silence-threshold must be 0 or greater")
    if args.min_text_length < 0:
        raise ValueError("--min-text-length must be 0 or greater")


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

    from live_transcriber.audio import AudioRecorder

    recorder = AudioRecorder(
        sample_rate=args.sample_rate,
        channels=1,
        device=args.device,
        silence_threshold=args.silence_threshold,
    )

    print("Listening...", flush=True)
    print(f"Transcript text file: {text_path}", flush=True)
    if jsonl_path is not None:
        print(f"JSONL file: {jsonl_path}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print("", flush=True)

    try:
        while True:
            try:
                audio = recorder.record_chunk(args.chunk_seconds)
            except RuntimeError as exc:
                logger.error("%s", exc)
                continue

            if recorder.is_silent(audio):
                logger.debug("Skipping silent chunk.")
                continue

            try:
                result = transcriber.transcribe(audio, sample_rate=args.sample_rate)
            except RuntimeError as exc:
                logger.error("Transcription failed for this chunk: %s", exc)
                continue

            text = str(result.get("text", "")).strip()
            if len(text) < args.min_text_length:
                logger.debug("Skipping empty or too-short transcription.")
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration_seconds = round(float(len(audio)) / float(args.sample_rate), 3)
            record = {
                "timestamp": timestamp,
                "text": text,
                "language": result.get("language"),
                "language_probability": result.get("language_probability"),
                "duration_seconds": duration_seconds,
                "chunk_seconds": args.chunk_seconds,
                "model": args.model,
                "device_id": args.device,
                "segments": result.get("segments", []),
            }

            print(text, flush=True)
            try:
                writer.write_record(record)
            except OSError as exc:
                logger.error("Could not write transcript record: %s", exc)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        print(f"Transcript text file: {text_path}", flush=True)
        if jsonl_path is not None:
            print(f"JSONL file: {jsonl_path}", flush=True)

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
