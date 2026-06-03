from __future__ import annotations

import argparse
import logging
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
from live_transcriber.session import LiveTranscriberCallbacks, LiveTranscriberConfig, LiveTranscriberSession
from live_transcriber.speakers import DEFAULT_SPEAKER_BACKEND, SPEAKER_BACKENDS
from live_transcriber.translations import DEFAULT_TRANSLATION_TARGET_LANGUAGE


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
    parser.add_argument(
        "--speaker-labels",
        action="store_true",
        help="Label finalized transcript lines with anonymous speaker IDs like A: and B:. Partials stay unlabeled.",
    )
    parser.add_argument(
        "--speaker-backend",
        choices=SPEAKER_BACKENDS,
        default=DEFAULT_SPEAKER_BACKEND,
        help=f"Speaker-labeling backend. Default: {DEFAULT_SPEAKER_BACKEND}",
    )
    parser.add_argument(
        "--full-translation",
        action="store_true",
        help="Translate finalized speech segments to English with Whisper. Partials are not translated.",
    )
    parser.add_argument(
        "--selective-translation",
        action="store_true",
        help="Add English word hints for selected nouns, verbs, and other useful words on finalized lines.",
    )
    parser.add_argument(
        "--translation-target",
        default=DEFAULT_TRANSLATION_TARGET_LANGUAGE,
        choices=[DEFAULT_TRANSLATION_TARGET_LANGUAGE],
        help=f"Translation target language. Default: {DEFAULT_TRANSLATION_TARGET_LANGUAGE}",
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
    if args.speaker_backend not in SPEAKER_BACKENDS:
        raise ValueError(f"--speaker-backend must be one of: {', '.join(SPEAKER_BACKENDS)}")
    if args.translation_target != DEFAULT_TRANSLATION_TARGET_LANGUAGE:
        raise ValueError("--translation-target must be en")


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

    try:
        validate_args(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    config = LiveTranscriberConfig(
        model=args.model,
        device=args.device,
        language=args.language,
        text_path=Path(args.output),
        jsonl_path=None if args.no_jsonl else Path(args.jsonl_output),
        save_transcript=True,
        overwrite=args.overwrite,
        sample_rate=args.sample_rate,
        silence_threshold=args.silence_threshold,
        pause_seconds=args.pause_seconds,
        pre_roll_seconds=args.pre_roll_seconds,
        min_speech_seconds=args.min_speech_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
        frame_duration_ms=args.frame_duration_ms,
        partial_seconds=args.partial_seconds,
        min_text_length=args.min_text_length,
        compute_type=args.compute_type,
        device_type=args.device_type,
        beam_size=args.beam_size,
        speaker_labels=args.speaker_labels,
        speaker_backend=args.speaker_backend,
        full_translation=args.full_translation,
        selective_translation=args.selective_translation,
        translation_target_language=args.translation_target,
    )

    def on_status(message: str) -> None:
        if message == "Listening":
            print("Listening...", flush=True)
            print(f"Transcript text file: {config.text_path}", flush=True)
            if config.jsonl_path is not None:
                print(f"JSONL file: {config.jsonl_path}", flush=True)
            print("Press Ctrl+C to stop.", flush=True)
            print("", flush=True)
        elif message == "Stopping":
            print("\nStopping...", flush=True)
        elif message != "Stopped":
            print(message, flush=True)

    def on_final_text(text: str, record: dict) -> None:
        print(text, flush=True)
        translation_text = record.get("translation_text")
        if translation_text:
            print(f"    EN: {translation_text}", flush=True)
        selective_translations = record.get("selective_translations") or []
        if selective_translations:
            hints = ", ".join(
                f"{item.get('source')}={item.get('translation')}" for item in selective_translations[:10]
            )
            print(f"    words: {hints}", flush=True)

    callbacks = LiveTranscriberCallbacks(
        on_status=on_status,
        on_partial_text=lambda text: print(f"[partial] {text}", flush=True),
        on_final_text=on_final_text,
        on_error=lambda message: logger.error("%s", message),
    )

    session = LiveTranscriberSession()
    exit_code = session.run_blocking(config, callbacks)

    print(f"Transcript text file: {config.text_path}", flush=True)
    if config.jsonl_path is not None:
        print(f"JSONL file: {config.jsonl_path}", flush=True)

    return exit_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
