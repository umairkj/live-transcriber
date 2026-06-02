from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL = "base"
DEFAULT_CHUNK_SECONDS = 6
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SILENCE_THRESHOLD = 0.003
DEFAULT_MIN_TEXT_LENGTH = 1
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_DEVICE_TYPE = "cpu"
DEFAULT_OUTPUT = Path("transcripts/transcript.txt")
DEFAULT_JSONL_OUTPUT = Path("transcripts/transcript.jsonl")
