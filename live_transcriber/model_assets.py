from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib import error, request


APP_ORGANIZATION = "LiveTranscriber"
APP_NAME = "Live Transcriber"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class AssetRole(str, Enum):
    TRANSCRIPTION = "transcription"
    MINUTES = "minutes"
    WORD_HINTS = "word_hints"
    DICTIONARY = "dictionary"
    SPACY = "spacy"
    AUDIO = "audio"


@dataclass(frozen=True)
class ModelAsset:
    asset_id: str
    label: str
    role: AssetRole
    provider: str
    description: str
    size_hint: str = ""
    recommended: bool = False
    install_command: str = ""


@dataclass(frozen=True)
class AssetStatus:
    asset_id: str
    installed: bool
    detail: str
    path: str | None = None


WHISPER_MODELS: tuple[ModelAsset, ...] = (
    ModelAsset(
        asset_id="tiny",
        label="Tiny",
        role=AssetRole.TRANSCRIPTION,
        provider="faster-whisper",
        description="Fastest startup and lowest CPU use. Best for quick tests.",
        size_hint="~75 MB",
    ),
    ModelAsset(
        asset_id="base",
        label="Base",
        role=AssetRole.TRANSCRIPTION,
        provider="faster-whisper",
        description="Best default balance for live captions on Apple silicon.",
        size_hint="~150 MB",
        recommended=True,
    ),
    ModelAsset(
        asset_id="small",
        label="Small",
        role=AssetRole.TRANSCRIPTION,
        provider="faster-whisper",
        description="Better German accuracy with more latency than base.",
        size_hint="~480 MB",
    ),
    ModelAsset(
        asset_id="medium",
        label="Medium",
        role=AssetRole.TRANSCRIPTION,
        provider="faster-whisper",
        description="High accuracy for slower or offline transcription.",
        size_hint="~1.5 GB",
    ),
    ModelAsset(
        asset_id="large-v3",
        label="Large v3",
        role=AssetRole.TRANSCRIPTION,
        provider="faster-whisper",
        description="Highest quality option; not ideal for low-latency MacBook Air use.",
        size_hint="~3 GB",
    ),
)


OLLAMA_MODELS: tuple[ModelAsset, ...] = (
    ModelAsset(
        asset_id="llama3.1:latest",
        label="Llama 3.1",
        role=AssetRole.MINUTES,
        provider="ollama",
        description="Default local model for meeting minutes and careful word hints.",
        size_hint="~4.7 GB",
        recommended=True,
        install_command="ollama pull llama3.1",
    ),
    ModelAsset(
        asset_id="llama3.2:3b",
        label="Llama 3.2 3B",
        role=AssetRole.MINUTES,
        provider="ollama",
        description="Smaller and faster local model if available in your Ollama library.",
        size_hint="~2 GB",
        install_command="ollama pull llama3.2:3b",
    ),
    ModelAsset(
        asset_id="mistral:7b",
        label="Mistral 7B",
        role=AssetRole.MINUTES,
        provider="ollama",
        description="Alternative general-purpose local model.",
        size_hint="~4 GB",
        install_command="ollama pull mistral:7b",
    ),
)


SUPPORT_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        asset_id="de_core_news_sm",
        label="German spaCy Model",
        role=AssetRole.SPACY,
        provider="spaCy",
        description="Improves noun and verb detection for live word hints.",
        size_hint="~15 MB",
        install_command=f"{sys.executable} -m pip install spacy && {sys.executable} -m spacy download de_core_news_sm",
    ),
    ModelAsset(
        asset_id="de_en.sqlite",
        label="German-English Dictionary",
        role=AssetRole.DICTIONARY,
        provider="FreeDict",
        description="Local SQLite dictionary for fast word hints without waiting for Ollama.",
        size_hint="~450 MB source download",
        install_command=f"{sys.executable} scripts/build_dictionary.py",
    ),
    ModelAsset(
        asset_id="blackhole-2ch",
        label="BlackHole 2ch",
        role=AssetRole.AUDIO,
        provider="Homebrew",
        description="Virtual audio device for transcribing Mac system audio.",
        install_command="brew install --cask blackhole-2ch",
    ),
)


def asset_catalog() -> tuple[ModelAsset, ...]:
    return (*WHISPER_MODELS, *OLLAMA_MODELS, *SUPPORT_ASSETS)


def default_app_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "live-transcriber"


def whisper_download_root(app_data_dir: Path) -> Path:
    return app_data_dir / "models" / "whisper"


def huggingface_hub_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def dictionary_path(app_data_dir: Path) -> Path:
    return app_data_dir / "data" / "de_en.sqlite"


def ensure_app_dirs(app_data_dir: Path) -> None:
    whisper_download_root(app_data_dir).mkdir(parents=True, exist_ok=True)
    dictionary_path(app_data_dir).parent.mkdir(parents=True, exist_ok=True)


def whisper_model_status(model_id: str, app_data_dir: Path) -> AssetStatus:
    app_path = _find_whisper_model_path(model_id, whisper_download_root(app_data_dir))
    if app_path is not None:
        return AssetStatus(model_id, True, "Ready in app cache", str(app_path))

    system_path = _find_whisper_model_path(model_id, huggingface_hub_cache_dir())
    if system_path is not None:
        return AssetStatus(model_id, True, "Ready in system cache", str(system_path))

    return AssetStatus(model_id, False, "Not downloaded")


def _find_whisper_model_path(model_id: str, root: Path) -> Path | None:
    if not root.exists():
        return None
    marker = f"faster-whisper-{model_id}".casefold()
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if ".locks" in path.parts:
            continue
        if marker in path.name.casefold():
            return path
    return None



def dictionary_status(app_data_dir: Path) -> AssetStatus:
    app_path = dictionary_path(app_data_dir)
    repo_path = Path("data/de_en.sqlite")
    if app_path.exists():
        return AssetStatus("de_en.sqlite", True, "Ready", str(app_path))
    if repo_path.exists():
        return AssetStatus("de_en.sqlite", True, "Ready in project folder", str(repo_path.resolve()))
    return AssetStatus("de_en.sqlite", False, "Not built")


def spacy_model_status() -> AssetStatus:
    try:
        import spacy
    except ModuleNotFoundError:
        return AssetStatus("de_core_news_sm", False, "spaCy package missing")

    try:
        spacy.load("de_core_news_sm", disable=["ner"])
    except OSError:
        return AssetStatus("de_core_news_sm", False, "German model missing")
    return AssetStatus("de_core_news_sm", True, "Ready")


def homebrew_path() -> str | None:
    return shutil.which("brew")


def blackhole_status(device_names: list[str] | None = None) -> AssetStatus:
    if device_names and any("blackhole" in name.casefold() for name in device_names):
        return AssetStatus("blackhole-2ch", True, "Visible as an input device")
    if Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver").exists():
        return AssetStatus("blackhole-2ch", True, "Installed. Restart audio apps if it is not visible.")
    return AssetStatus("blackhole-2ch", False, "Not installed or not visible")


def ollama_model_names(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> list[str]:
    req = request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def ollama_status(model_id: str, installed_names: list[str]) -> AssetStatus:
    if not installed_names:
        return AssetStatus(model_id, False, "Ollama unavailable or no models installed")

    normalized = {name.casefold() for name in installed_names}
    model_key = model_id.casefold()
    base_key = model_key.split(":", 1)[0]
    if model_key in normalized or base_key in normalized:
        return AssetStatus(model_id, True, "Available")
    return AssetStatus(model_id, False, "Not pulled")
