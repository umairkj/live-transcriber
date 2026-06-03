from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.model_assets import (  # noqa: E402
    OLLAMA_MODELS,
    WHISPER_MODELS,
    blackhole_status,
    dictionary_path,
    dictionary_status,
    ensure_app_dirs,
    ollama_status,
    whisper_download_root,
    whisper_model_status,
)


def main() -> None:
    assert any(model.asset_id == "base" and model.recommended for model in WHISPER_MODELS)
    assert any(model.asset_id == "llama3.1:latest" and model.recommended for model in OLLAMA_MODELS)

    old_hf_home = os.environ.get("HF_HOME")
    with tempfile.TemporaryDirectory() as temp_dir:
        app_data = Path(temp_dir)
        os.environ["HF_HOME"] = str(app_data / "hf")
        ensure_app_dirs(app_data)
        assert whisper_download_root(app_data).exists()
        assert dictionary_path(app_data).parent.exists()

        missing_whisper = whisper_model_status("base", app_data)
        assert not missing_whisper.installed

        fake_model = whisper_download_root(app_data) / "models--Systran--faster-whisper-base"
        fake_model.mkdir(parents=True)
        installed_whisper = whisper_model_status("base", app_data)
        assert installed_whisper.installed
        assert installed_whisper.detail == "Ready in app cache"
        assert installed_whisper.path == str(fake_model)

        assert not dictionary_status(app_data).installed
        dictionary_path(app_data).write_bytes(b"sqlite")
        assert dictionary_status(app_data).installed

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["HF_HOME"] = temp_dir
        app_data = Path(temp_dir) / "app"
        ensure_app_dirs(app_data)
        lock_model = Path(temp_dir) / "hub" / ".locks" / "models--Systran--faster-whisper-small"
        lock_model.mkdir(parents=True)
        assert not whisper_model_status("small", app_data).installed

        system_model = Path(temp_dir) / "hub" / "models--Systran--faster-whisper-small"
        system_model.mkdir(parents=True)
        system_status = whisper_model_status("small", app_data)
        assert system_status.installed
        assert system_status.detail == "Ready in system cache"
        assert system_status.path == str(system_model)
    if old_hf_home is None:
        os.environ.pop("HF_HOME", None)
    else:
        os.environ["HF_HOME"] = old_hf_home

    assert ollama_status("llama3.1:latest", ["llama3.1:latest"]).installed
    assert ollama_status("llama3.2:3b", ["llama3.2:3b"]).installed
    assert not ollama_status("mistral:7b", ["llama3.1:latest"]).installed
    assert blackhole_status(["BlackHole 2ch"]).installed
    print("model asset tests passed")


if __name__ == "__main__":
    main()
