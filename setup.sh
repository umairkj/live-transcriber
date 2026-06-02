#!/usr/bin/env bash
set -e

info() {
  printf '\n%s\n' "$1"
}

print_banner() {
  cat <<'BANNER'

 _      _____     _______   _____ ____      _    _   _ ____   ____ ____  ___ ____  _____ ____
| |    |_ _\ \   / / ____| |_   _|  _ \    / \  | \ | / ___| / ___|  _ \|_ _| __ )| ____|  _ \
| |     | | \ \ / /|  _|     | | | |_) |  / _ \ |  \| \___ \| |   | |_) || ||  _ \|  _| | |_) |
| |___  | |  \ V / | |___    | | |  _ <  / ___ \| |\  |___) | |___|  _ < | || |_) | |___|  _ <
|_____|____|  \_/  |_____|   |_| |_| \_\/_/   \_\_| \_|____/ \____|_| \_\___|____/|_____|_| \_\

BANNER
}

warn() {
  printf '\nWarning: %s\n' "$1"
}

confirm() {
  local prompt="$1"
  local default="${2:-N}"
  local answer

  read -r -p "$prompt " answer
  if [ -z "$answer" ]; then
    answer="$default"
  fi

  case "$answer" in
    y|Y|yes|YES|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

have_brew() {
  command -v brew >/dev/null 2>&1
}

try_load_brew_shellenv() {
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -x /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

python_version_ok() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

python_can_create_venv() {
  local python_bin="$1"
  local temp_dir

  temp_dir="$(mktemp -d)"
  if "$python_bin" -m venv "$temp_dir/venv-test" >/dev/null 2>&1 && \
    "$temp_dir/venv-test/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$temp_dir"
    return 0
  fi

  rm -rf "$temp_dir"
  return 1
}

find_python_bin() {
  local candidate
  local candidates=()
  local brew_prefix

  if have_brew; then
    brew_prefix="$(brew --prefix 2>/dev/null || true)"
    if [ -n "$brew_prefix" ]; then
      candidates+=("$brew_prefix/opt/python@3.13/bin/python3.13")
      candidates+=("$brew_prefix/opt/python@3.12/bin/python3.12")
      candidates+=("$brew_prefix/opt/python@3.11/bin/python3.11")
      candidates+=("$brew_prefix/bin/python3")
    fi
  fi

  candidate="$(command -v python3 2>/dev/null || true)"
  if [ -n "$candidate" ]; then
    candidates+=("$candidate")
  fi

  for candidate in "${candidates[@]}"; do
    if [ -x "$candidate" ] && python_version_ok "$candidate" && python_can_create_venv "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

venv_healthy() {
  [ -f venv/bin/activate ] && \
    [ -x venv/bin/python ] && \
    python_version_ok venv/bin/python && \
    venv/bin/python -m pip --version >/dev/null 2>&1
}

blackhole_installed() {
  if have_brew && brew list --cask blackhole-2ch >/dev/null 2>&1; then
    return 0
  fi

  [ -d /Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver ] || \
    [ -d "$HOME/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver" ]
}

print_blackhole_setup_help() {
  cat <<'BLACKHOLE_HELP'

System audio capture setup:
1. In Audio MIDI Setup, click the + button in the lower-left corner.
2. Choose Create Multi-Output Device.
3. Check your speakers/headphones and BlackHole 2ch.
4. Optional but recommended: enable Drift Correction for BlackHole 2ch.
5. Control-click the Multi-Output Device and choose Use This Device For Sound Output.
6. Do not set macOS Output directly to BlackHole 2ch, or your speakers will go silent.
7. In this app, use BlackHole 2ch as the input device.

macOS does not provide a reliable supported shell command for creating a
Multi-Output Device, so setup opens the right app and guides that step.
BLACKHOLE_HELP
}

open_audio_midi_setup() {
  if [ "$(uname -s)" = "Darwin" ]; then
    open -a "Audio MIDI Setup" >/dev/null 2>&1 || true
  fi
}

open_sound_settings() {
  if [ "$(uname -s)" = "Darwin" ]; then
    open "x-apple.systempreferences:com.apple.Sound-Settings.extension" >/dev/null 2>&1 || \
      open -a "System Settings" >/dev/null 2>&1 || true
  fi
}

guide_blackhole_system_audio_setup() {
  print_blackhole_setup_help

  if confirm "Open Audio MIDI Setup now? [Y/n]" "Y"; then
    open_audio_midi_setup
  fi

  printf '\nCreate the Multi-Output Device now, then come back here.\n'
  read -r -p "Press Enter when the Multi-Output Device is ready..."

  printf '\nIn Audio MIDI Setup, control-click the Multi-Output Device and choose "Use This Device For Sound Output".\n'
  printf 'Do not choose BlackHole 2ch as macOS Output directly; it should only be checked inside the Multi-Output Device.\n'
  read -r -p "Press Enter when the Multi-Output Device is set as sound output..."

  if confirm "Open macOS Sound settings to double-check Output? [y/N]" "N"; then
    open_sound_settings
  fi
}

blackhole_visible_to_app() {
  [ -x venv/bin/python ] && \
    venv/bin/python app.py --list-devices 2>/dev/null | grep -qi 'blackhole'
}

restart_coreaudio() {
  if [ "$(uname -s)" != "Darwin" ]; then
    warn "CoreAudio restart is only available on macOS."
    return 1
  fi

  printf '\nRestarting CoreAudio. macOS may ask for your password.\n'
  sudo killall coreaudiod
  sleep 3
}

recommended_model() {
  printf 'small'
}

choose_model() {
  local choice
  local default_model

  default_model="$(recommended_model)"
  printf '\nChoose a Whisper model:\n' >&2
  printf '1) tiny - fastest, lower accuracy\n' >&2
  printf '2) base - good low-latency choice\n' >&2
  printf '3) small - recommended for Apple Silicon MacBook Air\n' >&2
  printf '4) medium - better accuracy, much slower\n' >&2
  printf '5) large-v3 - likely too slow for this terminal prototype\n' >&2
  read -r -p "Model [$default_model]: " choice

  case "$choice" in
    1|tiny) printf 'tiny' ;;
    2|base) printf 'base' ;;
    '') printf '%s' "$default_model" ;;
    3|small) printf 'small' ;;
    4|medium) printf 'medium' ;;
    5|large-v3|large) printf 'large-v3' ;;
    *) printf '%s' "$choice" ;;
  esac
}

print_banner
info "Setup"

if [ "$(uname -s)" != "Darwin" ]; then
  warn "This prototype targets macOS. You can continue, but audio setup may differ."
  if ! confirm "Continue anyway? [y/N]" "N"; then
    exit 1
  fi
fi

mkdir -p transcripts
LIST_DEVICES_AFTER_SETUP="N"

if ! have_brew; then
  warn "Homebrew was not found."
  if confirm "Homebrew is required to install system audio dependencies. Install Homebrew now? [y/N]" "N"; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    try_load_brew_shellenv
  else
    warn "Continuing without Homebrew. portaudio/ffmpeg installation may fail."
  fi
fi

if have_brew; then
  try_load_brew_shellenv

  missing_deps=()
  if ! brew list portaudio >/dev/null 2>&1; then
    missing_deps+=("portaudio")
  fi
  if ! brew list ffmpeg >/dev/null 2>&1; then
    missing_deps+=("ffmpeg")
  fi

  if [ "${#missing_deps[@]}" -gt 0 ]; then
    printf '\nMissing system dependencies: %s\n' "${missing_deps[*]}"
    if confirm "Install missing system dependencies with Homebrew? [y/N]" "N"; then
      brew install "${missing_deps[@]}"
    else
      warn "Skipping system dependency installation."
    fi
  else
    info "System dependencies already installed."
  fi
else
  warn "Homebrew is unavailable. Skipping system dependency checks."
fi

if confirm "Do you want to set up BlackHole 2ch for capturing system audio from macOS apps? [y/N]" "N"; then
  if blackhole_installed; then
    info "BlackHole 2ch already appears to be installed."
  elif have_brew; then
    brew install --cask blackhole-2ch
  else
    warn "Homebrew is required to install BlackHole 2ch automatically."
  fi

  guide_blackhole_system_audio_setup
  LIST_DEVICES_AFTER_SETUP="Y"
fi

if ! PYTHON_BIN="$(find_python_bin)"; then
  warn "Could not find Python 3.11+ that can create a pip-ready virtual environment."
  if have_brew && confirm "Install Python 3.13 with Homebrew now? [y/N]" "N"; then
    brew install python@3.13
    try_load_brew_shellenv
    if ! PYTHON_BIN="$(find_python_bin)"; then
      printf 'Python 3.13 was installed, but setup still could not create a pip-ready venv.\n'
      exit 1
    fi
  else
    printf 'Please install Python 3.11+ and rerun this script. Suggested command: brew install python@3.13\n'
    exit 1
  fi
fi

printf '\nUsing Python: %s\n' "$PYTHON_BIN"
"$PYTHON_BIN" --version

if [ -d venv ]; then
  if venv_healthy; then
    if confirm "A Python virtual environment already exists. Reuse it? [Y/n]" "Y"; then
      info "Reusing existing venv."
    else
      if confirm "Recreate venv? This deletes ./venv. [y/N]" "N"; then
        rm -rf venv
        "$PYTHON_BIN" -m venv venv
      else
        printf 'Setup stopped. Existing venv was left unchanged.\n'
        exit 1
      fi
    fi
  else
    warn "Existing ./venv is incomplete, too old, or missing pip."
    if confirm "Recreate venv? This deletes ./venv. [y/N]" "N"; then
      rm -rf venv
      "$PYTHON_BIN" -m venv venv
    else
      printf 'Setup stopped. Existing venv was left unchanged.\n'
      exit 1
    fi
  fi
else
  "$PYTHON_BIN" -m venv venv
fi

source venv/bin/activate

info "Installing Python packages"
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

if confirm "Do you want to download a Whisper model now? [y/N]" "N"; then
  MODEL_NAME="$(choose_model)"
  MODEL_NAME="$MODEL_NAME" venv/bin/python - <<'PY'
import os
from faster_whisper import WhisperModel

model_name = os.environ["MODEL_NAME"]
WhisperModel(model_name, device="cpu", compute_type="int8")
print(f"Model downloaded/loaded: {model_name}")
PY
fi

if confirm "Do you want to list your audio input devices now? [Y/n]" "Y"; then
  if [ "$LIST_DEVICES_AFTER_SETUP" = "Y" ]; then
    info "Look for BlackHole 2ch in this device list."
  fi

  if ! venv/bin/python app.py --list-devices; then
    warn "Could not list audio input devices."
  fi

  if [ "$LIST_DEVICES_AFTER_SETUP" = "Y" ] && ! blackhole_visible_to_app; then
    warn "BlackHole 2ch is installed, but it is not visible to the app yet."
    if confirm "Restart CoreAudio now? This usually makes BlackHole appear. [Y/n]" "Y"; then
      restart_coreaudio || warn "Could not restart CoreAudio automatically."
      printf '\nAudio input devices after CoreAudio restart:\n'
      venv/bin/python app.py --list-devices || warn "Could not list audio input devices."
    fi
    if ! blackhole_visible_to_app; then
      warn "BlackHole 2ch still is not visible. Restart your Mac, then run: venv/bin/python app.py --list-devices"
    fi
  fi
fi

if confirm "Do you want to start live transcription now? [y/N]" "N"; then
  ./t start
fi

cat <<'NEXT_STEPS'

Setup complete.

Useful next commands:

source venv/bin/activate
venv/bin/python app.py --list-devices
./t start
NEXT_STEPS
