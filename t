#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PYTHON_BIN="venv/bin/python"

print_header() {
  cat <<'HEADER'

 _      _____     _______   _____ ____      _    _   _ ____   ____ ____  ___ ____  _____ ____
| |    |_ _\ \   / / ____| |_   _|  _ \    / \  | \ | / ___| / ___|  _ \|_ _| __ )| ____|  _ \
| |     | | \ \ / /|  _|     | | | |_) |  / _ \ |  \| \___ \| |   | |_) || ||  _ \|  _| | |_) |
| |___  | |  \ V / | |___    | | |  _ <  / ___ \| |\  |___) | |___|  _ < | || |_) | |___|  _ <
|_____|____|  \_/  |_____|   |_| |_| \_\/_/   \_\_| \_|____/ \____|_| \_\___|____/|_____|_| \_\

HEADER
}

run_command() {
  printf '\nRunning:'
  printf ' %q' "$@"
  printf '\n\n'
  "$@"
}

require_venv() {
  if [ ! -x "$PYTHON_BIN" ]; then
    printf 'Virtual environment not found. Run ./setup.sh first.\n'
    exit 1
  fi
}

list_commands() {
  print_header
  cat <<'MENU'
1) List audio input devices
2) Start transcription
3) Run setup
4) Show help
q) Quit
MENU
}

detect_blackhole_device_id() {
  "$PYTHON_BIN" app.py --list-devices 2>/dev/null | awk -F: '
    tolower($0) ~ /blackhole/ {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      print $1
      exit
    }
  '
}

start_transcription() {
  require_venv
  local device_id
  local -a cmd

  cmd=("$PYTHON_BIN" app.py --model base --language de)
  device_id="$(detect_blackhole_device_id)"
  if [ -n "$device_id" ]; then
    cmd+=(--device "$device_id")
    printf 'Using BlackHole 2ch input device: %s\n' "$device_id"
  else
    printf 'BlackHole 2ch input was not found. Starting with the default input device.\n'
  fi

  run_command "${cmd[@]}"
}

case "${1:-}" in
  devices|list-devices)
    require_venv
    run_command "$PYTHON_BIN" app.py --list-devices
    ;;
  de|german)
    start_transcription
    ;;
  start|run)
    start_transcription
    ;;
  setup)
    run_command ./setup.sh
    ;;
  help|-h|--help)
    require_venv
    run_command "$PYTHON_BIN" app.py --help
    ;;
  "")
    while true; do
      list_commands
      printf '\n'
      read -r -p 'Choose command: ' choice
      case "$choice" in
        1) require_venv; run_command "$PYTHON_BIN" app.py --list-devices ;;
        2) start_transcription ;;
        3) run_command ./setup.sh ;;
        4) require_venv; run_command "$PYTHON_BIN" app.py --help ;;
        q|Q|quit|exit) exit 0 ;;
        *) printf 'Unknown choice: %s\n' "$choice" ;;
      esac
      printf '\n'
      read -r -p 'Press Enter to return to menu...' _
    done
    ;;
  *)
    printf 'Unknown command: %s\n' "$1"
    printf 'Run ./t to open the command menu.\n'
    exit 1
    ;;
esac
