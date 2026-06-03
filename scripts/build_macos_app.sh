#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x venv/bin/python ]]; then
  echo "Missing virtual environment. Run ./setup.sh first." >&2
  exit 1
fi

sign_app_bundle() {
  local app_bundle="$1"
  local identity="${CODESIGN_IDENTITY:--}"

  if [[ ! -d "${app_bundle}" ]]; then
    echo "App bundle not found: ${app_bundle}" >&2
    exit 1
  fi

  xattr -dr com.apple.quarantine "${app_bundle}" 2>/dev/null || true

  codesign --force --deep --sign "${identity}" "${app_bundle}"
  codesign --verify --deep --strict --verbose=4 "${app_bundle}"
}

set_plist_value() {
  local plist="$1"
  local key="$2"
  local type="$3"
  local value="$4"

  /usr/libexec/PlistBuddy -c "Set :${key} ${value}" "${plist}" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Add :${key} ${type} ${value}" "${plist}"
}

venv/bin/python -m pip install -r requirements.txt

if [[ "${1:-}" == "--init" ]]; then
  venv/bin/pyside6-deploy \
    --init \
    --name "Live Transcriber" \
    --extra-ignore-dirs=venv,transcripts,__pycache__,data,pretrained_models,.git \
    ui_app.py
  exit 0
fi

venv/bin/pyside6-deploy \
  --force \
  --name "Live Transcriber" \
  --extra-ignore-dirs=venv,transcripts,__pycache__,data,pretrained_models,.git \
  ui_app.py

app_bundle="$(find . -name 'Live Transcriber.app' -type d -print -quit)"
if [[ -n "${app_bundle}" && -f assets/app_icon.icns ]]; then
  mkdir -p "${app_bundle}/Contents/Resources"
  cp assets/app_icon.icns "${app_bundle}/Contents/Resources/app_icon.icns"
  plist="${app_bundle}/Contents/Info.plist"
  if [[ -f "${plist}" ]]; then
    set_plist_value "${plist}" "CFBundleIconFile" "string" "app_icon"
    set_plist_value "${plist}" "CFBundleName" "string" "Live Transcriber"
    set_plist_value "${plist}" "CFBundleDisplayName" "string" "Live Transcriber"
    set_plist_value "${plist}" "CFBundleIdentifier" "string" "${BUNDLE_IDENTIFIER:-com.umairkj.LiveTranscriber}"
    set_plist_value "${plist}" "NSMicrophoneUsageDescription" "string" \
      "Live Transcriber needs microphone access to capture audio for local transcription."
    touch "${app_bundle}"
  fi
fi

if [[ -z "${app_bundle}" ]]; then
  echo "Could not find generated Live Transcriber.app" >&2
  exit 1
fi

sign_app_bundle "${app_bundle}"
