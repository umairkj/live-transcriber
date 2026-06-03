#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x venv/bin/python ]]; then
  echo "Missing virtual environment. Run ./setup.sh first." >&2
  exit 1
fi

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
    /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile app_icon" "${plist}" 2>/dev/null \
      || /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string app_icon" "${plist}"
    touch "${app_bundle}"
  fi
fi
