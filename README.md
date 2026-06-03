# live-transcriber

A local macOS terminal prototype that continuously captures audio from an input device, detects speech segments, transcribes completed speech with `faster-whisper`, and saves the transcript to text and JSONL files.

This is an early prototype for a future live translation and vocabulary-learning desktop app. It currently does live transcription, finalized-line translation, selective word hints, speaker labels, and file saving.

## Why not Docker for macOS audio capture?

Docker is not recommended for the audio listener on macOS because containers do not get simple, reliable access to CoreAudio input devices. A normal Python virtual environment can talk directly to microphone input and virtual devices like BlackHole 2ch through `sounddevice`.

## Recommended installation

Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script can install system dependencies, create a virtual environment, install Python packages, optionally install BlackHole 2ch, optionally pre-download a Whisper model, and help you list or test audio devices.

## Manual installation

Advanced users can install dependencies manually:

```bash
brew install portaudio ffmpeg
brew install python@3.13

/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv venv
source venv/bin/activate
venv/bin/python -m pip install -r requirements.txt
```

## List audio input devices

```bash
venv/bin/python app.py --list-devices
```

Each input device is printed with its numeric device ID, name, input channel count, and default sample rate.

## Run with microphone input

Use the default macOS input device:

```bash
venv/bin/python app.py --model base
```

With a German language hint:

```bash
venv/bin/python app.py --model base --language de --output transcripts/german.txt
```

The terminal prints fast provisional lines during active speech and clean final lines after pauses. Provisional lines start with `[partial]` and are not saved. Final lines are saved to the transcript files.

Use a specific input device:

```bash
venv/bin/python app.py --device 3 --model base --language de
```

The recorder uses `sounddevice.InputStream`, so audio capture keeps running while Whisper transcribes previous speech. The app sends active speech snapshots to Whisper every 2 seconds for fast provisional output, then sends the completed speech segment after a short pause for the saved final transcript.

Disable provisional output with:

```bash
venv/bin/python app.py --model base --language de --partial-seconds 0
```

Translate finalized lines to English with Whisper:

```bash
venv/bin/python app.py --model base --language de --full-translation
```

Add selective English word hints for useful German nouns and verbs:

```bash
venv/bin/python app.py --model base --language de --selective-translation
```

Full translation applies only to finalized transcript lines. Selective noun/verb hints can also appear on live partial lines because they use the lightweight built-in glossary.

Enable anonymous speaker labels on finalized lines with:

```bash
venv/bin/python app.py --model base --language de --speaker-labels
```

Speaker labels are optional and best-effort. The local backend uses SpeechBrain ECAPA speaker embeddings if the optional speaker packages are installed:

```bash
venv/bin/python -m pip install speechbrain torch torchaudio
```

When enabled, final terminal and text transcript lines are prefixed with labels like `A:` or `B:`. Partial lines remain unlabeled, and JSONL records keep raw text plus `speaker_label`, `speaker_confidence`, and `speaker_backend` fields. The `pyannote` backend is reserved for a later advanced implementation.

## Run the macOS desktop UI

Install the Python requirements, then launch the PySide6 app:

```bash
venv/bin/python ui_app.py
```

The UI uses the same transcription engine as the CLI. It defaults to BlackHole 2ch when that input is visible, `base` model, German language, saved transcripts, 2-second provisional partials, full English translation, and selective word hints.

Selective word hints appear directly above the German nouns and verbs inside finalized and live partial transcript lines. The right-side vocabulary panel accumulates new words from finalized lines with English meanings, so provisional mis-hearings do not become study words.

Speaker labels can be enabled from the UI. Use the reset button if the app learns poor speaker profiles during the current listening session.

## Capturing system audio with BlackHole 2ch

BlackHole needs to be installed separately:

```bash
brew install --cask blackhole-2ch
```

Basic macOS setup:

1. Open Audio MIDI Setup.
2. Create a Multi-Output Device.
3. Add your built-in output or headphones and BlackHole 2ch.
4. Set macOS system output to the Multi-Output Device.
5. Run `venv/bin/python app.py --list-devices`.
6. Use the BlackHole input device ID with `venv/bin/python app.py --device <ID>`.

Example:

```bash
venv/bin/python app.py --device 3 --model base --language de
```

For a good Apple Silicon MacBook Air balance, use:

```bash
venv/bin/python app.py --device <BLACKHOLE_ID> --model small --language de --beam-size 1
```

For lower latency, use `base` and keep the default 2-second provisional output:

```bash
venv/bin/python app.py --device <BLACKHOLE_ID> --model base --language de --beam-size 1
```

## Recommended models

`faster-whisper` downloads the selected model on first use if needed.

- `tiny`: fastest, lower quality
- `base`: good low-latency first choice
- `small`: recommended balance for Apple Silicon MacBook Air
- `medium` and `large-v3`: likely too slow on an old Intel MacBook

## Output files

By default, transcripts are appended to:

```text
transcripts/transcript.txt
transcripts/transcript.jsonl
```

Clear output files at startup with:

```bash
venv/bin/python app.py --overwrite
```

Disable JSONL output with:

```bash
venv/bin/python app.py --no-jsonl
```

## Troubleshooting

### No input devices found

Check macOS Sound settings and confirm that your microphone or virtual input device is visible. Then run:

```bash
venv/bin/python app.py --list-devices
```

### macOS microphone permission problem

If recording fails, open System Settings, then Privacy & Security, then Microphone. Make sure your terminal app has microphone access.

### BlackHole does not show up

Restart the app you are using for audio, or restart your Mac after installing BlackHole. Then run:

```bash
venv/bin/python app.py --list-devices
```

### Model download takes time

The first run can take a while because `faster-whisper` downloads the selected model. Later runs reuse the cached model.

### Old Intel MacBook is slow

Try the smallest model first:

```bash
venv/bin/python app.py --model tiny
```

Use `--model tiny` or `--model base` for lower latency.

### Homebrew Python 3.14 venv or ensurepip error

If Homebrew's latest `python3` fails while creating `venv`, install Python 3.13 and recreate the virtual environment:

```bash
brew install python@3.13
rm -rf venv
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv venv
source venv/bin/activate
venv/bin/python -m pip install -r requirements.txt
```
