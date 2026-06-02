# live-transcriber

A local macOS terminal prototype that records short chunks from an audio input device, transcribes them with `faster-whisper`, and saves the transcript continuously to text and JSONL files.

This is an early prototype for a future live translation and vocabulary-learning desktop app. For now, it only does live transcription and file saving. Translation and vocabulary extraction are intentionally left as local placeholders.

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
venv/bin/python app.py --model tiny --chunk-seconds 5
```

With a German language hint:

```bash
venv/bin/python app.py --model base --language de --chunk-seconds 6 --output transcripts/german.txt
```

The terminal prints recognized speech directly. Diagnostic logs are hidden by default; add `--verbose` when debugging setup or model issues.

Use a specific input device:

```bash
venv/bin/python app.py --device 3 --model base --language de
```

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
venv/bin/python app.py --model tiny --chunk-seconds 5
```

You can also increase `--chunk-seconds` to reduce how often transcription runs.

### Homebrew Python 3.14 venv or ensurepip error

If Homebrew's latest `python3` fails while creating `venv`, install Python 3.13 and recreate the virtual environment:

```bash
brew install python@3.13
rm -rf venv
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv venv
source venv/bin/activate
venv/bin/python -m pip install -r requirements.txt
```
