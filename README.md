# live-transcriber

A local macOS terminal prototype that captures audio continuously, transcribes pause-bounded utterances with `faster-whisper`, and saves the transcript continuously to text and JSONL files.

This is an early prototype for a future live translation and vocabulary-learning desktop app. For now, it only does live transcription and file saving. Translation and vocabulary extraction are intentionally left as local placeholders.

## Why not Docker for macOS audio capture?

Docker is not recommended for the audio listener on macOS because containers do not get simple, reliable access to CoreAudio input devices. A normal Python virtual environment can talk directly to microphone input and virtual devices like BlackHole 2ch through `sounddevice`.

## Recommended installation

Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script can install system dependencies, create a virtual environment, install Python packages, optionally install BlackHole 2ch, guide the macOS Multi-Output Device setup, optionally pre-download a Whisper model, and help you list or test audio devices.

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

You can also run the command menu:

```bash
./t
```

Choose **Start transcription**, or run it directly:

```bash
./t start
```

The launcher automatically uses `BlackHole 2ch` when it is visible. If BlackHole is not visible, it starts with the system default input device.

## No-gap live transcript

The default mode is a no-gap live transcript. The old blocking mode could miss audio because recording stopped while Whisper was transcribing. The current mode keeps a `sounddevice.InputStream` running continuously, stores audio in a time-indexed rolling buffer, and refreshes the current transcript line from short rolling windows.

The terminal shows a transcript-style line immediately, for example `[20:38:32] Wie wichtig ist Geld`. While the speaker keeps talking, that same line is corrected in place as more words arrive. After a pause, the line is committed and the next spoken segment starts on a new line.

In the background, final transcript text still comes from pause-bounded utterance audio, so the saved transcript gets the start, end, and a little padding around each phrase before text is committed.

Best starting command for fast German live transcript:

```bash
venv/bin/python app.py --device 0 --model base --language de
```

For even lower latency:

```bash
venv/bin/python app.py --device 0 --model tiny --language de --window-seconds 2.5 --step-seconds 0.35
```

For slower but steadier text:

```bash
venv/bin/python app.py --device 0 --model small --language de --window-seconds 4 --step-seconds 1
```

Vocabulary mode:

```bash
venv/bin/python app.py --device 0 --model base --language de --vocab-mode
```

Diagnostic logs are hidden by default; add `--verbose` when debugging setup or model issues.

`--no-live-transcript` disables live line updates and prints only pause-finalized transcript lines.

`--window-seconds` controls how much current utterance audio Whisper sees for the live line. Smaller windows feel more immediate; larger windows give more context.

`--step-seconds` controls how often the live line refreshes.

`--pause-finalize-seconds` controls how much silence ends an utterance. Smaller values feel more realtime; larger values give Whisper more confidence that the phrase is complete.

`--activity-poll-seconds` controls how often the app scans captured audio for speech and silence.

`--activity-frame-seconds` controls the RMS frame size used by the speech detector.

`--buffer-seconds` controls how much recent audio is kept in memory.

Pause-finalized utterances are not skipped based on their average RMS once speech has already been detected. `--no-silence-skip` mainly affects optional live rolling-window hypotheses.

Recommended settings:

```bash
venv/bin/python app.py --model tiny --language de --pause-finalize-seconds 0.7
venv/bin/python app.py --model base --language de
venv/bin/python app.py --model small --language de --window-seconds 4 --step-seconds 1
```

## Improving Rolling-Window Quality

Rolling windows intentionally overlap. Whisper repeats context, so the raw output is a hypothesis, not a final transcript. The stabilizer waits for candidate sentences to appear consistently, rejects broken fragments, and suppresses recent near-duplicates.

The quality layer includes:

- a hypothesis stability tracker
- a sentence quality filter for fragments like `Uhr.`, `uf.`, and one-word non-allowlisted text
- a recent sentence cache for exact and fuzzy duplicates

Larger windows and slower steps produce cleaner output with a little more delay.

## Why Rolling-Window Output Repeats

The app transcribes overlapping rolling windows. Whisper repeats context from each window, so raw window text often contains phrases that were already heard. Raw window text must not be treated as final transcript.

The stabilizer pipeline is:

```text
raw Whisper window -> clean text -> stability tracker -> duplicate filter -> quality filter -> transcript output
```

Default output only prints stable final sentences. To inspect raw Whisper hypotheses:

```bash
venv/bin/python app.py --device 0 --model small --language de --window-seconds 8 --step-seconds 2 --debug-raw-windows
```

To inspect stabilizer decisions:

```bash
venv/bin/python app.py --device 0 --model small --language de --window-seconds 8 --step-seconds 2 --debug-stabilizer
```

## Capturing system audio with BlackHole 2ch

The recommended path is:

```bash
./setup.sh
```

Choose the BlackHole/system-audio setup option. The script installs BlackHole 2ch with Homebrew, opens Audio MIDI Setup, opens Sound settings, and then lists devices so you can find the BlackHole input ID.

You can also install BlackHole manually:

```bash
brew install --cask blackhole-2ch
```

Basic macOS setup, guided by `./setup.sh`:

1. Open Audio MIDI Setup.
2. Create a Multi-Output Device.
3. Add your built-in output or headphones and BlackHole 2ch.
4. Enable Drift Correction for BlackHole 2ch.
5. Control-click the Multi-Output Device and choose **Use This Device For Sound Output**.
6. Do not set macOS Output directly to BlackHole 2ch, or your speakers will go silent.
7. Run `venv/bin/python app.py --list-devices`.
8. Use the BlackHole input device ID with `venv/bin/python app.py --device <ID>`.

If the Multi-Output Device does not appear in Sound settings, set it from Audio MIDI Setup with **Use This Device For Sound Output**. The Multi-Output Device is an output device, so it will not appear in this app's input-device list; the app should use `BlackHole 2ch` as its input.

macOS does not provide a reliable supported shell command for creating a Multi-Output Device, so `./setup.sh` opens the right apps and guides that step.

Example:

```bash
venv/bin/python app.py --device 3 --model small --language de --window-seconds 8 --step-seconds 2
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

If BlackHole is installed but does not show up in `--list-devices`, restart CoreAudio or restart your Mac. The setup script can guide this:

```bash
./setup.sh
```

Choose the BlackHole/system-audio setup option, then accept the CoreAudio restart if BlackHole still is not visible.

Manual CoreAudio restart:

```bash
sudo killall coreaudiod
```

Then run:

```bash
venv/bin/python app.py --list-devices
```

### Model download takes time

The first run can take a while because `faster-whisper` downloads the selected model. Later runs reuse the cached model.

### Old Intel MacBook is slow

Try the smallest model first:

```bash
venv/bin/python app.py --model tiny --window-seconds 5 --step-seconds 2
```

You can also increase `--step-seconds` to reduce how often transcription runs.

### Homebrew Python 3.14 venv or ensurepip error

If Homebrew's latest `python3` fails while creating `venv`, install Python 3.13 and recreate the virtual environment:

```bash
brew install python@3.13
rm -rf venv
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv venv
source venv/bin/activate
venv/bin/python -m pip install -r requirements.txt
```
