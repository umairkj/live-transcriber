# macOS Packaging Notes

The desktop app is built with PySide6, so the first packaging target is a local macOS `.app` produced by `pyside6-deploy`.

The app icon source is `assets/app_icon.png`; the macOS bundle icon is `assets/app_icon.icns`. The build helper stamps the `.icns` into the generated `.app` after `pyside6-deploy` finishes.

## Build

From the project root:

```bash
./scripts/build_macos_app.sh
```

To create or refresh the PySide deploy spec first:

```bash
./scripts/build_macos_app.sh --init
```

## Runtime Assets

Large or user-specific assets are intentionally not bundled into the app:

- faster-whisper models
- Ollama models
- German-English dictionary database
- BlackHole 2ch

The UI Settings tab installs or pulls those assets into the user's machine after launch. App-managed data is stored under the platform app-data directory, which is `~/Library/Application Support/Live Transcriber` on macOS.

## Shipping Checklist

- Build the `.app` with `pyside6-deploy`.
- Test first launch on a clean macOS user account.
- Confirm Settings can download the selected Whisper model.
- Confirm Settings detects or explains Ollama, BlackHole 2ch, spaCy, and the dictionary.
- Sign and notarize the `.app` or DMG for distribution outside local development.
