from __future__ import annotations

from typing import Any


def _get_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError("sounddevice is not installed. Run ./setup.sh or pip install -r requirements.txt.") from exc

    return sd


def _get_input_devices() -> list[tuple[int, dict[str, Any]]]:
    sd = _get_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:  # sounddevice can raise PortAudioError or OSError here.
        raise RuntimeError(f"Could not query audio devices: {exc}") from exc

    input_devices: list[tuple[int, dict[str, Any]]] = []
    for device_id, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels > 0:
            input_devices.append((device_id, dict(device)))

    return input_devices


def list_input_devices() -> None:
    """Print available sounddevice input devices."""
    input_devices = _get_input_devices()
    if not input_devices:
        raise RuntimeError("No audio input devices found.")

    print("Input devices:")
    for device_id, device in input_devices:
        name = device.get("name", "Unknown device")
        channels = int(device.get("max_input_channels", 0))
        default_sample_rate = device.get("default_samplerate")
        sample_rate_text = "unknown"
        if isinstance(default_sample_rate, (int, float)):
            sample_rate_text = f"{default_sample_rate:.0f} Hz"

        print(
            f"{device_id}: {name} | "
            f"input channels: {channels} | "
            f"default sample rate: {sample_rate_text}"
        )


def ensure_input_devices_available() -> None:
    """Raise a clear error when no audio input devices are available."""
    if not _get_input_devices():
        raise RuntimeError("No audio input devices found.")


def validate_input_device(device_id: int | None) -> None:
    """Validate that a device ID exists and can be used for input."""
    if device_id is None:
        return

    sd = _get_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Could not query audio devices: {exc}") from exc

    if device_id < 0 or device_id >= len(devices):
        raise ValueError(f"Invalid input device ID {device_id}. Run with --list-devices to see valid IDs.")

    device = devices[device_id]
    max_input_channels = int(device.get("max_input_channels", 0))
    if max_input_channels <= 0:
        name = device.get("name", "Unknown device")
        raise ValueError(f"Device {device_id} ({name}) is not an input device.")
