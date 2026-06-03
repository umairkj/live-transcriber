from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.minutes import MinutesError, OllamaMinutesClient, format_minutes_events


def test_minutes_client_uses_fake_ollama_response() -> None:
    captured: dict[str, object] = {}

    def fake_transport(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"message": {"content": "Decisions\n- [09:00] Keep the plan."}}

    client = OllamaMinutesClient(transport=fake_transport, timeout=3.0)
    result = client.summarize(
        "",
        [
            {
                "kind": "final",
                "record": {
                    "timestamp": "2026-06-03 09:00:00",
                    "speaker_label": "A",
                    "text": "Wir behalten den Plan.",
                    "translation_text": "We keep the plan.",
                },
            }
        ],
    )

    assert "Keep the plan" in result
    assert captured["url"] == "http://localhost:11434/api/chat"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama3.1:latest"
    assert payload["stream"] is False


def test_minutes_client_rejects_empty_response() -> None:
    client = OllamaMinutesClient(transport=lambda _url, _payload, _timeout: {"message": {"content": ""}})
    try:
        client.summarize("", [{"kind": "final", "record": {"text": "Hallo"}}])
    except MinutesError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty minutes response should fail")


def test_format_minutes_events_includes_context() -> None:
    formatted = format_minutes_events(
        [
            {
                "kind": "final",
                "record": {
                    "timestamp": "2026-06-03 09:00:00",
                    "speaker_label": "B",
                    "text": "Ich putze die Zaehne.",
                    "translation_text": "I brush my teeth.",
                },
            }
        ]
    )

    assert "2026-06-03 09:00:00" in formatted
    assert "Speaker B" in formatted
    assert "Ich putze die Zaehne." in formatted
    assert "I brush my teeth." in formatted


def main() -> None:
    test_minutes_client_uses_fake_ollama_response()
    test_minutes_client_rejects_empty_response()
    test_format_minutes_events_includes_context()
    print("minutes client tests passed")


if __name__ == "__main__":
    main()
