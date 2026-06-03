from __future__ import annotations

import json
from typing import Any, Callable
from urllib import error, request


DEFAULT_MINUTES_MODEL = "llama3.1:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
MINUTES_UPDATE_FINAL_COUNT = 3


class MinutesError(RuntimeError):
    """Raised when local minutes generation cannot produce usable output."""


Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class OllamaMinutesClient:
    """Small stdlib-only client for generating meeting minutes with Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MINUTES_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 90.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport or self._post_json

    def summarize(self, existing_minutes: str, events: list[dict[str, Any]]) -> str:
        if not events:
            return existing_minutes

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(existing_minutes, events)},
            ],
            "options": {"temperature": 0.1},
        }
        response = self._transport(f"{self.base_url}/api/chat", payload, self.timeout)
        content = response.get("message", {}).get("content") if isinstance(response, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise MinutesError("Ollama returned an empty minutes response.")
        return content.strip()

    def _system_prompt(self) -> str:
        return (
            "You update concise English meeting minutes from finalized live transcript lines. "
            "Do not write a top-level title. Keep these exact section names in this order: "
            "Decisions, Action Items, Open Questions, Important Points. "
            "Write plain text section names with no Markdown, no bold markers, no hashtags, and no trailing colons. "
            "Use '-' bullets only. Include timestamps or timestamp ranges when available. "
            "Preserve still-relevant existing minutes, merge duplicates, and do not invent facts."
        )

    def _user_prompt(self, existing_minutes: str, events: list[dict[str, Any]]) -> str:
        return (
            "Existing minutes:\n"
            f"{existing_minutes.strip() or '(none yet)'}\n\n"
            "Finalized transcript events:\n"
            f"{format_minutes_events(events)}\n\n"
            "Return the complete updated minutes now."
        )

    def _post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise MinutesError(_ollama_error_message(body, self.model)) from exc
        except error.URLError as exc:
            raise MinutesError("Ollama is not running. Start it with: ollama serve") from exc
        except TimeoutError as exc:
            raise MinutesError("Ollama minutes update timed out.") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MinutesError("Ollama returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise MinutesError("Ollama returned an unexpected response.")
        if isinstance(parsed.get("error"), str):
            raise MinutesError(_ollama_error_message(str(parsed["error"]), self.model))
        return parsed


def format_minutes_events(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        record = event.get("record") if isinstance(event.get("record"), dict) else {}
        text = str(record.get("text") or event.get("display_text") or "").strip()
        if not text:
            continue

        timestamp = str(record.get("timestamp") or "").strip()
        speaker = str(record.get("speaker_label") or "").strip()
        translation = str(record.get("translation_text") or "").strip()

        prefix_parts = [f"#{index}"]
        if timestamp:
            prefix_parts.append(timestamp)
        if speaker:
            prefix_parts.append(f"Speaker {speaker}")

        line = f"- [{' | '.join(prefix_parts)}] {text}"
        if translation:
            line += f"\n  English translation: {translation}"
        lines.append(line)

    return "\n".join(lines) if lines else "(no usable finalized transcript text)"


def _ollama_error_message(message: str, model: str) -> str:
    normalized = message.casefold()
    if "not found" in normalized or "pull" in normalized or ("model" in normalized and "missing" in normalized):
        model_name = model.removesuffix(":latest")
        return f"Model {model} is missing. Install it with: ollama pull {model_name}"
    return message.strip() or "Ollama minutes update failed."
