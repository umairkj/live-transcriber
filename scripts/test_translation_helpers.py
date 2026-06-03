from __future__ import annotations

import sys
import sqlite3
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.translations import (
    DictionaryWordHintClient,
    OllamaWordHintClient,
    TranslationError,
    build_selective_translation_result,
    build_selective_translations,
)


def test_selective_translations_extract_useful_words() -> None:
    hints = build_selective_translations("Ich putze mir die Zaehne und trinke Wasser.")
    pairs = {(hint["source"], hint["translation"]) for hint in hints}
    assert ("putze", "clean") in pairs
    assert ("trinke", "drink") in pairs
    assert ("Wasser", "water") in pairs
    assert any(hint["source"] == "Wasser" and hint["article"] == "das" for hint in hints)


def test_selective_translations_are_english_only_for_now() -> None:
    assert build_selective_translations("Ich trinke Wasser.", target_language="fr") == []


def test_ollama_word_hints_extract_many_nouns_and_verbs() -> None:
    def fake_transport(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        del url, payload, timeout
        return {
            "message": {
                "content": (
                    '{"items":['
                    '{"source":"sitzt","translation":"sits","kind":"verb"},'
                    '{"source":"Opfer","translation":"victim","kind":"noun","article":"das"},'
                    '{"source":"möchte","translation":"wants","kind":"verb"},'
                    '{"source":"gefragt","translation":"asked","kind":"verb"},'
                    '{"source":"Panzer","translation":"armor","kind":"noun","article":"der"}'
                    "]}"
                )
            }
        }

    client = OllamaWordHintClient(transport=fake_transport)
    hints, backend = build_selective_translation_result(
        "Und ich war und sitzt ein Opfer, das kein Opfer mehr sein möchte. "
        "Und ich hab mich gefragt, warum hat sie sich eigentlich keinen Panzer angelegt?",
        backend="ollama",
        client=client,
    )
    pairs = {(hint["source"], hint["translation"]) for hint in hints}
    assert backend == "ollama"
    assert ("sitzt", "sits") in pairs
    assert ("Opfer", "victim") in pairs
    assert ("möchte", "wants") in pairs
    assert ("Panzer", "armor") in pairs
    assert any(hint["source"] == "Panzer" and hint["article"] == "der" for hint in hints)


def test_ollama_word_hints_fall_back_to_builtin_glossary() -> None:
    class FailingClient:
        def build_hints(self, text: str, max_items: int = 80) -> list[dict[str, str]]:
            del text, max_items
            raise TranslationError("Ollama unavailable")

    hints, backend = build_selective_translation_result(
        "Ich putze die Zaehne.",
        backend="ollama",
        client=FailingClient(),  # type: ignore[arg-type]
    )
    assert backend in {"dictionary", "built_in_glossary"}
    assert any(hint["source"] == "putze" for hint in hints)


def test_dictionary_backend_uses_local_sqlite_lookup() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "de_en.sqlite"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE entries ("
                "key TEXT, source TEXT, lemma_key TEXT, lemma TEXT, translation TEXT, "
                "kind TEXT, article TEXT, priority INTEGER)"
            )
            conn.executemany(
                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ("gericht", "Gericht", "gericht", "Gericht", "court", "noun", "das", 0),
                    ("lauft", "läuft", "laufen", "laufen", "runs", "verb", "", 0),
                ],
            )

        client = DictionaryWordHintClient(db_path=db_path)
        hints, backend = build_selective_translation_result(
            "Das Gericht läuft.",
            backend="dictionary",
            client=client,
        )
        pairs = {(hint["source"], hint["translation"]) for hint in hints}
        assert backend == "dictionary"
        assert ("Gericht", "court") in pairs
        assert ("läuft", "runs") in pairs
        assert any(hint["source"] == "Gericht" and hint["article"] == "das" for hint in hints)


def main() -> None:
    test_selective_translations_extract_useful_words()
    test_selective_translations_are_english_only_for_now()
    test_ollama_word_hints_extract_many_nouns_and_verbs()
    test_ollama_word_hints_fall_back_to_builtin_glossary()
    test_dictionary_backend_uses_local_sqlite_lookup()
    print("translation helper tests passed")


if __name__ == "__main__":
    main()
