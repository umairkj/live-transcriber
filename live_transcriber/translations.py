from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal
from urllib import error, request


DEFAULT_TRANSLATION_TARGET_LANGUAGE = "en"
DEFAULT_SELECTIVE_TRANSLATION_BACKEND = "built_in_glossary"
SELECTIVE_TRANSLATION_BACKENDS = ("built_in_glossary", "dictionary", "ollama")
DEFAULT_WORD_HINT_MODEL = "llama3.1:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_DICTIONARY_PATH = Path("data/de_en.sqlite")
TranslationKind = Literal["noun", "verb", "adjective", "adverb", "phrase"]
Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class TranslationError(RuntimeError):
    """Raised when a translation backend cannot produce usable word hints."""


@dataclass(frozen=True)
class SelectiveTranslation:
    source: str
    translation: str
    kind: TranslationKind
    article: str | None = None

    def to_record(self) -> dict[str, str]:
        record = asdict(self)
        return {key: value for key, value in record.items() if value is not None}


_LEXICON: dict[str, tuple[str, TranslationKind]] = {
    "abend": ("evening", "noun"),
    "angelegt": ("put on", "verb"),
    "angefangen": ("started", "verb"),
    "anwälte": ("lawyers", "noun"),
    "anwalte": ("lawyers", "noun"),
    "anwaelte": ("lawyers", "noun"),
    "arbeit": ("work", "noun"),
    "arbeitet": ("works", "verb"),
    "aufgeklappt": ("opened up", "verb"),
    "bereit": ("ready", "adjective"),
    "besetzt": ("occupied", "verb"),
    "bezahlen": ("pay", "verb"),
    "kommt": ("comes", "verb"),
    "durften": ("were allowed", "verb"),
    "diskutieren": ("discuss", "verb"),
    "ende": ("end", "noun"),
    "erzählen": ("tell", "verb"),
    "erzahlen": ("tell", "verb"),
    "erzaehlen": ("tell", "verb"),
    "ehemann": ("husband", "noun"),
    "fruhsport": ("morning exercise", "noun"),
    "fruhstucke": ("eat breakfast", "verb"),
    "fruhstucken": ("eat breakfast", "verb"),
    "geführt": ("led", "verb"),
    "gefuhrt": ("led", "verb"),
    "gefuehrt": ("led", "verb"),
    "gefragt": ("asked", "verb"),
    "gearbeitet": ("worked", "verb"),
    "gehe": ("go", "verb"),
    "gehen": ("go", "verb"),
    "geld": ("money", "noun"),
    "genau": ("exactly", "adverb"),
    "gerade": ("just now", "adverb"),
    "gericht": ("court", "noun"),
    "gesund": ("healthy", "adjective"),
    "glaskasten": ("glass box", "noun"),
    "glas": ("glass", "noun"),
    "glucklich": ("happy", "adjective"),
    "goldkette": ("gold necklace", "noun"),
    "grundlage": ("foundation", "noun"),
    "hab": ("have", "verb"),
    "habe": ("have", "verb"),
    "haben": ("have", "verb"),
    "hals": ("neck", "noun"),
    "handy": ("phone", "noun"),
    "hat": ("has", "verb"),
    "jahr": ("year", "noun"),
    "jahren": ("years", "noun"),
    "kleid": ("dress", "noun"),
    "kette": ("necklace", "noun"),
    "leben": ("life", "noun"),
    "laptop": ("laptop", "noun"),
    "langsam": ("slowly", "adverb"),
    "läuft": ("runs", "verb"),
    "lauft": ("runs", "verb"),
    "laeuft": ("runs", "verb"),
    "liegen": ("lie", "verb"),
    "menschen": ("people", "noun"),
    "mache": ("do", "verb"),
    "machen": ("do", "verb"),
    "meditiere": ("meditate", "verb"),
    "meditieren": ("meditate", "verb"),
    "monaten": ("months", "noun"),
    "möchte": ("wants", "verb"),
    "mochte": ("wants", "verb"),
    "morgen": ("morning", "noun"),
    "morgenroutine": ("morning routine", "noun"),
    "muster": ("pattern", "noun"),
    "opfer": ("victim", "noun"),
    "panzer": ("armor", "noun"),
    "platz": ("place", "noun"),
    "prozess": ("trial", "noun"),
    "putze": ("clean", "verb"),
    "putzen": ("clean", "verb"),
    "rechnung": ("bill", "noun"),
    "routine": ("routine", "noun"),
    "saal": ("hall", "noun"),
    "saals": ("hall", "noun"),
    "salz": ("salt", "noun"),
    "salzs": ("salt", "noun"),
    "sein": ("be", "verb"),
    "september": ("September", "noun"),
    "sitzt": ("sits", "verb"),
    "sonne": ("sun", "noun"),
    "sommer": ("summer", "noun"),
    "sommertag": ("summer day", "noun"),
    "sport": ("exercise", "noun"),
    "starten": ("start", "verb"),
    "tag": ("day", "noun"),
    "tatsaechlich": ("actually", "adverb"),
    "tatsachlich": ("actually", "adverb"),
    "trägt": ("wears", "verb"),
    "tragt": ("wears", "verb"),
    "traegt": ("wears", "verb"),
    "trinke": ("drink", "verb"),
    "trinken": ("drink", "verb"),
    "vorhange": ("curtains", "noun"),
    "war": ("was", "verb"),
    "wach": ("awake", "adjective"),
    "warm": ("warm", "adjective"),
    "warmes": ("warm", "adjective"),
    "waschen": ("wash", "verb"),
    "wasser": ("water", "noun"),
    "wichtig": ("important", "adjective"),
    "wohlstand": ("prosperity", "noun"),
    "zaehne": ("teeth", "noun"),
    "zahne": ("teeth", "noun"),
    "zahneputzen": ("brush teeth", "verb"),
    "zeit": ("time", "noun"),
}


_NOUN_ARTICLES = {
    "abend": "der",
    "arbeit": "die",
    "anwälte": "die",
    "anwalte": "die",
    "anwaelte": "die",
    "ehemann": "der",
    "ende": "das",
    "fruhsport": "der",
    "geld": "das",
    "gericht": "das",
    "glaskasten": "der",
    "glas": "das",
    "goldkette": "die",
    "grundlage": "die",
    "hals": "der",
    "handy": "das",
    "jahr": "das",
    "jahren": "die",
    "kleid": "das",
    "kette": "die",
    "leben": "das",
    "laptop": "der",
    "menschen": "die",
    "monaten": "die",
    "morgen": "der",
    "morgenroutine": "die",
    "muster": "das",
    "opfer": "das",
    "panzer": "der",
    "platz": "der",
    "prozess": "der",
    "rechnung": "die",
    "routine": "die",
    "saal": "der",
    "saals": "der",
    "salz": "das",
    "salzs": "das",
    "september": "der",
    "sonne": "die",
    "sommer": "der",
    "sommertag": "der",
    "sport": "der",
    "tag": "der",
    "vorhange": "die",
    "wasser": "das",
    "wohlstand": "der",
    "zaehne": "die",
    "zahne": "die",
    "zeit": "die",
}


_GERMAN_FUNCTION_WORDS = {
    "aber",
    "am",
    "an",
    "auf",
    "aus",
    "bis",
    "da",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "diesem",
    "dieser",
    "du",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eigentlich",
    "er",
    "es",
    "hat",
    "ich",
    "ihm",
    "ihn",
    "ihre",
    "ihren",
    "im",
    "in",
    "ist",
    "ja",
    "kein",
    "keinen",
    "mehr",
    "mich",
    "mit",
    "nach",
    "nicht",
    "noch",
    "nun",
    "oder",
    "sein",
    "sie",
    "sich",
    "und",
    "vom",
    "von",
    "vor",
    "warum",
    "wir",
    "zu",
    "zum",
}

_COMMON_GERMAN_VERBS = {
    "bin",
    "bist",
    "durften",
    "darf",
    "dürfen",
    "durfen",
    "gehabt",
    "hab",
    "habe",
    "haben",
    "hat",
    "hatte",
    "ist",
    "kann",
    "können",
    "konnen",
    "läuft",
    "lauft",
    "muss",
    "möchte",
    "mochte",
    "sein",
    "sind",
    "sitz",
    "sitzt",
    "war",
    "waren",
    "wird",
}


_SKIPPED_LIGHT_VERB_KEYS = {
    "hab",
    "habe",
    "haben",
    "hast",
    "hat",
    "hatte",
    "hatten",
    "gehabt",
    "mach",
    "mache",
    "machen",
    "macht",
    "machte",
    "machten",
    "gemacht",
    "tu",
    "tue",
    "tun",
    "tut",
    "tat",
    "taten",
    "getan",
}

_SKIPPED_LIGHT_VERB_TRANSLATIONS = {
    "do",
    "does",
    "did",
    "done",
    "have",
    "has",
    "had",
    "make",
    "makes",
    "made",
}


def build_selective_translations(
    text: str,
    target_language: str = DEFAULT_TRANSLATION_TARGET_LANGUAGE,
    max_items: int = 80,
    backend: str = DEFAULT_SELECTIVE_TRANSLATION_BACKEND,
    client: "OllamaWordHintClient | DictionaryWordHintClient | None" = None,
) -> list[dict[str, str]]:
    hints, _backend = build_selective_translation_result(
        text,
        target_language=target_language,
        max_items=max_items,
        backend=backend,
        client=client,
    )
    return hints


def build_selective_translation_result(
    text: str,
    target_language: str = DEFAULT_TRANSLATION_TARGET_LANGUAGE,
    max_items: int = 80,
    backend: str = DEFAULT_SELECTIVE_TRANSLATION_BACKEND,
    client: "OllamaWordHintClient | DictionaryWordHintClient | None" = None,
) -> tuple[list[dict[str, str]], str]:
    if target_language != "en":
        return [], backend

    builtin_hints = _build_builtin_selective_translations(text, max_items=max_items)
    if backend == "built_in_glossary":
        return builtin_hints, "built_in_glossary"
    if backend == "dictionary":
        try:
            dictionary_client = client if client is not None else DictionaryWordHintClient()
            dictionary_hints = dictionary_client.build_hints(text, max_items=max_items)
        except TranslationError:
            return builtin_hints, "built_in_glossary"
        return _merge_hints(dictionary_hints, builtin_hints, max_items=max_items), "dictionary"
    if backend != "ollama":
        return builtin_hints, "built_in_glossary"

    try:
        llm_client = client if client is not None else OllamaWordHintClient()
        llm_hints = llm_client.build_hints(text, max_items=max_items)
    except TranslationError:
        dictionary_hints = DictionaryWordHintClient().build_hints(text, max_items=max_items)
        fallback = _merge_hints(dictionary_hints, builtin_hints, max_items=max_items)
        return fallback, "dictionary" if dictionary_hints else "built_in_glossary"

    dictionary_hints = DictionaryWordHintClient().build_hints(text, max_items=max_items)
    fallback = _merge_hints(dictionary_hints, builtin_hints, max_items=max_items)
    return _merge_hints(llm_hints, fallback, max_items=max_items), "ollama"


class DictionaryWordHintClient:
    """Fast German noun/verb hints from spaCy POS tags plus local dictionary lookup."""

    def __init__(self, db_path: Path | str = DEFAULT_DICTIONARY_PATH) -> None:
        self.db_path = Path(db_path)
        self._nlp = _load_german_spacy_model()

    def build_hints(self, text: str, max_items: int = 80) -> list[dict[str, str]]:
        text = text.strip()
        if not text:
            return []

        candidates = self._spacy_candidates(text) if self._nlp is not None else self._heuristic_candidates(text)
        hints: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = _lookup_key(candidate["source"])
            if not key or key in seen:
                continue

            lookup = self._lookup(candidate["source"], candidate.get("lemma", ""))
            kind = candidate.get("kind") or lookup.get("kind", "")
            if kind not in {"noun", "verb"}:
                continue

            translation = lookup.get("translation", "")
            if not translation:
                continue

            article = candidate.get("article") or lookup.get("article", "")
            record = {
                "source": candidate["source"],
                "translation": translation,
                "kind": kind,
            }
            if kind == "noun" and article:
                record["article"] = article
            hints.append(record)
            seen.add(key)
            if len(hints) >= max_items:
                break

        return hints

    def _spacy_candidates(self, text: str) -> list[dict[str, str]]:
        assert self._nlp is not None
        candidates: list[dict[str, str]] = []
        for token in self._nlp(text):
            if not token.text or not any(char.isalpha() for char in token.text):
                continue
            kind = _kind_from_spacy(token)
            if kind not in {"noun", "verb"}:
                continue
            article = _article_from_spacy(token) if kind == "noun" else ""
            candidates.append(
                {
                    "source": token.text,
                    "lemma": token.lemma_ if token.lemma_ and token.lemma_ != "--" else token.text,
                    "kind": kind,
                    "article": article,
                }
            )
        return candidates

    def _heuristic_candidates(self, text: str) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        for index, word in enumerate(_words(text)):
            key = _lookup_key(word)
            lookup = self._lookup(word, "")
            kind = lookup.get("kind", "")
            if not kind and _looks_like_german_noun(word, index):
                kind = "noun"
            elif not kind and _looks_like_german_verb(word):
                kind = "verb"
            if kind in {"noun", "verb"}:
                candidates.append(
                    {
                        "source": word,
                        "lemma": word,
                        "kind": kind,
                        "article": lookup.get("article", "") or _NOUN_ARTICLES.get(key, ""),
                    }
                )
        return candidates

    def _lookup(self, source: str, lemma: str) -> dict[str, str]:
        source_key = _lookup_key(source)
        lemma_key = _lookup_key(lemma)
        for key in (source_key, lemma_key):
            if not key:
                continue
            seeded = _seed_lookup(key)
            if seeded:
                return seeded

        sqlite_result = _sqlite_lookup(str(self.db_path), source_key, lemma_key)
        return sqlite_result or {}


class OllamaWordHintClient:
    """Ask a local Ollama model for German noun/verb hints as strict JSON."""

    def __init__(
        self,
        model: str = DEFAULT_WORD_HINT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 25.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport or self._post_json

    def build_hints(self, text: str, max_items: int = 80) -> list[dict[str, str]]:
        text = text.strip()
        if not text:
            return []

        candidates = _words(text)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self._system_prompt(max_items)},
                {"role": "user", "content": self._user_prompt(text, candidates)},
            ],
            "options": {"temperature": 0.0},
        }
        response = self._transport(f"{self.base_url}/api/chat", payload, self.timeout)
        content = response.get("message", {}).get("content") if isinstance(response, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise TranslationError("Ollama returned an empty word-hint response.")
        hints = _parse_llm_hints(content, text=text, max_items=max_items)
        missing_candidates = _likely_missing_candidates(text, hints, max_candidates=max(0, max_items - len(hints)))
        if missing_candidates:
            try:
                extra_hints = self._classify_candidates(text, missing_candidates, max_items=max_items)
            except TranslationError:
                extra_hints = []
            hints = _merge_hints(hints, extra_hints, max_items=max_items)
        return hints

    def _system_prompt(self, max_items: int) -> str:
        return (
            "You create word-by-word German learning hints. Return JSON only, with this shape: "
            '{"items":[{"source":"German surface word","translation":"English lemma","kind":"noun|verb","article":"der|die|das"}]}. '
            "Find every German noun and verb in the user's transcript snippet, up to "
            f"{max_items} items. You will receive a candidate word list; inspect every candidate in order. "
            "Include each candidate that is a noun or verb. The source must be exactly one word as it appears in the text. "
            "For nouns, include the correct German article der, die, or das, using your best guess when needed. "
            "For verbs, omit article. "
            "Use short English translations like victim, ask, sit, run, wear. "
            "Include finite verbs, infinitives, auxiliaries, and participles such as gefragt, angelegt, besetzt, getragen. "
            "Do not include adjectives, adverbs, pronouns, determiners, numbers, punctuation, or invented words."
        )

    def _user_prompt(self, text: str, candidates: list[str]) -> str:
        candidate_lines = "\n".join(f"{index}. {word}" for index, word in enumerate(candidates, start=1))
        return (
            "Transcript:\n"
            f"{text}\n\n"
            "Candidate words in order:\n"
            f"{candidate_lines}\n\n"
            "Return JSON only. Include every candidate that is a German noun or verb."
        )

    def _classify_candidates(self, text: str, candidates: list[str], max_items: int) -> list[dict[str, str]]:
        candidate_lines = "\n".join(f"{index}. {word}" for index, word in enumerate(candidates, start=1))
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. Start with { and end with }. Do not explain. Do not use Markdown. "
                        "Use this shape: "
                        '{"items":[{"source":"word","translation":"English lemma","kind":"noun|verb|other","article":"der|die|das"}]}. '
                        "Classify every candidate as noun, verb, or other. Include translations only for nouns and verbs. "
                        "For nouns, include der, die, or das. Include participles and auxiliary verbs as verbs."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Transcript:\n"
                        f"{text}\n\n"
                        "Likely missed candidates:\n"
                        f"{candidate_lines}"
                    ),
                },
            ],
            "options": {"temperature": 0.0},
        }
        response = self._transport(f"{self.base_url}/api/chat", payload, min(self.timeout, 18.0))
        content = response.get("message", {}).get("content") if isinstance(response, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise TranslationError("Ollama returned an empty word-hint response.")
        return _parse_llm_hints(content, text=text, max_items=max_items)

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
            raise TranslationError(_ollama_error_message(body, self.model)) from exc
        except error.URLError as exc:
            raise TranslationError("Ollama is not running. Start it with: ollama serve") from exc
        except TimeoutError as exc:
            raise TranslationError("Ollama word-hint update timed out.") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TranslationError("Ollama returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise TranslationError("Ollama returned an unexpected response.")
        if isinstance(parsed.get("error"), str):
            raise TranslationError(_ollama_error_message(str(parsed["error"]), self.model))
        return parsed


def _build_builtin_selective_translations(text: str, max_items: int = 80) -> list[dict[str, str]]:
    word_list = _words(text)
    return _records_from_entries(
        [
            {
                "source": word,
                "translation": _LEXICON[_lookup_key(word)][0],
                "kind": _LEXICON[_lookup_key(word)][1],
                "article": _NOUN_ARTICLES.get(_lookup_key(word)),
            }
            for word in word_list
            if _lookup_key(word) in _LEXICON
        ],
        text=text,
        max_items=max_items,
    )


def _parse_llm_hints(content: str, text: str, max_items: int) -> list[dict[str, str]]:
    raw_json = _extract_json(content)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise TranslationError("Ollama returned word hints that were not JSON.") from exc

    if isinstance(parsed, dict):
        items = parsed.get("items")
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = None
    if not isinstance(items, list):
        raise TranslationError("Ollama word hints did not include an items list.")

    return _records_from_entries(items, text=text, max_items=max_items)


def _extract_json(content: str) -> str:
    value = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        value = fence_match.group(1).strip()
    if not value.startswith(("{", "[")):
        json_match = re.search(r"(\{.*\}|\[.*\])", value, flags=re.DOTALL)
        if json_match:
            value = json_match.group(1)
    return value


def _records_from_entries(items: list[object], text: str, max_items: int) -> list[dict[str, str]]:
    allowed_sources = {_lookup_key(word): word for word in _words(text)}
    hints: list[SelectiveTranslation] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source") or "").strip()
        translation = str(item.get("translation") or "").strip()
        kind = str(item.get("kind") or "").strip().casefold()
        article = str(item.get("article") or "").strip().casefold()
        if kind not in {"noun", "verb"} or not source or not translation:
            continue

        source_key = _lookup_key(source)
        if source_key not in allowed_sources or source_key in seen:
            continue
        if kind == "noun" and article not in {"der", "die", "das"}:
            article = _NOUN_ARTICLES.get(source_key, "")

        hints.append(
            SelectiveTranslation(
                source=allowed_sources[source_key],
                translation=translation,
                kind=kind,  # type: ignore[arg-type]
                article=article or None,
            )
        )
        seen.add(source_key)
        if len(hints) >= max_items:
            break

    return [hint.to_record() for hint in hints]


def _merge_hints(
    primary: list[dict[str, str]],
    fallback: list[dict[str, str]],
    max_items: int,
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    fallback_by_key = {_lookup_key(str(item.get("source") or "")): item for item in fallback}
    for item in [*primary, *fallback]:
        source = str(item.get("source") or "")
        key = _lookup_key(source)
        if not key or key in seen:
            continue
        enriched = dict(item)
        fallback_item = fallback_by_key.get(key)
        if fallback_item is not None:
            enriched.setdefault("translation", str(fallback_item.get("translation") or ""))
            enriched.setdefault("article", str(fallback_item.get("article") or ""))
        merged.append({key: value for key, value in enriched.items() if value})
        seen.add(key)
        if len(merged) >= max_items:
            break
    return merged


@lru_cache(maxsize=1)
def _load_german_spacy_model():  # noqa: ANN202
    try:
        import spacy
    except ModuleNotFoundError:
        return None

    try:
        return spacy.load("de_core_news_sm", disable=["ner"])
    except OSError:
        return None


def _kind_from_spacy(token) -> str:  # noqa: ANN001
    pos = str(getattr(token, "pos_", "") or "")
    tag = str(getattr(token, "tag_", "") or "")
    if pos in {"NOUN", "PROPN"} or tag in {"NN", "NE"}:
        return "noun"
    if pos in {"VERB", "AUX"} or tag.startswith(("V", "VA", "VM", "VV")):
        return "verb"
    return ""


def _article_from_spacy(token) -> str:  # noqa: ANN001
    genders = token.morph.get("Gender") if getattr(token, "morph", None) is not None else []
    if "Masc" in genders:
        return "der"
    if "Fem" in genders:
        return "die"
    if "Neut" in genders:
        return "das"
    return _NOUN_ARTICLES.get(_lookup_key(str(getattr(token, "lemma_", "") or token.text)), "")


@lru_cache(maxsize=4096)
def _seed_lookup(key: str) -> dict[str, str]:
    entry = _LEXICON.get(key)
    if entry is None:
        return {}
    translation, kind = entry
    result = {
        "translation": translation,
        "kind": kind,
    }
    article = _NOUN_ARTICLES.get(key)
    if kind == "noun" and article:
        result["article"] = article
    return result


@lru_cache(maxsize=8192)
def _sqlite_lookup(db_path: str, source_key: str, lemma_key: str) -> dict[str, str]:
    path = Path(db_path)
    if not path.exists():
        return {}

    keys = [key for key in (source_key, lemma_key) if key]
    if not keys:
        return {}

    placeholders = ",".join("?" for _key in keys)
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT translation, kind, article FROM entries "
                f"WHERE key IN ({placeholders}) OR lemma_key IN ({placeholders}) "
                "ORDER BY priority ASC, LENGTH(translation) ASC LIMIT 1",
                [*keys, *keys],
            ).fetchone()
    except sqlite3.Error:
        return {}

    if row is None:
        return {}
    translation, kind, article = row
    return {
        "translation": str(translation or ""),
        "kind": str(kind or ""),
        "article": str(article or ""),
    }


def _likely_missing_candidates(
    text: str,
    existing_hints: list[dict[str, str]],
    max_candidates: int,
) -> list[str]:
    if max_candidates <= 0:
        return []

    existing = {_lookup_key(str(item.get("source") or "")) for item in existing_hints}
    candidates: list[str] = []
    seen: set[str] = set()
    for index, word in enumerate(_words(text)):
        key = _lookup_key(word)
        if key in seen or key in existing:
            continue
        if key in _LEXICON:
            continue
        if key in _GERMAN_FUNCTION_WORDS and key not in _COMMON_GERMAN_VERBS:
            continue
        if _looks_like_german_noun(word, index) or _looks_like_german_verb(word):
            candidates.append(word)
            seen.add(key)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _looks_like_german_noun(word: str, index: int) -> bool:
    key = _lookup_key(word)
    if key in _NOUN_ARTICLES:
        return True
    if not word[:1].isupper():
        return False
    if index == 0 and key in _GERMAN_FUNCTION_WORDS:
        return False
    return len(word) > 2


def _looks_like_german_verb(word: str) -> bool:
    key = _lookup_key(word)
    if key in _COMMON_GERMAN_VERBS:
        return True
    if len(key) <= 3:
        return False
    return bool(
        re.search(r"(en|ern|eln|est|st|te|ten|tet|t)$", key)
        or re.match(r"ge\w+(t|en)$", key)
    )


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"\b\w+\b", text, flags=re.UNICODE) if any(char.isalpha() for char in word)]


def _lookup_key(word: str) -> str:
    word = word.casefold().replace("\u00df", "ss")
    normalized = unicodedata.normalize("NFKD", word)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped


def _ollama_error_message(message: str, model: str) -> str:
    normalized = message.casefold()
    if "not found" in normalized or "pull" in normalized or ("model" in normalized and "missing" in normalized):
        model_name = model.split(":", 1)[0]
        return f"Model {model} is missing. Install it with: ollama pull {model_name}"
    return message.strip() or "Ollama word-hint update failed."
