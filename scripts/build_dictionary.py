from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transcriber.translations import DEFAULT_DICTIONARY_PATH, _lookup_key


DEFAULT_FREEDICT_URL = "https://download.freedict.org/generated/deu-eng/deu-eng.tei"
TEI_NS = "{http://www.tei-c.org/ns/1.0}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local German-English word-hint dictionary.")
    parser.add_argument("--source", default=DEFAULT_FREEDICT_URL, help="FreeDict TEI path or URL.")
    parser.add_argument("--db", default=str(DEFAULT_DICTIONARY_PATH), help="Output SQLite DB path.")
    parser.add_argument("--cache-dir", default="data", help="Directory used for downloaded dictionary source files.")
    parser.add_argument("--limit", type=int, default=0, help="Optional entry limit for tests/debugging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = _download_if_needed(args.source, Path(args.cache_dir))
    db_path = Path(args.db)
    count = build_dictionary(source, db_path, limit=args.limit or None)
    print(f"Dictionary built: {db_path} ({count} rows)")
    return 0


def build_dictionary(source: Path, db_path: Path, limit: int | None = None) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_db = db_path.with_suffix(".tmp.sqlite")
    if temp_db.exists():
        temp_db.unlink()

    with sqlite3.connect(temp_db) as conn:
        _create_schema(conn)
        count = 0
        batch: list[tuple[str, str, str, str, str, str, str, int]] = []
        for entry_index, rows in enumerate(_iter_freedict_rows(source), start=1):
            batch.extend(rows)
            if len(batch) >= 1000:
                count += _insert_rows(conn, batch)
                batch.clear()
            if limit is not None and entry_index >= limit:
                break
        if batch:
            count += _insert_rows(conn, batch)
        conn.execute("CREATE INDEX idx_entries_key ON entries(key)")
        conn.execute("CREATE INDEX idx_entries_lemma_key ON entries(lemma_key)")
        conn.commit()

    temp_db.replace(db_path)
    return count


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE entries ("
        "key TEXT NOT NULL, "
        "source TEXT NOT NULL, "
        "lemma_key TEXT NOT NULL, "
        "lemma TEXT NOT NULL, "
        "translation TEXT NOT NULL, "
        "kind TEXT NOT NULL, "
        "article TEXT NOT NULL, "
        "priority INTEGER NOT NULL"
        ")"
    )


def _insert_rows(conn: sqlite3.Connection, rows: list[tuple[str, str, str, str, str, str, str, int]]) -> int:
    deduped = list(dict.fromkeys(rows))
    conn.executemany(
        "INSERT INTO entries (key, source, lemma_key, lemma, translation, kind, article, priority) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        deduped,
    )
    return len(deduped)


def _iter_freedict_rows(source: Path) -> Iterable[list[tuple[str, str, str, str, str, str, str, int]]]:
    context = ET.iterparse(source, events=("end",))
    for _event, elem in context:
        if elem.tag != f"{TEI_NS}entry":
            continue
        rows = _rows_from_entry(elem)
        if rows:
            yield rows
        elem.clear()


def _rows_from_entry(entry: ET.Element) -> list[tuple[str, str, str, str, str, str, str, int]]:
    sources = _entry_sources(entry)
    if not sources:
        return []

    kind = _entry_kind(entry)
    if kind not in {"noun", "verb"}:
        return []

    article = _entry_article(entry) if kind == "noun" else ""
    translations = _entry_translations(entry)
    if not translations:
        return []

    rows: list[tuple[str, str, str, str, str, str, str, int]] = []
    for source in sources:
        key = _lookup_key(source)
        if not key:
            continue
        for priority, translation in enumerate(translations[:4]):
            rows.append((key, source, key, source, translation, kind, article, priority))
    return rows


def _entry_sources(entry: ET.Element) -> list[str]:
    sources: list[str] = []
    for orth in entry.findall(f"./{TEI_NS}form/{TEI_NS}orth"):
        text = _clean_text("".join(orth.itertext()))
        if text and _is_single_word(text):
            sources.append(text)
    return list(dict.fromkeys(sources))


def _entry_kind(entry: ET.Element) -> str:
    pos_values = [
        _clean_text("".join(pos.itertext())).casefold()
        for pos in entry.findall(f"./{TEI_NS}gramGrp/{TEI_NS}pos")
    ]
    joined = " ".join(pos_values)
    if re.search(r"\b(n|noun|subst|s)\b", joined):
        return "noun"
    if re.search(r"\b(v|verb)\b", joined):
        return "verb"
    return ""


def _entry_article(entry: ET.Element) -> str:
    genders = [
        _clean_text("".join(gen.itertext())).casefold()
        for gen in entry.findall(f"./{TEI_NS}gramGrp/{TEI_NS}gen")
    ]
    if any(gender.startswith(("masc", "m")) for gender in genders):
        return "der"
    if any(gender.startswith(("fem", "f")) for gender in genders):
        return "die"
    if any(gender.startswith(("neut", "n")) for gender in genders):
        return "das"
    return ""


def _entry_translations(entry: ET.Element) -> list[str]:
    translations: list[str] = []
    for cit in entry.findall(f".//{TEI_NS}cit[@type='trans']"):
        quote = cit.find(f"./{TEI_NS}quote")
        if quote is None:
            continue
        text = _clean_translation("".join(quote.itertext()))
        if text:
            translations.append(text)
    return list(dict.fromkeys(translations))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_translation(value: str) -> str:
    value = _clean_text(value)
    value = re.sub(r"\s*\{[^}]*\}", "", value)
    value = re.sub(r"\s*\[[^]]*]", "", value)
    value = value.strip(" ;,")
    if not value or len(value) > 60:
        return ""
    return value


def _is_single_word(value: str) -> bool:
    return bool(re.fullmatch(r"[\wÄÖÜäöüß.-]+", value, flags=re.UNICODE))


def _download_if_needed(source: str, cache_dir: Path = Path("data")) -> Path:
    if not source.startswith(("http://", "https://")):
        return Path(source)

    target = cache_dir / Path(source).name
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dictionary source: {source}")
    with urllib.request.urlopen(source, timeout=30) as response:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
    os.replace(temp_path, target)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
