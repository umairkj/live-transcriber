from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dictionary import build_dictionary


def test_build_dictionary_from_small_tei() -> None:
    tei = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body>
    <entry>
      <form><orth>Gericht</orth></form>
      <gramGrp><gen>neut</gen><pos>n</pos></gramGrp>
      <sense><cit type="trans"><quote xml:lang="en">court</quote></cit></sense>
    </entry>
    <entry>
      <form><orth>laufen</orth></form>
      <gramGrp><pos>v</pos></gramGrp>
      <sense><cit type="trans"><quote xml:lang="en">run</quote></cit></sense>
    </entry>
  </body></text>
</TEI>
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "mini.tei"
        db_path = Path(temp_dir) / "de_en.sqlite"
        source.write_text(tei, encoding="utf-8")
        count = build_dictionary(source, db_path)

        assert count == 2
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT source, translation, kind, article FROM entries ORDER BY source").fetchall()
        assert ("Gericht", "court", "noun", "das") in rows
        assert ("laufen", "run", "verb", "") in rows


def main() -> None:
    test_build_dictionary_from_small_tei()
    print("dictionary builder tests passed")


if __name__ == "__main__":
    main()
