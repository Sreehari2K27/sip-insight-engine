"""Regression tests for the CSV evidence store. All tests run offline."""

import csv
from datetime import date

import pytest

from sip_engine.evidence_store import COLUMNS, EvidenceStore

TRICKY_TEXT = (
    "I paused my SIP after 6 months.\n"
    "The \u20b95,000 outflow felt heavy, \"no growth\" in a year, "
    "so I said, \"enough\" — \u092c\u0938 \u0930\u0941\u0915 \u0917\u092f\u093e\u0964"
)
TRICKY_URL = "https://example.com/r/IndiaInvestments/comments/abc/sip_pause/?utm=one,two"


def make_record(**overrides):
    record = {
        "platform": "reddit",
        "source_url": "https://example.com/thread/1",
        "author_handle": "u/investor123",
        "raw_text": "I paused my SIP after six months.",
        "topic_tags": "sip;pause",
    }
    record.update(overrides)
    return record


def test_round_trip_preserves_raw_text_and_source_url(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    stored = store.append(make_record(raw_text=TRICKY_TEXT, source_url=TRICKY_URL))

    rows = store.read_all()

    assert len(rows) == 1
    assert rows[0]["raw_text"] == TRICKY_TEXT
    assert rows[0]["source_url"] == TRICKY_URL
    assert rows[0]["evidence_id"] == stored["evidence_id"]


def test_csv_physical_format(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    store.append(make_record(raw_text=TRICKY_TEXT, source_url=TRICKY_URL))

    with (tmp_path / "evidence.csv").open("r", newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))

    assert len(parsed) == 1
    assert parsed[0]["raw_text"] == TRICKY_TEXT
    assert "\n" in parsed[0]["raw_text"]


def test_append_preserves_earlier_rows(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    first = store.append(make_record())
    second = store.append(make_record(source_url="https://example.com/thread/2"))

    rows = store.read_all()

    assert [r["evidence_id"] for r in rows] == [first["evidence_id"], second["evidence_id"]]
    assert rows[0]["raw_text"] == first["raw_text"]
    assert rows[0]["source_url"] == first["source_url"]
    assert rows[1]["source_url"] == "https://example.com/thread/2"


def test_append_to_existing_header_only_file(tmp_path):
    path = tmp_path / "evidence.csv"
    path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
    store = EvidenceStore(path)

    store.append(make_record())

    text = path.read_text(encoding="utf-8")
    assert text.count("evidence_id") == 1
    assert store.count() == 1


def test_missing_required_fields_raise(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")

    with pytest.raises(ValueError):
        store.append({"raw_text": "no url supplied"})
    with pytest.raises(ValueError):
        store.append({"source_url": "https://example.com/thread/3"})
    with pytest.raises(ValueError):
        store.append({})


def test_unknown_column_raises(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")

    with pytest.raises(ValueError):
        store.append(make_record(sentiment="happy"))


def test_defaults_fill_id_and_capture_date(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")

    stored = store.append(make_record())

    assert stored["evidence_id"]
    assert stored["captured_date"] == date.today().isoformat()


def test_explicit_id_and_date_are_kept(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")

    stored = store.append(
        make_record(evidence_id="abc123", captured_date="2026-01-15")
    )

    assert stored["evidence_id"] == "abc123"
    assert stored["captured_date"] == "2026-01-15"


def test_read_all_on_missing_file_returns_empty(tmp_path):
    store = EvidenceStore(tmp_path / "missing.csv")

    assert store.read_all() == []
