"""Regression tests for the CSV evidence store. All tests run offline."""

import csv
from datetime import date

import pytest

from sip_engine import evidence_store as evidence_store_module
from sip_engine.evidence_store import (
    ANALYSIS_TO_CSV,
    COLUMNS,
    INTERIM_COLUMNS,
    LEGACY_COLUMNS,
    EvidenceStore,
)

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


def test_current_20_column_header_reads_without_rewrite(tmp_path):
    path = tmp_path / "evidence.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerow([
            "id20", "2026-10-20", "appstore", "https://example.com/r/1", "u/x",
            'I paused my SIP. \u20b95,000, "heavy" — \u092c\u0938.', "tags",
            "", "", "",
            "negative", "unknown", "paused behaviour", "paused", "market losses",
            "I paused my SIP.", "rule-based mock",
            "a" * 64, "2026-10-19T10:00:00+05:30", "2026-10-20T09:00:00+00:00",
        ])
    before = path.read_bytes()
    store = EvidenceStore(path)

    assert store.migrate() is False

    rows = store.read_all()
    assert rows[0]["evidence_id"] == "id20"
    assert rows[0]["raw_text"] == 'I paused my SIP. \u20b95,000, "heavy" — \u092c\u0938.'
    assert rows[0]["source_url"] == "https://example.com/r/1"
    assert rows[0]["source_key"] == "a" * 64
    assert rows[0]["published_at"] == "2026-10-19T10:00:00+05:30"
    assert rows[0]["collected_at"] == "2026-10-20T09:00:00+00:00"
    assert rows[0]["ai_sentiment"] == "negative"
    assert rows[0]["ai_sip_status"] == "paused"
    assert path.read_bytes() == before


def test_superset_header_from_stale_process_raises_actionable_error(tmp_path, monkeypatch):
    path = tmp_path / "evidence.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerow([
            "id21", "2026-10-20", "appstore", "https://example.com/r/2", "u/y",
            "text", "tags", "", "", "",
            "", "", "", "", "", "", "",
            "k" * 64, "2026-10-19", "2026-10-20T09:00:00+00:00",
        ])
    before = path.read_bytes()

    stale_columns = COLUMNS[:17]
    monkeypatch.setattr(evidence_store_module, "COLUMNS", stale_columns)
    monkeypatch.setattr(evidence_store_module, "INTERIM_COLUMNS", LEGACY_COLUMNS)

    store = EvidenceStore(path)

    with pytest.raises(ValueError, match="restart the Streamlit server"):
        store.migrate()

    assert path.read_bytes() == before


def test_genuinely_incompatible_header_keeps_original_error(tmp_path):
    path = tmp_path / "evidence.csv"
    path.write_bytes(b"foo,bar\n1,2\n")
    store = EvidenceStore(path)

    with pytest.raises(ValueError, match="Incompatible evidence CSV header") as raised:
        store.migrate()

    assert "newer than this running process" not in str(raised.value)
    assert path.read_bytes() == b"foo,bar\n1,2\n"


def make_analysis(**overrides):
    analysis = {
        "sentiment": "negative",
        "motivation": "unknown",
        "behaviour": "describes pausing the SIP",
        "problem_theme": "market volatility",
        "sip_status": "paused",
        "discontinuation_reason": "market losses",
        "supporting_quote": "I paused my SIP.",
        "uncertainty": "rule-based mock output",
    }
    analysis.update(overrides)
    return analysis


def test_schema_keeps_original_columns_and_adds_analysis_columns():
    assert len(COLUMNS) == 20
    assert len(set(COLUMNS)) == len(COLUMNS)
    assert COLUMNS[:10] == LEGACY_COLUMNS
    assert COLUMNS[:17] == INTERIM_COLUMNS
    assert COLUMNS[17:] == ("source_key", "published_at", "collected_at")
    assert set(ANALYSIS_TO_CSV.values()) <= set(COLUMNS)


def test_legacy_file_migrates_additively(tmp_path):
    path = tmp_path / "evidence.csv"
    legacy_header = ",".join(LEGACY_COLUMNS)
    legacy_row = (
        "id001,2026-01-15,reddit,https://example.com/t1,u/ami,"
        '"I paused my SIP.\nCosts \u20b95,000, felt ""heavy"".",sip;pause,,,'
    )
    path.write_bytes((legacy_header + "\n" + legacy_row + "\n").encode("utf-8"))
    store = EvidenceStore(path)

    assert store.migrate() is True

    rows = store.read_all()
    assert len(rows) == 1
    assert rows[0]["evidence_id"] == "id001"
    assert rows[0]["raw_text"] == "I paused my SIP.\nCosts \u20b95,000, felt \"heavy\"."
    assert rows[0]["source_url"] == "https://example.com/t1"
    for column in COLUMNS[10:]:
        assert rows[0][column] == ""
    assert store.migrate() is False


def test_interim_17_column_file_migrates(tmp_path):
    path = tmp_path / "evidence.csv"
    path.write_bytes((",".join(INTERIM_COLUMNS) + "\n").encode("utf-8"))
    store = EvidenceStore(path)

    assert store.migrate() is True
    assert store.migrate() is False
    assert store.read_all() == []


def test_interim_file_with_data_migrates_preserving_values(tmp_path):
    path = tmp_path / "evidence.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(INTERIM_COLUMNS)
        writer.writerow(
            ["id17", "2026-10-20", "reddit", "https://example.com/t17", "u/x",
             TRICKY_TEXT, "tags", "", "", "", "", "", "", "", "", "", ""]
        )
    store = EvidenceStore(path)

    assert store.migrate() is True

    rows = store.read_all()
    assert rows[0]["evidence_id"] == "id17"
    assert rows[0]["raw_text"] == TRICKY_TEXT
    assert rows[0]["source_key"] == ""
    assert rows[0]["published_at"] == ""
    assert rows[0]["collected_at"] == ""
    assert store.migrate() is False


def test_new_collection_fields_round_trip(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    stored = store.append(
        make_record(
            raw_text=TRICKY_TEXT,
            source_key="a" * 64,
            published_at="2026-10-19T10:00:00+05:30",
            collected_at="2026-10-20T09:00:00+05:30",
        )
    )

    rows = store.read_all()

    assert rows[0]["source_key"] == "a" * 64
    assert rows[0]["published_at"] == "2026-10-19T10:00:00+05:30"
    assert rows[0]["collected_at"] == "2026-10-20T09:00:00+05:30"
    assert rows[0]["raw_text"] == TRICKY_TEXT
    assert rows[0]["source_url"] == stored["source_url"]


def test_incompatible_header_fails_safely(tmp_path):
    path = tmp_path / "evidence.csv"
    bad_content = "foo,bar\n1,2\n"
    path.write_text(bad_content, encoding="utf-8")
    store = EvidenceStore(path)

    with pytest.raises(ValueError):
        store.read_all()
    with pytest.raises(ValueError):
        store.append(make_record())
    with pytest.raises(ValueError):
        store.migrate()

    assert path.read_text(encoding="utf-8") == bad_content


def test_save_analysis_persists_and_preserves_originals(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    stored = store.append(make_record(raw_text=TRICKY_TEXT, source_url=TRICKY_URL))
    analysis = make_analysis(evidence_id=stored["evidence_id"])

    saved = store.save_analysis(stored["evidence_id"], analysis)

    rows = store.read_all()
    assert len(rows) == 1
    assert saved["evidence_id"] == stored["evidence_id"]
    assert rows[0]["ai_sentiment"] == "negative"
    assert rows[0]["ai_motivation"] == "unknown"
    assert rows[0]["ai_behaviour"] == "describes pausing the SIP"
    assert rows[0]["ai_theme"] == "market volatility"
    assert rows[0]["ai_sip_status"] == "paused"
    assert rows[0]["ai_discontinuation_reason"] == "market losses"
    assert rows[0]["ai_supporting_quote"] == "I paused my SIP."
    assert rows[0]["ai_uncertainty"] == "rule-based mock output"
    assert rows[0]["ai_summary"] == ""
    assert rows[0]["confidence"] == ""
    assert rows[0]["raw_text"] == TRICKY_TEXT
    assert rows[0]["source_url"] == TRICKY_URL
    assert rows[0]["captured_date"] == stored["captured_date"]
    assert rows[0]["topic_tags"] == stored["topic_tags"]


def test_repeated_analysis_updates_do_not_duplicate_rows(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    stored = store.append(make_record())
    first = make_analysis(evidence_id=stored["evidence_id"])
    second = make_analysis(
        evidence_id=stored["evidence_id"],
        sentiment="mixed",
        uncertainty="updated after re-read",
    )

    store.save_analysis(stored["evidence_id"], first)
    store.save_analysis(stored["evidence_id"], second)

    rows = store.read_all()
    assert store.count() == 1
    assert rows[0]["ai_sentiment"] == "mixed"
    assert rows[0]["ai_uncertainty"] == "updated after re-read"
    assert rows[0]["raw_text"] == stored["raw_text"]
    assert rows[0]["source_url"] == stored["source_url"]


def test_save_analysis_rejects_unknown_evidence_id(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    store.append(make_record())
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        store.save_analysis("missing-id", make_analysis())

    assert path.read_text(encoding="utf-8") == before


def test_save_analysis_rejects_malformed_analysis(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    stored = store.append(make_record())
    before = path.read_text(encoding="utf-8")
    incomplete = {k: v for k, v in make_analysis().items() if k != "uncertainty"}

    with pytest.raises(ValueError):
        store.save_analysis(stored["evidence_id"], make_analysis(credit="A+"))
    with pytest.raises(ValueError):
        store.save_analysis(stored["evidence_id"], incomplete)

    assert path.read_text(encoding="utf-8") == before


def test_save_analysis_rejects_mismatched_evidence_id(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    stored = store.append(make_record())

    with pytest.raises(ValueError):
        store.save_analysis(stored["evidence_id"], make_analysis(evidence_id="other"))


def test_save_analysis_refuses_ambiguous_duplicate_ids(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    store.append(make_record(evidence_id="dup"))
    store.append(make_record(evidence_id="dup", source_url="https://example.com/t2"))

    with pytest.raises(ValueError):
        store.save_analysis("dup", make_analysis())
