"""Focused offline tests for the AI analysis CLI (mocks + temp data only).

No network, no real LLM/API calls during implementation or tests.
"""

import csv

import pytest

from sip_engine import analyse as analyse_module
from sip_engine.analyse import analyse_records, is_analysed, main, select_records
from sip_engine.evidence_store import EvidenceStore


class FakeAnalyser:
    name = "fake"

    def __init__(self, fail_for=None):
        self.calls = []
        self.fail_for = set(fail_for or ())

    def analyse(self, record):
        self.calls.append(record.get("evidence_id", ""))
        if record.get("evidence_id", "") in self.fail_for:
            raise ValueError("provider exploded")
        return {
            "evidence_id": record.get("evidence_id", ""),
            "sentiment": "negative",
            "motivation": "unknown",
            "behaviour": "describes a payment failure",
            "problem_theme": "payment issues",
            "sip_status": "paused",
            "discontinuation_reason": "unknown",
            "supporting_quote": record.get("raw_text", "")[:30],
            "uncertainty": "single comment",
        }


def seed_store(tmp_path, rows):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    for record in rows:
        store.append(record)
    return store


def append_raw_row(store: EvidenceStore, record: dict):
    """Bypass append() validation for fixtures that simulate manual CSV edits."""
    from sip_engine.evidence_store import COLUMNS

    full = {c: record.get(c, "") for c in COLUMNS}
    full.update(record)
    import os as os_module

    with open(store.path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writerow(full)


def make_record(**overrides):
    record = {
        "platform": "youtube",
        "source_url": "https://example.com/watch?v=x",
        "author_handle": "u/first-timer",
        "raw_text": "I paused my SIP after two failed debits last month.",
        "topic_tags": "",
        "ai_theme": "",
        "ai_uncertainty": "",
    }
    record.update(overrides)
    return record


def test_is_analysed_checks_theme_or_uncertainty():
    assert is_analysed({"ai_theme": "", "ai_uncertainty": ""}) is False
    assert is_analysed({"ai_theme": "x", "ai_uncertainty": ""}) is True
    assert is_analysed({"ai_theme": "", "ai_uncertainty": "rule-based mock"}) is True


def test_select_skips_analysed_and_blank_rows(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="fresh1"),
        make_record(evidence_id="already", ai_theme="market volatility"),
    ])
    append_raw_row(store, make_record(evidence_id="fresh2", raw_text="   "))

    selected = select_records(store)

    assert [r["evidence_id"] for r in selected] == ["fresh1"]


def test_select_source_and_limit(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="yt1", platform="youtube"),
        make_record(evidence_id="yt2", platform="youtube"),
        make_record(evidence_id="rd1", platform="reddit"),
    ])

    reddit_only = select_records(store, source="reddit")
    assert [r["evidence_id"] for r in reddit_only] == ["rd1"]

    limited = select_records(store, limit=2)
    assert [r["evidence_id"] for r in limited] == ["yt1", "yt2"]

    zero = select_records(store, limit=0)
    assert zero == []


def test_force_includes_already_analysed(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="done", ai_uncertainty="rule-based mock output"),
        make_record(evidence_id="fresh"),
    ])

    assert [r["evidence_id"] for r in select_records(store)] == ["fresh"]
    assert len(select_records(store, force=True)) == 2


def test_failures_reported_and_never_marked_analysed(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="good-row"),
        make_record(evidence_id="bad-row"),
    ])
    records = select_records(store)
    analyser = FakeAnalyser(fail_for={"bad-row"})

    summary = analyse_records(store, records, analyser)

    assert summary == {"analysed": 1, "failed": 1}
    assert analyser.calls == ["good-row", "bad-row"]

    rows = {r["evidence_id"]: r for r in store.read_all()}
    good = rows["good-row"]
    assert good["ai_theme"] == "payment issues"
    assert good["ai_sentiment"] == "negative"
    assert good["ai_uncertainty"] == "single comment"
    bad = rows["bad-row"]
    assert bad["ai_theme"] == ""
    assert bad["ai_uncertainty"] == ""
    assert bad["raw_text"] == next(r["raw_text"] for r in records if r["evidence_id"] == "bad-row")


def test_main_end_to_end_saves_mock_results(tmp_path, monkeypatch):
    path = seed_store(tmp_path, [
        make_record(evidence_id="m1"),
        make_record(evidence_id="m2"),
    ]).path

    monkeypatch.setattr(analyse_module, "EVIDENCE_PATH", path)
    monkeypatch.setenv("SIP_AI_PROVIDER", "mock")

    exit_code = main(["--limit", "2"])
    out = []

    assert exit_code == 0

    rows = {r["evidence_id"]: r for r in EvidenceStore(path).read_all()}
    assert rows["m1"]["ai_theme"] != ""
    assert rows["m2"]["ai_uncertainty"] != ""
    assert rows["m1"]["raw_text"].endswith("last month.")
    assert rows["m2"]["source_url"] == "https://example.com/watch?v=x"


def test_zero_match_run_reports_zero_and_succeeds(tmp_path, monkeypatch):
    path = seed_store(tmp_path, []).path
    monkeypatch.setattr(analyse_module, "EVIDENCE_PATH", path)
    monkeypatch.setenv("SIP_AI_PROVIDER", "mock")

    assert main([]) == 0


def test_evidence_ids_select_exact_batch_and_nothing_else(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="aaaa1111bbbb2222cccc3333dddd4444"),
        make_record(evidence_id="bbbb2222cccc3333dddd4444eeee5555", raw_text="Second comment for the pilot."),
        make_record(evidence_id="cccc3333dddd4444eeee5555ffff6666"),
    ])
    records = select_records(
        store,
        evidence_ids="AAAA1111,bbbb2222cccc3333dddd4444eeee5555",
    )

    assert [r["evidence_id"][:8] for r in records] == ["aaaa1111", "bbbb2222"]
    assert all(r["raw_text"].strip() for r in records)
    assert records[1]["raw_text"].endswith("pilot.")


def test_evidence_ids_intersects_analysed_skip_and_source(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="done1100", ai_uncertainty="rule-based mock output"),
        make_record(evidence_id="fresh2200", platform="reddit"),
        make_record(evidence_id="fresh3300"),
    ])

    fresh_only = select_records(store, evidence_ids="done1100, fresh2200")
    assert [r["evidence_id"] for r in fresh_only] == ["fresh2200"]

    forced = select_records(store, evidence_ids="done1100, fresh2200", force=True, source="youtube")
    assert [r["evidence_id"] for r in forced] == ["done1100"]


def test_evidence_ids_unknown_prefix_fails_before_any_analysis(tmp_path):
    store = seed_store(tmp_path, [make_record(evidence_id="knownrow")])
    analyser = FakeAnalyser()

    with pytest.raises(ValueError, match="not found"):
        select_records(store, evidence_ids="missing01")

    assert analyser.calls == []


def test_evidence_ids_ambiguous_prefix_fails_before_any_analysis(tmp_path):
    store = seed_store(tmp_path, [
        make_record(evidence_id="dup00001"),
        make_record(evidence_id="dup00002"),
    ])
    analyser = FakeAnalyser()

    with pytest.raises(ValueError, match="ambiguous"):
        select_records(store, evidence_ids="dup0000")

    assert analyser.calls == []


def test_backup_preserves_mock_fields_before_reanalysis(tmp_path):
    from sip_engine.analyse import BACKUP_COLUMNS, backup_existing_analysis

    original_values = {
        "ai_theme": "market volatility",
        "ai_sentiment": "negative",
        "ai_uncertainty": "rule-based mock output; keyword matching on a tiny vocabulary",
    }
    store = seed_store(tmp_path, [
        make_record(evidence_id="mockrow", **original_values),
        make_record(evidence_id="freshrow"),
    ])
    records = select_records(store, force=True)
    backup_path = tmp_path / "analysis_backup.json"

    backed = backup_existing_analysis(records, backup_path)

    assert backed == 1
    payload = __import__("json").loads(backup_path.read_text(encoding="utf-8"))
    entry = payload["records"]["mockrow"]
    assert entry["ai_theme"] == original_values["ai_theme"]
    assert entry["ai_sentiment"] == original_values["ai_sentiment"]
    assert entry["ai_uncertainty"] == original_values["ai_uncertainty"]
    assert set(entry.keys()) == set(BACKUP_COLUMNS)
    assert payload["record_count"] == 1


def test_backup_skips_unanalysed_rows(tmp_path):
    from sip_engine.analyse import backup_existing_analysis

    store = seed_store(tmp_path, [make_record(evidence_id="freshonly")])
    records = select_records(store, force=True)
    backup_path = tmp_path / "analysis_backup.json"

    backed = backup_existing_analysis(records, backup_path)

    assert backed == 0
    payload = __import__("json").loads(backup_path.read_text(encoding="utf-8"))
    assert payload["records"] == {}


def test_backup_merges_without_losing_earlier_entries(tmp_path):
    from sip_engine.analyse import backup_existing_analysis

    store = seed_store(tmp_path, [
        make_record(evidence_id="firstmock", ai_theme="market volatility"),
        make_record(evidence_id="secondmock", ai_uncertainty="rule-based mock output"),
    ])
    records = select_records(store, force=True)
    backup_path = tmp_path / "analysis_backup.json"

    first = backup_existing_analysis(records[:1], backup_path)
    second = backup_existing_analysis(records[1:], backup_path)

    assert (first, second) == (1, 1)
    payload = __import__("json").loads(backup_path.read_text(encoding="utf-8"))
    assert set(payload["records"].keys()) == {"firstmock", "secondmock"}
    assert payload["records"]["firstmock"]["ai_theme"] == "market volatility"
