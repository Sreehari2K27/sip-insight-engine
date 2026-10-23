"""Offline tests for dashboard data helpers (CSV + pandas only, no Streamlit runtime)."""

import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.dashboard_data import (
    OPPORTUNITY_LABELS,
    FINAL_HYPOTHESES,
    OPPORTUNITY_PATTERNS,
    record_analysis_status,
    apply_filters,
    build_new_record,
    dataset_status_counts,
    evidence_log_rows,
    filter_options,
    latest_run_per_source,
    load_collection_log,
    load_evidence,
    opportunity_evidence,
    opportunity_summary,
    record_analysis_status,
    recent_runs,
    source_registry_table,
    summary_stats,
)
from sip_engine.collectors.base import LOG_COLUMNS, SOURCES
from sip_engine.evidence_store import EvidenceStore


def make(**overrides):
    record = {
        "platform": "reddit",
        "source_url": "https://example.com/r/1",
        "author_handle": "u/a",
        "raw_text": "I paused my SIP after the market fell.",
        "topic_tags": "sip;pause",
        "ai_sentiment": "negative",
        "ai_theme": "market volatility",
        "ai_sip_status": "paused",
    }
    record.update(overrides)
    return record


def seed_store(tmp_path, records):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    for record in records:
        store.append(record)
    return path


def test_load_evidence_empty_csv_has_schema_columns(tmp_path):
    path = seed_store(tmp_path, [])

    df = load_evidence(path)

    assert df.empty
    assert list(df.columns)[0] == "evidence_id"
    assert "raw_text" in df.columns
    assert "ai_sip_status" in df.columns


def test_summary_stats_counts_sources_and_status(tmp_path):
    path = seed_store(
        tmp_path,
        [
            make(),
            make(platform="twitter", ai_sip_status="stopped"),
            make(
                raw_text="Good discipline so far.",
                ai_sentiment="positive",
                ai_theme="unknown",
                ai_sip_status="active",
            ),
        ],
    )

    stats = summary_stats(load_evidence(path))

    assert stats["total"] == 3
    assert stats["sources"] == 2
    assert stats["analysed"] == 3
    assert stats["interrupted"] == 2
    assert stats["by_source"]["reddit"] == 2
    assert stats["top_theme"] == "market volatility"


def test_summary_stats_without_any_analysis(tmp_path):
    path = seed_store(tmp_path, [make(ai_sentiment="", ai_theme="", ai_sip_status="")])

    stats = summary_stats(load_evidence(path))

    assert stats["analysed"] == 0
    assert stats["interrupted"] == 0
    assert stats["top_theme"] == ""
    assert stats["by_theme"].empty


def test_filter_options_exclude_blank_values(tmp_path):
    path = seed_store(tmp_path, [make(), make(platform="twitter", ai_sentiment="")])

    df = load_evidence(path)

    assert filter_options(df, "source") == ["reddit", "twitter"]
    assert filter_options(df, "sentiment") == ["negative"]
    assert filter_options(df, "problem_theme") == ["market volatility"]
    assert filter_options(df, "sip_status") == ["paused"]


def test_apply_filters_by_each_dimension(tmp_path):
    path = seed_store(
        tmp_path,
        [
            make(),
            make(platform="twitter"),
            make(
                raw_text="Fees feel high.",
                ai_sentiment="",
                ai_theme="",
                ai_sip_status="",
            ),
        ],
    )
    df = load_evidence(path)

    assert len(apply_filters(df, source="reddit")) == 2
    assert len(apply_filters(df, sentiment="negative")) == 2
    assert len(apply_filters(df, problem_theme="market volatility")) == 2
    assert len(apply_filters(df, sip_status="paused")) == 2
    assert len(apply_filters(df, search="FEES")) == 1
    assert len(apply_filters(df, source="reddit", search="market fell")) == 1
    assert len(apply_filters(df, sentiment="positive")) == 0


def test_unicode_text_survives_loading(tmp_path):
    path = seed_store(tmp_path, [make(raw_text="बस रुक गया — ₹5,000 felt heavy.")])

    df = load_evidence(path)

    assert df.iloc[0]["raw_text"] == "बस रुक गया — ₹5,000 felt heavy."


def test_filters_never_mutate_source_dataframe(tmp_path):
    path = seed_store(tmp_path, [make(), make(platform="twitter")])
    df = load_evidence(path)
    before = len(df)

    apply_filters(df, source="twitter", search="paused")

    assert len(df) == before


def test_dashboard_shows_friendly_error_on_schema_mismatch(tmp_path, monkeypatch):
    import app.app as app_module
    import app.dashboard_data as dashboard_data_module
    from streamlit.testing.v1 import AppTest

    def broken_load(path):
        raise ValueError(
            "Incompatible evidence CSV header: simulated stale schema; "
            "the CSV contains fields newer than this running process supports. "
            "Restart the Streamlit server or update the running code, then reload."
        )

    monkeypatch.setattr(dashboard_data_module, "load_evidence", broken_load)

    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=15)

    assert at.exception == []
    assert len(at.error) == 1
    assert "restart" in at.error[0].value.lower()
    assert "evidence file" in at.error[0].value.lower()


def write_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_raw_row(store: EvidenceStore, record: dict):
    """Bypass append() validation for fixtures simulating manual/AI-written CSV."""
    from sip_engine.evidence_store import COLUMNS

    write_header = not store.path.exists() or store.path.stat().st_size == 0
    full = {c: record.get(c, "") for c in COLUMNS}
    full.update(record)
    with open(store.path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(full)


def test_load_collection_log_missing_or_empty_file_returns_empty_frame(tmp_path):
    missing = load_collection_log(tmp_path / "nope.csv")

    assert missing.empty
    assert list(missing.columns) == list(LOG_COLUMNS)

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")

    assert load_collection_log(empty).empty


def test_load_collection_log_normalises_rows_and_coerces_numbers(tmp_path):
    log = tmp_path / "collection_log.csv"
    with open(log, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(LOG_COLUMNS) + ["unexpected_extra"])
        writer.writerow([
            "r1", "2026-10-20T09:00:00+00:00", "appstore", "ok", "50", "1", "49", "", "ignored"
        ])
        writer.writerow([
            "r2", "2026-10-20T10:00:00+00:00", "appstore", "ok", "oops", "0", "0", "note: diagnostic"
        ])
        writer.writerow(["r3", "", "reddit", "failed"])

    df = load_collection_log(log)

    assert list(df.columns) == list(LOG_COLUMNS)
    assert df.iloc[0]["items_found"] == 50
    assert df.iloc[0]["error"] == ""
    assert pd.isna(df.iloc[1]["items_found"])
    assert df.iloc[1]["error"] == "note: diagnostic"
    assert df.iloc[2]["run_id"] == "r3"
    assert df.iloc[2]["collected_at"] == ""
    assert df.iloc[2]["status"] == "failed"


def test_latest_run_per_source_and_recent_runs_ordering(tmp_path):
    log = tmp_path / "collection_log.csv"
    write_log(log, [
        {"run_id": "a1", "collected_at": "2026-10-20T09:00:00+00:00", "source": "appstore",
         "status": "ok", "items_found": "50", "items_saved": "1", "items_skipped": "49", "error": ""},
        {"run_id": "a2", "collected_at": "2026-10-20T10:00:00+00:00", "source": "appstore",
         "status": "ok", "items_found": "60", "items_saved": "0", "items_skipped": "60", "error": ""},
        {"run_id": "t1", "collected_at": "2026-10-20T08:00:00+00:00", "source": "tradingqna",
         "status": "ok", "items_found": "5", "items_saved": "2", "items_skipped": "3", "error": ""},
        {"run_id": "x1", "collected_at": "", "source": "reddit",
         "status": "failed", "items_found": "", "items_saved": "", "items_skipped": "", "error": "e"},
    ])

    df = load_collection_log(log)

    latest = latest_run_per_source(df)
    by_source = dict(zip(latest["source"], latest["run_id"]))
    assert by_source == {"appstore": "a2", "tradingqna": "t1", "reddit": "x1"}
    assert list(latest["source"]) == ["appstore", "tradingqna", "reddit"]

    recent = recent_runs(df)
    assert list(recent["run_id"]) == ["a2", "a1", "t1", "x1"]
    assert list(recent_runs(df, limit=2)["run_id"]) == ["a2", "a1"]


def test_source_registry_table_reflects_registry():
    table = source_registry_table()

    assert list(table["source"]) == sorted(SOURCES.keys())
    tradingqna = table[table["source"] == "tradingqna"].iloc[0]
    assert tradingqna["status"] == "approved"
    assert tradingqna["adapter"] == "yes"
    appstore = table[table["source"] == "appstore"].iloc[0]
    assert appstore["status"] == "approved" and appstore["adapter"] == "yes"


def test_collection_health_renders_and_writes_nothing(tmp_path, monkeypatch):
    import app.app as app_module
    import app.dashboard_data as dashboard_data_module
    from streamlit.testing.v1 import AppTest

    synthetic = pd.DataFrame(
        [
            ["a1", "2026-10-20T09:00:00+00:00", "appstore", "ok", 50, 1, 49, ""],
            ["a2", "2026-10-20T10:00:00+00:00", "appstore", "ok", 50, 0, 50,
             "note: 1404871703 page 1: feed ok but no entries"],
        ],
        columns=list(LOG_COLUMNS),
    )
    monkeypatch.setattr(dashboard_data_module, "load_collection_log", lambda path: synthetic)

    real_log = Path(__file__).resolve().parents[1] / "data" / "collection_log.csv"
    real_log_before = real_log.read_bytes()

    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=20)

    assert at.exception == []
    assert any("Collection health" in s.value for s in at.subheader)
    assert real_log.read_bytes() == real_log_before


def test_collection_health_empty_state(monkeypatch):
    import app.dashboard_data as dashboard_data_module
    from streamlit.testing.v1 import AppTest

    empty = pd.DataFrame(columns=list(LOG_COLUMNS))
    monkeypatch.setattr(dashboard_data_module, "load_collection_log", lambda path: empty)

    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=20)

    assert at.exception == []
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "No collection runs recorded yet" in markdown_text


def test_dashboard_data_imports_no_network_or_llm_modules():
    code = (
        "import sys\n"
        "from app import dashboard_data\n"
        "loaded = {m for m in sys.modules if m.split('.')[0] in "
        "{'httpx', 'httpx2', 'requests', 'mistralai', 'google', 'urllib3'}}\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr


def test_analysis_status_classifies_mock_live_unanalysed():
    mock_signature = (
        "rule-based mock output; keyword matching on a tiny vocabulary, "
        "not model-generated"
    )
    assert record_analysis_status({"ai_uncertainty": mock_signature}) == "mock"
    assert record_analysis_status({"ai_uncertainty": "A real model wrote this note."}) == "live"
    assert record_analysis_status({"ai_uncertainty": ""}) == "unanalysed"


def test_dataset_status_counts_and_analysis_status_filter(tmp_path):
    from app.dashboard_data import _MOCK_ANALYSIS_SIGNATURE as mock_signature

    path = (opportunity_test_store(tmp_path)).path
    store = EvidenceStore(path)
    append_raw_row(store, dict(
        make(evidence_id="mockrow", raw_text="Groww par auto Prime lag gaya, switch kaise kare?"),
        ai_supporting_quote="auto Prime lag gaya", ai_uncertainty=mock_signature,
    ))
    append_raw_row(store, dict(
        make(evidence_id="liverow", raw_text="Groww regular plan switch confusion row."),
        ai_theme="plan confusion", ai_uncertainty="It is uncertain whether the patient switched.",
    ))

    df = load_evidence(store.path)
    counts = dataset_status_counts(df)
    assert counts == {"live": 1, "mock": 1, "unanalysed": 4}

    mock_rows = apply_filters(df, analysis_status="mock")
    assert list(mock_rows["evidence_id"]) == ["mockrow"]
    live_rows = apply_filters(df, analysis_status="live")
    assert list(live_rows["evidence_id"]) == ["liverow"]
    unana = apply_filters(df, analysis_status="unanalysed")
    assert "plain" in list(unana["evidence_id"]) and "liverow" not in list(unana["evidence_id"])
    assert len(apply_filters(df, analysis_status="Any")) == len(df)


def test_opportunity_summary_counts_ai_validated_not_mock(tmp_path):
    path = (opportunity_test_store(tmp_path)).path
    store = EvidenceStore(path)
    append_raw_row(store, dict(
        make(evidence_id="mockprime", raw_text="Groww par auto Prime lag gaya, switch kaise kare?"),
        ai_theme="plan confusion", ai_supporting_quote="auto Prime lag gaya",
        ai_uncertainty="rule-based mock output",
    ))

    summaries = opportunity_summary(load_evidence(store.path))
    prime_label = dict(FINAL_HYPOTHESES)["prime_or_plan_confusion"]
    prime = next(s for s in summaries if s["opportunity"] == prime_label)
    assert prime["evidence_strength"] == 2  # op1 + mockprime both match the window
    assert prime["ai_validated_rows"] == 1  # only the live op1 row counts as validated


AUDIT_FP_FOOD_SAFETY = (
    "Where is India's crackdown on food safety headed? Our goal with The Daily Brief "
    "is to simplify the biggest stories. A mandate failed and a SIP was paused; "
    "step up costs and charges could reduce it; the amount was credited to a bank account."
)
AUDIT_FP_KITE_ANNOUNCEMENT = (
    "US Stocks & ETFs on Kite: Beta is live. One of our most-requested features is here. "
    "You can now invest in US stocks instead of a regular fund; low fees and no charges; "
    "amount credited to your bank account after a refund."
)


def _seed_one(tmp_path, evidence_id, text):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    store.append(make(evidence_id=evidence_id, raw_text=text,
                      ai_theme="", ai_sentiment="", ai_sip_status=""))
    return store


def test_food_safety_announcement_matches_no_window(tmp_path):
    store = _seed_one(tmp_path, "fp1", AUDIT_FP_FOOD_SAFETY)
    evidence_map = opportunity_evidence(load_evidence(store.path))
    assert all(not group["supporting"] and not group["related_exploratory"]
               for group in evidence_map.values())


def test_kite_announcement_matches_no_window(tmp_path):
    store = _seed_one(tmp_path, "fp2", AUDIT_FP_KITE_ANNOUNCEMENT)
    evidence_map = opportunity_evidence(load_evidence(store.path))
    assert all(not group["supporting"] and not group["related_exploratory"]
               for group in evidence_map.values())


def test_future_tense_deduction_question_is_exploratory(tmp_path):
    store = _seed_one(
        tmp_path, "fp3",
        "kisi bhi app me sip amount and set up feed karne ke baad kya app automatic amount deduct kaega bank se ?",
    )
    evidence_map = opportunity_evidence(load_evidence(store.path))
    payment = evidence_map["failed_payment_resolution"]
    assert payment["supporting"] == []
    assert [r["evidence_id"] for r in payment["related_exploratory"]] == ["fp3"]


def test_past_tense_deduction_incident_remains_supporting(tmp_path):
    store = _seed_one(
        tmp_path, "f6b",
        "Sir mine  one sip missed from bank account deducted 474 rupees  please advise me what to do",
    )
    evidence_map = opportunity_evidence(load_evidence(store.path))
    payment = evidence_map["failed_payment_resolution"]
    assert [r["evidence_id"] for r in payment["supporting"]] == ["f6b"]


def test_existing_valid_examples_still_match_their_windows(tmp_path):
    store = _seed_one(
        tmp_path, "p1",
        "Bhai meine kal hi investment shuru kari groww par, automatic Prime hi laga tha, "
        "3 funds liye teeno regular hogai, isko direct mein kaise switch kare?",
    )
    store.append(make(
        evidence_id="p2",
        raw_text="sir kya hum stepup sip ko rok bhi sakte hai mujhe abhi aur nhi krna",
    ))

    evidence_map = opportunity_evidence(load_evidence(store.path))
    assert [r["evidence_id"] for r in evidence_map["prime_or_plan_confusion"]["supporting"]] == ["p1"]
    assert [r["evidence_id"] for r in evidence_map["pause_stop_stepup"]["supporting"]] == ["p2"]


def opportunity_test_store(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    blank_ai = dict(ai_theme="", ai_sentiment="", ai_sip_status="")

    store.append(make(
        evidence_id="op1", **blank_ai,
        raw_text="Groww par auto Prime lag gaya, teeno regular plan funds chal gaye, switch kaise kare.",
    ))
    store.append(make(
        evidence_id="op2", **blank_ai,
        raw_text="Sir mine one sip missed from bank account deducted 474 rupees please advise me what to do",
    ))
    store.append(make(
        evidence_id="op3", **blank_ai,
        raw_text="Kya hum apna stepup SIP rok sakte hai agar amount change karna ho?",
    ))
    store.append(make(
        evidence_id="plain", **blank_ai,
        raw_text="Beautiful app design, love using it every day!",
    ))
    return store


def test_opportunity_counts_compute_from_rows_and_reject_false_positives(tmp_path):
    evidence_map = opportunity_evidence(load_evidence(opportunity_test_store(tmp_path).path))

    assert [r["evidence_id"] for r in evidence_map["prime_or_plan_confusion"]["supporting"]] == ["op1"]
    assert [r["evidence_id"] for r in evidence_map["failed_payment_resolution"]["supporting"]] == ["op2"]
    assert [r["evidence_id"] for r in evidence_map["pause_stop_stepup"]["supporting"]] == ["op3"]
    summaries = opportunity_summary(load_evidence(opportunity_test_store(tmp_path).path))
    by_label = {s["opportunity"]: s for s in summaries}
    prime_label = dict(FINAL_HYPOTHESES)["prime_or_plan_confusion"]
    assert by_label[prime_label]["evidence_strength"] == 1
    assert by_label[prime_label]["confidence"] == "exploratory (sparse support)"
    assert "unverified" not in str(summaries)


def test_overlapping_row_counts_in_both_groups(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    store.append(make(
        evidence_id="ovl",
        raw_text="Regular plan fees badh gaye isliye maine SIP rok diya, switch karna chahta hu.",
    ))

    evidence_map = opportunity_evidence(load_evidence(path))

    assert [r["evidence_id"] for r in evidence_map["prime_or_plan_confusion"]["supporting"]] == ["ovl"]
    assert [r["evidence_id"] for r in evidence_map["pause_stop_stepup"]["supporting"]] == ["ovl"]


def test_counts_recalculate_when_rows_change(tmp_path):
    store = opportunity_test_store(tmp_path)
    df = load_evidence(store.path)
    before = {s["opportunity"]: s["evidence_strength"] for s in opportunity_summary(df)}

    store.append(make(
        evidence_id="op5",
        raw_text="Groww se SIP autopay mandate fail ho gaya, dobara payment deduct ho gaya.",
    ))
    df2 = load_evidence(store.path)
    after = {s["opportunity"]: s["evidence_strength"] for s in opportunity_summary(df2)}

    assert after[OPPORTUNITY_LABELS["failed_payment_resolution"]] == before[OPPORTUNITY_LABELS["failed_payment_resolution"]] + 1


def test_evidence_log_labels_unverified_until_ai_analysis(tmp_path):
    store = opportunity_test_store(tmp_path)
    df = load_evidence(store.path)

    log_rows = evidence_log_rows(df)
    assert len(log_rows) == 3
    assert all(r["validation_status"] == "unverified — no AI analysis yet" for r in log_rows)

    updated = df.copy()
    row_mask = updated["evidence_id"] == "op1"
    updated.loc[row_mask, "ai_theme"] = "plan confusion"
    updated.loc[row_mask, "ai_supporting_quote"] = "auto Prime lag gaya"
    updated.loc[row_mask, "ai_uncertainty"] = "single comment"

    log_after = evidence_log_rows(updated)
    op1 = next(r for r in log_after if r["evidence_id"] == "op1")
    assert op1["validation_status"] == "ai-validated"
    assert op1["assigned_theme"] == "plan confusion"
    assert op1["uncertainty"] == "single comment"
    others = [r for r in log_after if r["evidence_id"] != "op1"]
    assert all(r["validation_status"].startswith("unverified") for r in others)


def test_select_supporting_example_relevance_rules(tmp_path):
    from app.dashboard_data import select_supporting_example

    path = (opportunity_test_store(tmp_path)).path
    store = EvidenceStore(path)
    mock_signature = "rule-based mock output; keyword matching on a tiny vocabulary, not model-generated"
    append_raw_row(store, dict(
        make(evidence_id="quoted", raw_text="I switched my funds to direct after Prime surprised me."),
        ai_motivation="switch to direct funds", ai_supporting_quote="I switched my funds to direct",
        ai_uncertainty="It is uncertain whether the switch completed.",
    ))
    append_raw_row(store, dict(
        make(evidence_id="noquote", raw_text="Plan confusion row with no stored quote."),
        ai_motivation="unknown", ai_supporting_quote="unknown",
        ai_uncertainty="A real model wrote this note.",
    ))
    df = load_evidence(store.path)

    chosen = select_supporting_example(df, "ai_motivation")
    assert chosen is not None and chosen["evidence_id"].startswith("quoted")

    none_fields = select_supporting_example(df, "ai_theme")
    assert none_fields is None or record_analysis_status(none_fields) == "live"


def test_select_supporting_example_returns_none_without_suitable_evidence(tmp_path):
    from app.dashboard_data import select_supporting_example

    path = (opportunity_test_store(tmp_path)).path
    store = EvidenceStore(path)
    mock_signature = "rule-based mock output; keyword matching on a tiny vocabulary, not model-generated"
    append_raw_row(store, dict(
        make(evidence_id="opinrow", raw_text="Opinion only row."),
        ai_behaviour="shares an opinion without stating a SIP action",
        ai_uncertainty="A real model wrote this note.",
    ))
    append_raw_row(store, dict(
        make(evidence_id="mockrow", raw_text="Mock row with quote."),
        ai_motivation="switching funds", ai_supporting_quote="Mock row with quote.",
        ai_uncertainty=mock_signature,
    ))
    df = load_evidence(store.path)

    assert select_supporting_example(df, "ai_behaviour") is None
    assert select_supporting_example(df, "ai_motivation") is None


def test_hypothesis_cards_render_in_rank_order():
    from streamlit.testing.v1 import AppTest
    import re as re_module

    app_path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=25)

    assert at.exception == []

    badge_numbers = []
    for element in at.markdown:
        value = element.value
        if 'class="rank-badge"' in str(value):
            found = re_module.search(r'rank-badge">\s*([123])\s*<', str(value))
            if found:
                badge_numbers.append(int(found.group(1)))

    assert len(badge_numbers) == 3, f"expected 3 ranked cards, saw {badge_numbers}"
    assert badge_numbers == [1, 2, 3], f"cards out of rank order: {badge_numbers}"


def test_mock_signed_row_never_validated_even_with_quote(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    mock_signature = (
        "rule-based mock output; keyword matching on a tiny vocabulary, "
        "not model-generated"
    )
    rows_payload = [
        dict(
            make(
                evidence_id="mockrow",
                raw_text="Automatic Prime lag gaya aur teeno regular plan funds chal gaye.",
            ),
            ai_theme="plan confusion",
            ai_supporting_quote="auto Prime lag gaya",
            ai_uncertainty=mock_signature,
        ),
        dict(
            make(
                evidence_id="liverow",
                raw_text="Automatic Prime lag gaya aur teeno regular plan funds chal gaye.",
            ),
            ai_theme="plan confusion",
            ai_supporting_quote="auto Prime lag gaya",
            ai_uncertainty="It is uncertain whether the user later switched the funds.",
        ),
    ]
    for row in rows_payload:
        append_raw_row(store, row)

    log = {r["evidence_id"]: r for r in evidence_log_rows(load_evidence(store.path))}

    assert log["mockrow"]["validation_status"] == "unverified — mock placeholder output"
    assert log["liverow"]["validation_status"] == "ai-validated"


def test_mandate_only_matches_are_not_counted_as_failed_payments(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    blank_ai = dict(ai_theme="", ai_sentiment="", ai_sip_status="")
    store.append(make(
        evidence_id="mandate_only", **blank_ai,
        raw_text="Groww me SIP autopay mandate setup kaise kare? please bata do",
    ))
    store.append(make(
        evidence_id="real_failure", **blank_ai,
        raw_text="SIP missed from my bank account, 474 rupees deducted, what can I do",
    ))

    evidence_map = opportunity_evidence(load_evidence(path))
    group = evidence_map["failed_payment_resolution"]

    assert [r["evidence_id"] for r in group["supporting"]] == ["real_failure"]
    assert [r["evidence_id"] for r in group["related_exploratory"]] == ["mandate_only"]

    summary = {s["opportunity"]: s for s in opportunity_summary(load_evidence(path))}
    payment = summary["Failed SIP payment and unclear resolution"]
    assert payment["evidence_strength"] == 1
    assert payment["confidence"] == "exploratory (sparse support)"
    assert payment["related_process_confusion"] == 1


def test_vague_payment_word_rows_are_excluded_from_count(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    store.append(make(
        evidence_id="vague_pay", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="One time payment per sip.....",
    ))

    evidence_map = opportunity_evidence(load_evidence(path))
    group = evidence_map["failed_payment_resolution"]

    assert group["supporting"] == []
    assert [r["evidence_id"] for r in group["related_exploratory"]] == ["vague_pay"]


def test_group_one_drops_direct_share_sense_errors(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    store.append(make(
        evidence_id="sense_error", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Bhai direct share khareedo, sip bekar hai muze mat batao",
    ))

    evidence_map = opportunity_evidence(load_evidence(path))

    assert evidence_map["prime_or_plan_confusion"]["supporting"] == []
    assert evidence_map["pause_stop_stepup"]["supporting"] == []
    assert evidence_map["prime_or_plan_confusion"]["related_exploratory"] == []


def test_trading_in_options_spam_is_vetoed(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    store.append(make(
        evidence_id="spam", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Stop trading in options and mutual funds either buying or selling, you will loose money.",
    ))

    evidence_map = opportunity_evidence(load_evidence(path))

    assert all(not group["supporting"] and not group["related_exploratory"] for group in evidence_map.values())

    store.append(make(
        evidence_id="genuine_pause", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Kya hum stepup SIP rok sakte hai agar amount change karna ho?",
    ))
    evidence_map2 = opportunity_evidence(load_evidence(path))
    assert [r["evidence_id"] for r in evidence_map2["pause_stop_stepup"]["supporting"]] == ["genuine_pause"]


def test_opportunity_filter_selects_and_excludes(tmp_path):
    store = opportunity_test_store(tmp_path)
    store.append(make(
        evidence_id="spam", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Stop trading in options and mutual funds either buying or selling.",
    ))
    df = load_evidence(store.path)

    prime = apply_filters(df, opportunity=OPPORTUNITY_LABELS["prime_or_plan_confusion"])
    assert list(prime["evidence_id"]) == ["op1"]
    payment = apply_filters(df, opportunity=OPPORTUNITY_LABELS["failed_payment_resolution"])
    assert list(payment["evidence_id"]) == ["op2"]
    pause = apply_filters(df, opportunity=OPPORTUNITY_LABELS["pause_stop_stepup"])
    assert list(pause["evidence_id"]) == ["op3"]
    assert apply_filters(df, opportunity="All") is not None
    assert len(apply_filters(df, opportunity="All")) == len(df)

    with pytest.raises(ValueError, match="Unknown opportunity"):
        apply_filters(df, opportunity="made-up label")


def test_ai_filters_stay_inert_until_analysis_exists(tmp_path):
    store = opportunity_test_store(tmp_path)
    df = load_evidence(store.path)

    assert filter_options(df, "sentiment") == []
    assert filter_options(df, "problem_theme") == []
    assert filter_options(df, "sip_status") == []

    backfilled = df.copy()
    backfilled.loc[backfilled["evidence_id"] == "op1", "ai_sentiment"] = "negative"

    assert filter_options(backfilled, "sentiment") == ["negative"]
    assert list(apply_filters(backfilled, sentiment="negative")["evidence_id"]) == ["op1"]


def test_combined_search_source_and_opportunity_filters(tmp_path):
    store = opportunity_test_store(tmp_path)
    df = load_evidence(store.path)
    prime = OPPORTUNITY_LABELS["prime_or_plan_confusion"]

    combined = apply_filters(df, source="reddit", opportunity=prime, search="Prime")
    assert list(combined["evidence_id"]) == ["op1"]

    only_opportunity = apply_filters(df, opportunity=prime)
    assert list(only_opportunity["evidence_id"]) == ["op1"]

    none_passing = apply_filters(df, source="youtube", opportunity=prime, search="zzz-not-there")
    assert none_passing.empty


def test_published_since_date_filter(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    store.append(make(
        evidence_id="recent", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Groww regular plan switch describe row.",
        source_url="https://example.com/@",
        published_at="2026-10-15T09:00:00Z",
    ))
    store.append(make(
        evidence_id="older", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Groww regular plan switch earlier row.",
        published_at="2026-09-01T09:00:00Z",
    ))
    store.append(make(
        evidence_id="blank_date", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Groww regular plan switch undated row.",
        published_at="",
    ))
    store.append(make(
        evidence_id="bad_date", ai_theme="", ai_sentiment="", ai_sip_status="",
        raw_text="Groww regular plan switch malformed date row.",
        published_at="not-a-date-value",
    ))
    df = load_evidence(store.path)

    assert apply_filters(df)["evidence_id"].tolist() == ["recent", "older", "blank_date", "bad_date"]

    since = date(2026, 10, 1)
    active = apply_filters(df, published_since=since)
    assert list(active["evidence_id"]) == ["recent"]

    since_earlier = apply_filters(df, published_since=date(2026, 8, 1))
    assert list(since_earlier["evidence_id"]) == ["recent", "older"]

    since_end_of_time = apply_filters(df, published_since=date(2027, 1, 1))
    assert since_end_of_time.empty


def test_zero_match_state_is_representable(tmp_path):
    store = opportunity_test_store(tmp_path)
    df = load_evidence(store.path)

    zero = apply_filters(df, search="zzz-not-in-any-comment")
    assert len(zero) == 0
    summaries = opportunity_summary(load_evidence(store.path))
    assert len(summaries) == 4  # three final hypotheses + the demoted payment thread


def test_build_new_record_valid_submission():
    from sip_engine.collectors.base import CollectedItem

    expected_key = CollectedItem(
        platform="reddit", source_url="https://example.com/r/9", raw_text=""
    ).compute_source_key()
    record = build_new_record(
        "  reddit  ", " https://example.com/r/9 ", date(2026, 10, 20), "  I paused my SIP.  "
    )

    assert record == {
        "platform": "reddit",
        "source_url": "https://example.com/r/9",
        "captured_date": "2026-10-20",
        "raw_text": "I paused my SIP.",
        "source_key": expected_key,
    }
    assert len(record["source_key"]) == 64


def test_build_new_record_source_key_is_deterministic_and_url_distinct():
    first = build_new_record("reddit", " https://example.com/r/9 ", None, "text")
    again = build_new_record("reddit", "https://example.com/r/9", None, "text")
    other_url = build_new_record("reddit", "https://example.com/r/10", None, "text")
    other_platform = build_new_record("youtube", "https://example.com/r/9", None, "text")

    assert first["source_key"] == again["source_key"]
    assert first["source_key"] != other_url["source_key"]
    assert first["source_key"] != other_platform["source_key"]


def test_form_duplicate_precheck_round_trip(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    record = build_new_record(
        "reddit",
        "https://www.reddit.com/r/IndiaInvestments/comments/xyz/sip_paused/abc/",
        date(2026, 10, 21),
        "I paused; SIP band kar diya after the market crash.",
    )

    store.append(record)
    assert store.has_source_key(record["source_key"]) is True

    other_comment = build_new_record(
        "reddit",
        "https://www.reddit.com/r/IndiaInvestments/comments/xyz/sip_paused/def/",
        None,
        "Different commenter, also SIP related feedback text.",
    )
    assert store.has_source_key(other_comment["source_key"]) is False
    assert store.count() == 1


def test_build_new_record_missing_required_fields_raise():
    with pytest.raises(ValueError, match="Source URL is required"):
        build_new_record("reddit", "   ", None, "some feedback")
    with pytest.raises(ValueError, match="Raw feedback is required"):
        build_new_record("reddit", "https://example.com/r/9", None, "   ")
    with pytest.raises(ValueError) as both:
        build_new_record("reddit", "", None, "")
    assert "Source URL is required" in str(both.value)
    assert "Raw feedback is required" in str(both.value)


def test_build_new_record_rejects_non_http_url():
    with pytest.raises(ValueError, match="http"):
        build_new_record("reddit", "ftp://example.com/thread", None, "feedback")
    with pytest.raises(ValueError, match="http"):
        build_new_record("reddit", "javascript:alert(1)", None, "feedback")


def test_form_submission_round_trip_and_unique_ids(tmp_path):
    path = tmp_path / "evidence.csv"
    store = EvidenceStore(path)
    record = build_new_record(
        "reddit", "https://example.com/r/9", date(2026, 10, 20), "बस रुक गया — ₹5,000, felt \"heavy\"."
    )

    first = store.append(record)
    second = store.append(build_new_record("twitter", "https://example.com/x/1", None, "Same reason again."))

    rows = store.read_all()
    assert len(rows) == 2
    assert len({rows[0]["evidence_id"], rows[1]["evidence_id"]}) == 2
    assert rows[0]["raw_text"] == "बस रुक गया — ₹5,000, felt \"heavy\"."
    assert rows[0]["captured_date"] == "2026-10-20"
    assert rows[1]["captured_date"] == date.today().isoformat()
    assert rows[0]["ai_sentiment"] == ""
    assert df_round_trip_ok(path)


def df_round_trip_ok(path):
    df = load_evidence(path)
    return list(df.columns)[0] == "evidence_id" and len(df) == 2
