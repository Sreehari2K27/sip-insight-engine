"""Offline tests for the mock AI client. No network, no API spend."""

import pytest

from sip_engine.ai_client import (
    ANALYSIS_COLUMNS,
    MockAI,
    validate_analysis,
)
from sip_engine.evidence_store import ANALYSIS_TO_CSV, COLUMNS, EvidenceStore

PAUSED_TEXT = (
    "Markets kept falling and I got worried, so I paused my SIP last month. "
    "I invest \u20b95,000 every month usually."
)
ACTIVE_TEXT = "I lost my job last month, but I still continue my SIP every month."
NEUTRAL_TEXT = "I have been reading about mutual funds this week."


def make_record(**overrides):
    record = {
        "evidence_id": "ev001",
        "source_url": "https://example.com/thread/42",
        "author_handle": "u/investor123",
        "raw_text": PAUSED_TEXT,
        "topic_tags": "sip;pause",
    }
    record.update(overrides)
    return record


def test_paused_evidence_gets_full_structured_analysis():
    result = MockAI().analyse(make_record())

    assert set(result) == set(ANALYSIS_COLUMNS)
    assert result["evidence_id"] == "ev001"
    assert result["sip_status"] == "paused"
    assert result["sentiment"] == "negative"
    assert result["problem_theme"] == "market volatility"
    assert result["discontinuation_reason"] == "market losses"
    assert "paused" in result["supporting_quote"]
    assert "not model-generated" in result["uncertainty"]


def test_unsupported_claims_default_to_unknown():
    result = MockAI().analyse(make_record(raw_text=NEUTRAL_TEXT))

    assert result["sip_status"] == "unknown"
    assert result["sentiment"] == "neutral"
    assert result["motivation"] == "unknown"
    assert result["problem_theme"] == "unknown"
    assert result["discontinuation_reason"] == "unknown"
    assert result["supporting_quote"] == "unknown"
    assert "no explicit supporting sentence found" in result["uncertainty"]


def test_active_investor_gets_no_discontinuation_reason():
    result = MockAI().analyse(make_record(raw_text=ACTIVE_TEXT))

    assert result["sip_status"] == "active"
    assert result["discontinuation_reason"] == "unknown"
    assert result["problem_theme"] == "cash-flow strain"
    assert result["behaviour"] == "describes continuing the SIP"


def test_analysis_never_touches_original_evidence(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    stored = store.append(make_record())
    snapshot = dict(stored)

    result = MockAI().analyse(stored)

    assert stored == snapshot
    assert "raw_text" not in result
    assert "source_url" not in result

    rows = store.read_all()
    assert rows[0]["raw_text"] == snapshot["raw_text"]
    assert rows[0]["source_url"] == snapshot["source_url"]
    assert rows[0]["ai_summary"] == ""
    assert rows[0]["ai_theme"] == ""
    assert rows[0]["confidence"] == ""


def test_analysis_is_deterministic():
    record = make_record()

    assert MockAI().analyse(record) == MockAI().analyse(record)


def test_validate_analysis_rejects_bad_output():
    valid = MockAI().analyse(make_record())
    assert validate_analysis(valid) == valid

    with pytest.raises(ValueError):
        validate_analysis(dict(valid, sentiment="angry"))
    with pytest.raises(ValueError):
        validate_analysis(dict(valid, sip_status="sometimes"))
    incomplete = dict(valid)
    del incomplete["uncertainty"]
    with pytest.raises(ValueError):
        validate_analysis(incomplete)


def test_missing_required_evidence_raises():
    with pytest.raises(ValueError):
        MockAI().analyse({"source_url": "https://example.com/thread/1"})
    with pytest.raises(ValueError):
        MockAI().analyse({})


def test_mock_output_fits_csv_analysis_schema():
    result = MockAI().analyse(make_record())

    assert set(ANALYSIS_TO_CSV) < set(ANALYSIS_COLUMNS)
    assert len(set(ANALYSIS_TO_CSV.values())) == len(ANALYSIS_TO_CSV)
    assert set(ANALYSIS_TO_CSV.values()) <= set(COLUMNS)
    for key in ANALYSIS_TO_CSV:
        assert isinstance(result[key], str)
