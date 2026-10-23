"""Offline tests for the Mistral-backed client using an injected fake client.

No network, no API spend, and no mistralai package required.
"""

import json

import pytest

from sip_engine.ai_client import (
    ANALYSIS_COLUMNS,
    DEFAULT_MODEL,
    MistralAI,
    MockAI,
    SYSTEM_PROMPT,
    create_analyser,
    validate_analysis,
)

PAUSED_TEXT = (
    "Markets kept falling and I got worried, so I paused my SIP last month. "
    "I invest \u20b95,000 every month usually."
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeMistralClient:
    def __init__(self, content=""):
        self.content = content
        self.requests = []
        self.chat = self

    def complete(self, **kwargs):
        self.requests.append(kwargs)
        return FakeResponse(self.content)


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


def valid_payload(**overrides):
    payload = {
        "evidence_id": "ev001",
        "sentiment": "negative",
        "motivation": "unknown",
        "behaviour": "describes pausing the SIP",
        "problem_theme": "market volatility",
        "sip_status": "paused",
        "discontinuation_reason": "market losses",
        "supporting_quote": "so I paused my SIP last month",
        "uncertainty": "single comment, small sample",
    }
    payload.update(overrides)
    return payload


def make_analyser(content):
    fake = FakeMistralClient(content)
    return MistralAI(api_key="test-key-not-real", client=fake), fake


def test_valid_output_passes_schema_and_single_call():
    analyser, fake = make_analyser(json.dumps(valid_payload()))

    result = analyser.analyse(make_record())

    assert set(result) == set(ANALYSIS_COLUMNS)
    assert validate_analysis(result) == result
    assert result["evidence_id"] == "ev001"
    assert result["sip_status"] == "paused"
    assert result["supporting_quote"] == "so I paused my SIP last month"
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request["model"] == DEFAULT_MODEL
    assert request["temperature"] == 0
    assert request["max_tokens"] == 500
    assert request["response_format"] == {"type": "json_object"}


def test_returned_id_must_match_input():
    analyser, _ = make_analyser(json.dumps(valid_payload(evidence_id="other")))

    with pytest.raises(ValueError, match="evidence_id"):
        analyser.analyse(make_record())


def test_invalid_json_fails_clearly():
    analyser, _ = make_analyser("this is not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        analyser.analyse(make_record())


def test_non_object_json_fails():
    analyser, _ = make_analyser(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="not an object"):
        analyser.analyse(make_record())


def test_missing_response_field_fails_loudly():
    payload = valid_payload()
    del payload["uncertainty"]
    analyser, _ = make_analyser(json.dumps(payload))

    with pytest.raises(ValueError, match="missing analysis field"):
        analyser.analyse(make_record())


def test_off_evidence_quote_downgraded_to_unknown():
    payload = valid_payload(supporting_quote="Markets doubled and I became rich")
    analyser, _ = make_analyser(json.dumps(payload))

    result = analyser.analyse(make_record())

    assert result["supporting_quote"] == "unknown"
    assert "not verbatim" in result["uncertainty"]
    assert result["sip_status"] == "paused"


def test_verbatim_quote_across_line_break_survives_whitespace_folding():
    payload = valid_payload(
        evidence_id="ev002",
        supporting_quote="I stopped my SIP. My salary was cut.",
    )
    analyser, _ = make_analyser(json.dumps(payload))

    result = analyser.analyse(
        make_record(evidence_id="ev002", raw_text="I stopped my SIP.\nMy salary was cut.")
    )

    assert result["supporting_quote"] == "I stopped my SIP. My salary was cut."
    assert result["uncertainty"] == "single comment, small sample"


def test_missing_api_key_raises_without_leaking(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralAI()


def test_prompt_contains_schema_rules_and_evidence():
    analyser, fake = make_analyser(json.dumps(valid_payload()))

    analyser.analyse(make_record())

    messages = fake.requests[0]["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "verbatim" in system
    assert "unknown" in system
    for field in (
        "evidence_id",
        "sentiment",
        "motivation",
        "behaviour",
        "problem_theme",
        "sip_status",
        "discontinuation_reason",
        "supporting_quote",
        "uncertainty",
    ):
        assert field in system
    assert "paused my SIP last month" in user
    assert "https://example.com/thread/42" in user
    assert "ev001" in user
    assert SYSTEM_PROMPT == system


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("MISTRAL_MODEL", "open-mistral-nemo")
    analyser, fake = make_analyser(json.dumps(valid_payload()))

    analyser.analyse(make_record())

    assert fake.requests[0]["model"] == "open-mistral-nemo"


def test_create_analyser_selects_mock(monkeypatch):
    monkeypatch.delenv("SIP_AI_PROVIDER", raising=False)
    monkeypatch.setenv("SIP_MOCK_AI", "1")

    assert isinstance(create_analyser(), MockAI)


def test_create_analyser_selects_mistral(monkeypatch):
    monkeypatch.delenv("SIP_AI_PROVIDER", raising=False)
    monkeypatch.setenv("SIP_MOCK_AI", "0")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")

    analyser = create_analyser()

    assert isinstance(analyser, MistralAI)
    assert analyser.model == "mistral-small-latest"


def test_create_analyser_requires_configuration(monkeypatch):
    monkeypatch.delenv("SIP_AI_PROVIDER", raising=False)
    monkeypatch.delenv("SIP_MOCK_AI", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SIP_AI_PROVIDER"):
        create_analyser()
