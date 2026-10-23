"""Offline tests for the Gemini-backed client using an injected fake client.

No network, no API spend, and no google-genai package required.
"""

import json

import pytest

from sip_engine.ai_client import (
    ANALYSIS_COLUMNS,
    DEFAULT_GEMINI_MODEL,
    GeminiAI,
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


class FakeCandidate:
    def __init__(self, finish_reason=""):
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, thoughts_token_count=None, candidates_token_count=None):
        self.thoughts_token_count = thoughts_token_count
        self.candidates_token_count = candidates_token_count


class FakeGeminiResponse:
    def __init__(self, text, candidates=None, usage=None):
        self.text = text
        self.candidates = candidates
        self.usage_metadata = usage


class FakeModels:
    def __init__(self, content):
        self._content = content
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        payload = self._content
        candidates = None
        usage = None
        if isinstance(payload, tuple):
            payload = list(payload) + [None, None]
            payload, candidates, usage = payload[0], payload[1], payload[2]
        return FakeGeminiResponse(payload, candidates, usage)


class FakeGenaiClient:
    def __init__(self, content=""):
        self.models = FakeModels(content)


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
    fake = FakeGenaiClient(content)
    return GeminiAI(api_key="test-key-not-real", client=fake), fake


def test_valid_output_passes_schema_and_single_call(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    analyser, fake = make_analyser(json.dumps(valid_payload()))

    result = analyser.analyse(make_record())

    assert set(result) == set(ANALYSIS_COLUMNS)
    assert validate_analysis(result) == result
    assert result["evidence_id"] == "ev001"
    assert result["sip_status"] == "paused"
    assert len(fake.models.requests) == 1
    request = fake.models.requests[0]
    assert request["model"] == DEFAULT_GEMINI_MODEL
    config = request["config"]
    assert config["temperature"] == 0
    assert config["max_output_tokens"] == 500
    assert config["response_mime_type"] == "application/json"
    assert set(config["response_schema"]["properties"]) == set(ANALYSIS_COLUMNS)
    assert config["system_instruction"] == SYSTEM_PROMPT


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


def test_empty_response_fails_clearly():
    analyser, _ = make_analyser("")

    with pytest.raises(ValueError, match="empty response"):
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


def test_max_tokens_finish_reason_raises_truncation_error():
    truncated = json.dumps(valid_payload())[:80]
    fake = FakeGenaiClient((truncated, (FakeCandidate("MAX_TOKENS"),)))
    analyser = GeminiAI(api_key="test-key-not-real", client=fake)

    with pytest.raises(ValueError, match="truncated at max_output_tokens"):
        analyser.analyse(make_record())

    assert len(fake.models.requests) == 1
    assert fake.models.requests[0]["config"]["max_output_tokens"] == 500


def test_malformed_json_error_includes_finish_reason():
    fake = FakeGenaiClient(("this is not json", (FakeCandidate("STOP"),)))
    analyser = GeminiAI(api_key="test-key-not-real", client=fake)

    with pytest.raises(ValueError, match="finish_reason=STOP") as raised:
        analyser.analyse(make_record())

    assert "invalid JSON" in str(raised.value)


def test_max_tokens_without_candidates_falls_back_to_parse_error():
    truncated = json.dumps(valid_payload())[:60]
    fake = FakeGenaiClient(truncated)
    analyser = GeminiAI(api_key="test-key-not-real", client=fake)

    with pytest.raises(ValueError, match="invalid JSON"):
        analyser.analyse(make_record())


def test_max_tokens_error_reports_thinking_and_output_tokens():
    truncated = json.dumps(valid_payload())[:80]
    usage = FakeUsage(thoughts_token_count=412, candidates_token_count=88)
    fake = FakeGenaiClient((truncated, (FakeCandidate("MAX_TOKENS"),), usage))
    analyser = GeminiAI(api_key="test-key-not-real", client=fake)

    with pytest.raises(ValueError) as raised:
        analyser.analyse(make_record())

    message = str(raised.value)
    assert "truncated at max_output_tokens" in message
    assert "configured limit 500" in message
    assert "thinking tokens=412, output tokens=88" in message


def test_max_tokens_error_with_missing_or_partial_usage_metadata():
    truncated = json.dumps(valid_payload())[:80]

    no_usage = FakeGenaiClient((truncated, (FakeCandidate("MAX_TOKENS"),)))
    with pytest.raises(ValueError, match="thinking tokens=unavailable, output tokens=unavailable"):
        GeminiAI(api_key="test-key-not-real", client=no_usage).analyse(make_record())

    partial = FakeGenaiClient((truncated, (FakeCandidate("MAX_TOKENS"),), FakeUsage(candidates_token_count=88)))
    with pytest.raises(ValueError, match="thinking tokens=unavailable, output tokens=88"):
        GeminiAI(api_key="test-key-not-real", client=partial).analyse(make_record())


def test_prompt_requires_short_verbatim_quotes():
    assert "shortest verbatim substring" in SYSTEM_PROMPT
    assert "150 characters" in SYSTEM_PROMPT
    assert "never paraphrase" in SYSTEM_PROMPT


def test_prompt_distinguishes_hypothetical_questions():
    assert "hypothetical or conditional scenario" in SYSTEM_PROMPT
    assert "do not treat the described action as having occurred" in SYSTEM_PROMPT
    assert "asking a question" in SYSTEM_PROMPT


def test_missing_api_key_raises_without_leaking(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiAI()


def test_prompt_contains_schema_rules_and_evidence():
    analyser, fake = make_analyser(json.dumps(valid_payload()))

    analyser.analyse(make_record())

    request = fake.models.requests[0]
    system = request["config"]["system_instruction"]
    user = request["contents"]
    assert "verbatim" in system
    for field in ANALYSIS_COLUMNS:
        assert field in system
    assert "paused my SIP last month" in user
    assert "https://example.com/thread/42" in user
    assert "ev001" in user


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-pro")
    analyser, fake = make_analyser(json.dumps(valid_payload()))

    analyser.analyse(make_record())

    assert fake.models.requests[0]["model"] == "gemini-3.6-pro"


def test_provider_selection_gemini(monkeypatch):
    monkeypatch.setenv("SIP_AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")

    analyser = create_analyser()

    assert isinstance(analyser, GeminiAI)
    assert analyser.model == "gemini-3.6-flash"


def test_provider_selection_mock_overrides_legacy(monkeypatch):
    monkeypatch.setenv("SIP_AI_PROVIDER", "mock")
    monkeypatch.setenv("SIP_MOCK_AI", "0")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")

    assert isinstance(create_analyser(), MockAI)


def test_provider_selection_mistral(monkeypatch):
    monkeypatch.setenv("SIP_AI_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")

    assert isinstance(create_analyser(), MistralAI)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("SIP_AI_PROVIDER", "anthropic")

    with pytest.raises(ValueError, match="SIP_AI_PROVIDER"):
        create_analyser()


def test_legacy_selection_still_works(monkeypatch):
    monkeypatch.delenv("SIP_AI_PROVIDER", raising=False)
    monkeypatch.setenv("SIP_MOCK_AI", "1")

    assert isinstance(create_analyser(), MockAI)

    monkeypatch.setenv("SIP_MOCK_AI", "0")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")

    assert isinstance(create_analyser(), MistralAI)
