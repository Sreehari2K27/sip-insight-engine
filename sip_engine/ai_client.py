"""Mock AI analysis for one evidence item at a time.

Deterministic and offline: a tiny keyword-rule engine stands in for the future
Mistral-powered analyser, so tests never touch the network or spend money.
The analyser is read-only: it never modifies the evidence record and never
produces ``raw_text`` or ``source_url`` — original evidence stays separate.
"""

from __future__ import annotations

import json
import os

from .evidence_store import REQUIRED_COLUMNS

ANALYSIS_COLUMNS = (
    "evidence_id",
    "sentiment",
    "motivation",
    "behaviour",
    "problem_theme",
    "sip_status",
    "discontinuation_reason",
    "supporting_quote",
    "uncertainty",
)

SENTIMENT_VALUES = ("negative", "mixed", "neutral", "positive")
SIP_STATUS_VALUES = ("active", "paused", "reduced", "stopped", "unknown")

STATUS_RULES = (
    ("stopped", ("stop", "quit", "discontinu")),
    ("paused", ("pause", "on hold")),
    ("reduced", ("reduc", "lower")),
    ("active", ("continue", "ongoing", "every month", "regularly")),
)

NEGATIVE_WORDS = (
    "worri", "anxi", "afraid", "fear", "los", "fall", "crash", "burden",
    "disappoint", "regret", "confus", "frustrat",
)
POSITIVE_WORDS = (
    "confident", "happy", "peaceful", "glad", "disciplined", "worth it",
    "growing", "good return", "comfortable",
)

THEME_RULES = (
    ("market volatility", ("market", "crash", "fall", "volatil")),
    ("cash-flow strain", ("salary", "job", "income", "emi", "cash", "expens", "afford", "budget")),
    ("returns and fees", ("fee", "expense ratio", "return", "growth", "underperform")),
)

MOTIVATION_RULES = (
    ("tax savings", ("tax", "80c")),
    ("long-term wealth building", ("wealth", "goal", "retirement", "child", "house", "future")),
)

REASON_RULES = (
    ("market losses", ("market", "crash", "fall", "loss", "volatil")),
    ("income disruption", ("salary", "job", "income", "layoff", "emi")),
    ("financial emergency", ("emergency", "medical", "hospital")),
)

BEHAVIOUR_BY_STATUS = {
    "stopped": "describes stopping the SIP",
    "paused": "describes pausing the SIP",
    "reduced": "describes reducing the SIP amount",
    "active": "describes continuing the SIP",
    "unknown": "shares an opinion without stating a SIP action",
}

_INTERRUPTION_STATUSES = ("paused", "reduced", "stopped")

_MOCK_UNCERTAINTY = (
    "rule-based mock output; keyword matching on a tiny vocabulary, "
    "not model-generated"
)


def _matches(lowered_text: str, words: tuple[str, ...]) -> bool:
    return any(word in lowered_text for word in words)


def _first_keyword_sentence(text: str, keywords: tuple[str, ...]) -> str:
    for sentence in text.replace("!", ".").replace("?", ".").replace("\n", ".").split("."):
        candidate = sentence.strip()
        if candidate and _matches(candidate.lower(), keywords):
            return candidate
    return "unknown"


def validate_analysis(result: dict[str, str]) -> dict[str, str]:
    """Enforce the structured analysis schema; raise ValueError on violation."""
    if set(result) != set(ANALYSIS_COLUMNS):
        raise ValueError(f"Analysis keys must be exactly {list(ANALYSIS_COLUMNS)}")
    if not all(isinstance(value, str) for value in result.values()):
        raise ValueError("All analysis values must be strings")
    if result["sentiment"] not in SENTIMENT_VALUES:
        raise ValueError(f"sentiment must be one of {SENTIMENT_VALUES}")
    if result["sip_status"] not in SIP_STATUS_VALUES:
        raise ValueError(f"sip_status must be one of {SIP_STATUS_VALUES}")
    return result


class MockAI:
    """Deterministic rule-based analyser used until the Mistral integration."""

    name = "mock-rules-v1"

    def analyse(self, record: dict[str, str]) -> dict[str, str]:
        for column in REQUIRED_COLUMNS:
            if not str(record.get(column, "")).strip():
                raise ValueError(f"Cannot analyse without {column}")

        raw_text = str(record["raw_text"])
        lowered = raw_text.lower()

        sip_status = next(
            (status for status, words in STATUS_RULES if _matches(lowered, words)),
            "unknown",
        )

        negative_hits = sum(1 for word in NEGATIVE_WORDS if word in lowered)
        positive_hits = sum(1 for word in POSITIVE_WORDS if word in lowered)
        if negative_hits and negative_hits == positive_hits:
            sentiment = "mixed"
        elif negative_hits > positive_hits:
            sentiment = "negative"
        elif positive_hits > negative_hits:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        problem_theme = next(
            (theme for theme, words in THEME_RULES if _matches(lowered, words)),
            "unknown",
        )
        motivation = next(
            (label for label, words in MOTIVATION_RULES if _matches(lowered, words)),
            "unknown",
        )
        if sip_status in _INTERRUPTION_STATUSES:
            discontinuation_reason = next(
                (reason for reason, words in REASON_RULES if _matches(lowered, words)),
                "unknown",
            )
        else:
            discontinuation_reason = "unknown"

        rule_words = tuple(
            word
            for word_groups in (STATUS_RULES, THEME_RULES, MOTIVATION_RULES, REASON_RULES)
            for _, words in word_groups
            for word in words
        ) + NEGATIVE_WORDS + POSITIVE_WORDS
        supporting_quote = _first_keyword_sentence(raw_text, rule_words)

        uncertainty = _MOCK_UNCERTAINTY
        if supporting_quote == "unknown":
            uncertainty += "; no explicit supporting sentence found"

        return validate_analysis({
            "evidence_id": str(record.get("evidence_id", "")),
            "sentiment": sentiment,
            "motivation": motivation,
            "behaviour": BEHAVIOUR_BY_STATUS[sip_status],
            "problem_theme": problem_theme,
            "sip_status": sip_status,
            "discontinuation_reason": discontinuation_reason,
            "supporting_quote": supporting_quote,
            "uncertainty": uncertainty,
        })


DEFAULT_MODEL = "mistral-small-latest"

MAX_OUTPUT_TOKENS = 500


SYSTEM_PROMPT = (
    "You analyse one public comment about SIP investing in India.\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '- "evidence_id": the evidence ID given in the message\n'
    '- "sentiment": one of "negative", "mixed", "neutral", "positive"\n'
    '- "motivation": the investor\'s stated motivation or expectation, or "unknown"\n'
    '- "behaviour": a short description of the behaviour, experience, intention, or question described\n'
    '- "problem_theme": the single best-supported primary theme, or "unknown"\n'
    '- "sip_status": one of "active", "paused", "reduced", "stopped", "unknown"\n'
    '- "discontinuation_reason": explicit reason for pausing, reducing or stopping, or "unknown"\n'
    '- "supporting_quote": text copied character-for-character from the evidence text, or "unknown"\n'
    '- "uncertainty": one short sentence on uncertainty or limitations\n'
    "Theme guidance: choose a concise, specific theme when the comment supports one. "
    "Examples include fund selection confusion, lack of investment knowledge, "
    "affordability or cash-flow strain, returns or fees, market volatility, "
    "flexible contributions or top-ups, platform experience, feature requests, "
    "tax concerns, and withdrawal or SWP questions. "
    "These are examples, not mandatory labels; use another concise theme if more accurate. "
    "Do not use 'unknown' merely because the comment does not describe a problem: "
    "a clear feature request, positive platform experience, or investing intention can be a theme. "
    "Use 'unknown' when the text has too little context to identify a meaningful theme.\n"
    "Status guidance: an intention or question about starting a SIP is not proof of an active SIP. "
    "Use 'active' only when the text indicates the person currently has or continues a SIP; "
    "use 'paused', 'reduced', or 'stopped' only when that action actually occurred. "
    "For hypothetical or conditional scenarios, do not treat the described action as having occurred. "
    "Describe questions as questions and leave status unknown unless actual status is independently stated.\n"
    "Never invent facts or motivations. Use 'unknown' when evidence is insufficient. "
    "The supporting quote must be a verbatim substring of the evidence text; "
    "choose the shortest verbatim substring that supports the analysis, preferably one sentence "
    "and at most about 150 characters; never paraphrase the quote.\n"
    "For any hypothetical or conditional scenario, do not treat the described action as having occurred. "
    "Describe it as asking a question, and set status to unknown unless actual behaviour is independently stated.\n"
   )


def _require_evidence_fields(record: dict[str, str]) -> None:
    for column in REQUIRED_COLUMNS:
        if not str(record.get(column, "")).strip():
            raise ValueError(f"Cannot analyse without {column}")


def _fold_whitespace(text: str) -> str:
    return " ".join(text.split())


def _quote_in_text(quote: str, raw_text: str) -> bool:
    return _fold_whitespace(quote) in _fold_whitespace(raw_text)


def _build_user_message(record: dict[str, str]) -> str:
    return (
        f"Evidence ID: {record.get('evidence_id', '')}\n"
        f"Source URL: {record.get('source_url', '')}\n"
        "Evidence text:\n\"\"\"\n"
        f"{record['raw_text']}\n"
        '"""'
    )


def parse_ai_json(content) -> dict:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError) as exc:
        excerpt = str(content)[:120]
        raise ValueError(f"AI provider returned invalid JSON: {excerpt!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AI provider returned JSON that is not an object")
    return parsed


def to_validated_analysis(parsed: dict, record: dict[str, str]) -> dict[str, str]:
    evidence_id = str(record.get("evidence_id", ""))
    if evidence_id and str(parsed.get("evidence_id", "")) != evidence_id:
        raise ValueError(
            "Returned evidence_id does not match the input record "
            f"(got {str(parsed.get('evidence_id', ''))!r}, expected {evidence_id!r})"
        )
    missing_fields = [
        key for key in ANALYSIS_COLUMNS if key != "evidence_id" and key not in parsed
    ]
    if missing_fields:
        raise ValueError(f"AI response is missing analysis field(s): {missing_fields}")

    quote = str(parsed["supporting_quote"])
    downgraded = False
    if quote != "unknown" and not _quote_in_text(quote, str(record["raw_text"])):
        quote = "unknown"
        downgraded = True

    uncertainty = str(parsed["uncertainty"])
    if downgraded:
        uncertainty = (
            (uncertainty + "; " if uncertainty else "")
            + "supporting quote was not verbatim in the evidence text "
            "and was replaced with unknown"
        )

    return validate_analysis({
        "evidence_id": evidence_id,
        "sentiment": str(parsed["sentiment"]),
        "motivation": str(parsed["motivation"]),
        "behaviour": str(parsed["behaviour"]),
        "problem_theme": str(parsed["problem_theme"]),
        "sip_status": str(parsed["sip_status"]),
        "discontinuation_reason": str(parsed["discontinuation_reason"]),
        "supporting_quote": quote,
        "uncertainty": uncertainty,
    })


class MistralAI:
    """Mistral-backed analyser with the same interface as MockAI.

    The SDK client is built lazily on the first ``analyse()`` call, so this
    module and its tests run without the mistralai package installed. Pass
    ``client=`` to inject a fake for offline testing.
    """

    name = "mistral"

    def __init__(self, api_key=None, model=None, client=None) -> None:
        self._api_key = str(api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY", ""))
        if not self._api_key.strip():
            raise ValueError(
                "MISTRAL_API_KEY is not set; add it to your .env file. "
                "The key is read from the environment and never logged."
            )
        self.model = model or os.environ.get("MISTRAL_MODEL", "") or DEFAULT_MODEL
        self._client = client

    def analyse(self, record: dict[str, str]) -> dict[str, str]:
        _require_evidence_fields(record)
        client = self._client or self._build_client()
        response = client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(record)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        parsed = parse_ai_json(response.choices[0].message.content)
        return to_validated_analysis(parsed, record)

    def _build_client(self):
        try:
            from mistralai.client import Mistral
        except ImportError as exc:
            raise RuntimeError(
                "The mistralai package is not installed; "
                "run pip install -r requirements.txt"
            ) from exc
        return Mistral(api_key=self._api_key)


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

_GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "evidence_id": {"type": "STRING"},
        "sentiment": {"type": "STRING"},
        "motivation": {"type": "STRING"},
        "behaviour": {"type": "STRING"},
        "problem_theme": {"type": "STRING"},
        "sip_status": {"type": "STRING"},
        "discontinuation_reason": {"type": "STRING"},
        "supporting_quote": {"type": "STRING"},
        "uncertainty": {"type": "STRING"},
    },
    "required": list(ANALYSIS_COLUMNS),
}


class GeminiAI:
    """Gemini-backed analyser with the same interface as MockAI.

    Uses the official google-genai SDK with JSON output constrained by a
    response schema. The SDK client is built lazily on the first ``analyse()``
    call, so this module and its tests run without the package installed.
    Pass ``client=`` to inject a fake for offline testing.
    """

    name = "gemini"

    def __init__(self, api_key=None, model=None, client=None) -> None:
        self._api_key = str(api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", ""))
        if not self._api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set; add it to your .env file. "
                "The key is read from the environment and never logged."
            )
        self.model = model or os.environ.get("GEMINI_MODEL", "") or DEFAULT_GEMINI_MODEL
        self._client = client

    @staticmethod
    def _finish_reason(response) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        return str(getattr(candidates[0], "finish_reason", "") or "").split(".")[-1].upper()

    @staticmethod
    def _usage_tokens(response) -> tuple[str, str]:
        """(thinking tokens, output tokens) — 'unavailable' when not exposed."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return "unavailable", "unavailable"

        def token_count(attr: str) -> str:
            value = getattr(usage, attr, None)
            return str(value) if isinstance(value, int) else "unavailable"

        return token_count("thoughts_token_count"), token_count("candidates_token_count")

    def analyse(self, record: dict[str, str]) -> dict[str, str]:
        _require_evidence_fields(record)
        client = self._client or self._build_client()
        response = client.models.generate_content(
            model=self.model,
            contents=_build_user_message(record),
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": _GEMINI_RESPONSE_SCHEMA,
                "temperature": 0,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "thinking_config": {"thinking_level": "low"},
            },
        )
        finish_reason = self._finish_reason(response)
        if finish_reason == "MAX_TOKENS":
            thinking_tokens, output_tokens = self._usage_tokens(response)
            raise ValueError(
                "Gemini output truncated at max_output_tokens "
                f"(finish_reason=MAX_TOKENS; configured limit {MAX_OUTPUT_TOKENS}; "
                f"thinking tokens={thinking_tokens}, output tokens={output_tokens})"
            )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response")
        try:
            parsed = parse_ai_json(text)
        except ValueError as exc:
            raise ValueError(f"{exc} (finish_reason={finish_reason or 'unavailable'})") from exc
        return to_validated_analysis(parsed, record)

    def _build_client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is not installed; "
                "run pip install -r requirements.txt"
            ) from exc
        return genai.Client(api_key=self._api_key)


def create_analyser():
    """Pick the analyser: SIP_AI_PROVIDER wins; legacy SIP_MOCK_AI still honoured."""
    provider = os.environ.get("SIP_AI_PROVIDER", "").strip().lower()
    if provider:
        if provider == "mock":
            return MockAI()
        if provider == "mistral":
            return MistralAI()
        if provider == "gemini":
            return GeminiAI()
        raise ValueError(
            f"Unknown SIP_AI_PROVIDER={provider!r}; expected mock, mistral or gemini"
        )
    if os.environ.get("SIP_MOCK_AI", "") == "1":
        return MockAI()
    if os.environ.get("MISTRAL_API_KEY", "").strip():
        return MistralAI()
    raise ValueError(
        "No analyser available: set SIP_AI_PROVIDER=mock|mistral|gemini, "
        "or provide MISTRAL_API_KEY in your .env file"
    )
