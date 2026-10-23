"""Pure data helpers for the Streamlit dashboard.

Read-only over the evidence CSV via the existing EvidenceStore; no AI
provider calls, no writes, no invented numbers.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from sip_engine.collectors.base import CollectedItem, LOG_COLUMNS, SOURCES
from sip_engine.evidence_store import COLUMNS, EvidenceStore

AI_FIELDS = (
    "ai_sentiment",
    "ai_motivation",
    "ai_behaviour",
    "ai_theme",
    "ai_sip_status",
    "ai_discontinuation_reason",
    "ai_supporting_quote",
    "ai_uncertainty",
)

FILTER_FIELDS = {
    "source": "platform",
    "sentiment": "ai_sentiment",
    "problem_theme": "ai_theme",
    "sip_status": "ai_sip_status",
}

INTERRUPTED_STATUSES = ("paused", "reduced", "stopped")


def load_evidence(path) -> pd.DataFrame:
    rows = EvidenceStore(path).read_all()
    return pd.DataFrame(rows, columns=list(COLUMNS))


def filter_options(df: pd.DataFrame, field_key: str) -> list[str]:
    if field_key not in FILTER_FIELDS:
        raise ValueError(f"Unknown filter field: {field_key}")
    column = FILTER_FIELDS[field_key]
    values = df[column].fillna("").astype(str).str.strip()
    return sorted(values[values != ""].unique())


def apply_filters(
    df: pd.DataFrame,
    source: str = "All",
    sentiment: str = "All",
    problem_theme: str = "All",
    sip_status: str = "All",
    search: str = "",
    opportunity: str = "All",
    published_since=None,
    analysis_status: str = "Any",
) -> pd.DataFrame:
    filtered = df
    if opportunity != "All":
        wanted_key = _OPPORTUNITY_KEYS_BY_LABEL.get(opportunity)
        if wanted_key is None:
            raise ValueError(f"Unknown opportunity filter: {opportunity}")
        mask = filtered["raw_text"].apply(lambda t: wanted_key in _matched_opportunities(t))
        filtered = filtered[mask]
    if analysis_status not in ("Any", ""):
        wanted_status = analysis_status.strip().lower()
        mask = filtered.apply(
            lambda record: record_analysis_status(record) == wanted_status, axis=1
        )
        filtered = filtered[mask]
    if published_since is not None and str(published_since).strip() and str(published_since) != "None":
        since = str(published_since)[:10]
        dates = filtered["published_at"].fillna("").astype(str).str.strip()
        parsed = pd.to_datetime(dates, errors="coerce", utc=True)
        in_range = (dates != "") & parsed.notna() & (parsed.dt.strftime("%Y-%m-%d") >= since)
        filtered = filtered[in_range]
    if source != "All":
        filtered = filtered[filtered["platform"] == source]
    if sentiment != "All":
        filtered = filtered[filtered["ai_sentiment"] == sentiment]
    if problem_theme != "All":
        filtered = filtered[filtered["ai_theme"] == problem_theme]
    if sip_status != "All":
        filtered = filtered[filtered["ai_sip_status"] == sip_status]
    needle = search.strip().lower()
    if needle:
        haystack = filtered["raw_text"].fillna("").astype(str).str.lower()
        filtered = filtered[haystack.str.contains(needle, regex=False)]
    return filtered.reset_index(drop=True)


def build_new_record(platform, source_url, captured_date, raw_text) -> dict[str, str]:
    """Validate an Add Evidence submission; raise ValueError if incomplete."""
    url = str(source_url or "").strip()
    text = str(raw_text or "").strip()
    problems = []
    if not url:
        problems.append("Source URL is required.")
    elif not url.startswith(("http://", "https://")):
        problems.append("Source URL must start with http:// or https://.")
    if not text:
        problems.append("Raw feedback is required.")
    if problems:
        raise ValueError(" ".join(problems))
    captured = ""
    if captured_date is not None and str(captured_date).strip():
        captured = (
            captured_date.isoformat()
            if hasattr(captured_date, "isoformat")
            else str(captured_date).strip()
        )
    return {
        "platform": str(platform or "").strip(),
        "source_url": url,
        "captured_date": captured,
        "raw_text": text,
        "source_key": CollectedItem(
            platform=str(platform or "").strip(), source_url=url, raw_text=""
        ).compute_source_key(),
    }


def summary_stats(df: pd.DataFrame) -> dict:
    by_source = df["platform"].fillna("").astype(str).str.strip()
    by_source = by_source[by_source != ""].value_counts()

    by_date = df["captured_date"].fillna("").astype(str).str.strip()
    by_date = by_date[by_date != ""].value_counts().sort_index()

    themes = df["ai_theme"].fillna("").astype(str).str.strip()
    themes = themes[~themes.isin(["", "unknown"])]
    by_theme = themes.value_counts()
    top_theme = by_theme.index[0] if not by_theme.empty else ""

    analysed = int(
        df[list(AI_FIELDS)]
        .apply(lambda col: col.fillna("").astype(str).str.strip() != "")
        .any(axis=1)
        .sum()
    )
    interrupted = int(
        df["ai_sip_status"].fillna("").astype(str).str.strip().isin(INTERRUPTED_STATUSES).sum()
    )

    return {
        "total": int(len(df)),
        "sources": int(by_source.size),
        "analysed": analysed,
        "interrupted": interrupted,
        "by_source": by_source,
        "by_date": by_date,
        "by_theme": by_theme,
        "top_theme": top_theme,
    }


def load_collection_log(path) -> pd.DataFrame:
    """Read-only view of the collection run log; missing file means no runs yet."""
    log_path = Path(path)
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame(columns=list(LOG_COLUMNS))
    rows = []
    with log_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({column: row.get(column) or "" for column in LOG_COLUMNS})
    df = pd.DataFrame(rows, columns=list(LOG_COLUMNS))
    for column in ("items_found", "items_saved", "items_skipped"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _sorted_by_timestamp(log_df: pd.DataFrame, newest_first: bool) -> pd.DataFrame:
    """Valid ISO timestamps ordered by recency; blank/malformed always last."""
    ordered = log_df.copy()
    ordered["_ts"] = ordered["collected_at"].fillna("").astype(str)
    ordered["_ts_rank"] = (ordered["_ts"] == "").astype(int)
    ordered = ordered.sort_values(
        ["_ts_rank", "_ts"], ascending=[True, not newest_first], kind="stable"
    )
    return ordered.drop(columns=["_ts", "_ts_rank"])


def latest_run_per_source(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return log_df
    ordered = _sorted_by_timestamp(log_df, newest_first=False)
    latest = ordered.groupby("source", sort=False).tail(1)
    return _sorted_by_timestamp(latest, newest_first=True)


def recent_runs(log_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if log_df.empty:
        return log_df
    return _sorted_by_timestamp(log_df, newest_first=True).head(limit)


def source_registry_table() -> pd.DataFrame:
    rows = [
        {
            "source": spec.name,
            "status": spec.status,
            "adapter": "yes" if spec.implemented else "not yet",
            "requires": ", ".join(spec.requires) if spec.requires else "-",
        }
        for spec in SOURCES.values()
    ]
    table = pd.DataFrame(rows, columns=["source", "status", "adapter", "requires"])
    return table.sort_values("source").reset_index(drop=True)


OPPORTUNITY_LABELS = {
    "prime_or_plan_confusion": "Groww Prime / regular-versus-direct plan confusion",
    "failed_payment_resolution": "Failed SIP payment and unclear resolution",
    "pause_stop_stepup": "Changing, pausing, or stopping SIP contributions",
    "loss_safety_misconceptions": "Return guarantees and loss-safety misconceptions",
}

FINAL_HYPOTHESES = (
    ("prime_or_plan_confusion", "Direct versus regular plans, Groww Prime and fee transparency"),
    ("pause_stop_stepup", "SIP flexibility: changing, step-ups, pausing, stopping and withdrawals"),
    ("loss_safety_misconceptions", "Return guarantees and loss-safety misconceptions"),
)

def record_analysis_status(record) -> str:
    """'live' | 'mock' | 'unanalysed' for a single evidence record."""
    uncertainty = str(record.get("ai_uncertainty", "") or "")
    if _MOCK_ANALYSIS_SIGNATURE in uncertainty.lower():
        return "mock"
    if uncertainty.strip():
        return "live"
    return "unanalysed"


def dataset_status_counts(df: pd.DataFrame) -> dict:
    """Processing-status counts (not insight counts): live/mock/unanalysed."""
    statuses = (record_analysis_status(record) for _, record in df.iterrows())
    counts = Counter(statuses)
    return {key: counts.get(key, 0) for key in ("live", "mock", "unanalysed")}


_OPPORTUNITY_KEYS_BY_LABEL = {label: key for key, label in OPPORTUNITY_LABELS.items()}

OPPORTUNITY_PATTERNS = {
    "prime_or_plan_confusion": re.compile(
        r"\b(?:prime|(?:regular|direct)\s+(?:sip|plan|fund)|charges?|commission|fees?|expense|switch)\b",
        re.IGNORECASE,
    ),
    "failed_payment_resolution": re.compile(
        r"\b(?:payment|autopay|mandate|miss(?:ed)?|fail(?:ed|s)?|bounce(?:d)?|"
        r"deduct(?:ed)?|refund|reversal|charged|credited|bank account)\b",
        re.IGNORECASE,
    ),
    "pause_stop_stepup": re.compile(
        r"\b(?:paused?|stop(?:ped)?|band|rok|ruka|step.?up|reduc(?:e|ed)|resume|"
        r"plan change|amount change)\b",
        re.IGNORECASE,
    ),
    "loss_safety_misconceptions": re.compile(
        r"\b(?:guarantee|गारंटी|insur(?:ed|ance)|bima|yojana|death|fixed\s+deposit|\bfd\b|"
        r"dub\s*jaye|paisa\s*dub|surakshit|milega\s+ya\s+dub)\b",
        re.IGNORECASE,
    ),
}

_FAILURE_CUE = re.compile(
    r"\b(?:miss(?:ed)?|fail(?:ed|s)?|bounce(?:d)?|deduct(?:ed)?|refund|reversal|charged)\b",
    re.IGNORECASE,
)

_TRADING_OPTIONS_PHRASE = "trading in options"
_ANNOUNCEMENT_PHRASES = (
    "beta is live",
    "the daily brief",
    "most-requested features",
    "rolling out",
)


def _matched_opportunities(text: str) -> list[str]:
    lowered = str(text or "").lower()
    if _TRADING_OPTIONS_PHRASE in lowered:
        return []
    if any(phrase in lowered for phrase in _ANNOUNCEMENT_PHRASES):
        return []
    return [key for key, rx in OPPORTUNITY_PATTERNS.items() if rx.search(str(text or ""))]


_FUTURE_MARKER = re.compile(r"\b(?:kaega|karega|karenge|jayega|hoga|dega|denge)\b", re.IGNORECASE)


def _failure_cue_is_future_tense(text: str) -> bool:
    """True only when the SENTENCE containing a failure cue frames it as
    future/conditional (e.g. 'kya app deduct kaega?') — such sentences are
    process questions, not incident reports. Sentence-scoped, not whole-text."""
    sentences = re.split(r"[\n.;?!]+", str(text or ""))
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and _FAILURE_CUE.search(cleaned):
            return bool(_FUTURE_MARKER.search(cleaned))
    return False


def opportunity_evidence(df: pd.DataFrame) -> dict[str, dict[str, list[dict]]]:
    """Candidate-evidence rows per opportunity, counted live from the CSV.

    Returns {opportunity_key: {"supporting": [...], "related_exploratory": [...]}}.
    "supporting" rows count toward the opportunity's evidence strength.
    "related_exploratory" rows matched the broad group but lack the cue the
    group requires: failed-payment matches without a concrete
    failure/missed/deduction/charge/refund/reversal signal are mandate or
    process-confusion questions, kept out of the count.
    A row matching several groups appears in each.
    """
    result: dict[str, dict[str, list[dict]]] = {
        key: {"supporting": [], "related_exploratory": []} for key in OPPORTUNITY_PATTERNS
    }
    for _, record in df.iterrows():
        hits = _matched_opportunities(record.get("raw_text", ""))
        for key in hits:
            entry = {
                "evidence_id": record.get("evidence_id", ""),
                "raw_text": record.get("raw_text", ""),
                "source_url": record.get("source_url", ""),
                "published_at": record.get("published_at", ""),
                "platform": record.get("platform", ""),
                "author_handle": record.get("author_handle", ""),
                "matched_groups": ", ".join(OPPORTUNITY_LABELS[k] for k in hits),
                "ai_supporting_quote": record.get("ai_supporting_quote", ""),
                "ai_uncertainty": record.get("ai_uncertainty", ""),
                "ai_theme": record.get("ai_theme", ""),
            }
            if key == "failed_payment_resolution" and (
                not _FAILURE_CUE.search(str(record.get("raw_text", "")))
                or _failure_cue_is_future_tense(str(record.get("raw_text", "")))
            ):
                result[key]["related_exploratory"].append(entry)
            else:
                result[key]["supporting"].append(entry)
    return result


_SEVERITY_WORDS = re.compile(
    r"(money|\u20b9|\brs\b|deduct|charges?|loss|fail|refund|credited|withdraw)",
    re.IGNORECASE,
)


def _video_token(source_url: str) -> str:
    parsed = _url_parts(source_url)
    if parsed is not None and parsed.path == "/watch":
        return parsed.query.split("v=")[-1].split("&")[0] if "v=" in parsed.query else str(source_url)
    return str(source_url)


def _url_parts(source_url: str):
    from urllib.parse import urlparse

    value = str(source_url or "").strip()
    if not value.startswith(("http://", "https://")):
        return None
    return urlparse(value)


def opportunity_summary(df: pd.DataFrame) -> list[dict]:
    """Comparison table: evidence strength / recurrence / severity signals /
    confidence. Counts and heuristic labels only — never retention, churn,
    causality or prevalence. Groups with fewer than 2 supporting rows are
    labelled exploratory (sparse support)."""
    evidence_map = opportunity_evidence(df)
    summaries = []
    for key, label in OPPORTUNITY_LABELS.items():
        supporting = evidence_map[key]["supporting"]
        related_count = len(evidence_map[key]["related_exploratory"])
        label = dict(FINAL_HYPOTHESES).get(key, label)
        distinct_rows = {r["evidence_id"] for r in supporting}
        distinct_videos = {_video_token(r["source_url"]) for r in supporting}
        distinct_authors = {r["author_handle"] for r in supporting if r.get("author_handle")}
        severity_hits = sum(len(_SEVERITY_WORDS.findall(r["raw_text"])) for r in supporting)
        ai_validated = sum(
            1 for r in supporting
            if record_analysis_status(r) == "live" and r.get("ai_supporting_quote", "").strip()
        )
        if key == "loss_safety_misconceptions" and ai_validated == 0:
            # Candidate matches are unanalysed; keyword counts cannot validate them.
            confidence = "unvalidated candidate"
        elif len(distinct_rows) < 2:
            confidence = "exploratory (sparse support)"
        elif key == "prime_or_plan_confusion":
            # Audit assessment: multiple concrete experiences, but source coverage is limited.
            confidence = "medium" if len(distinct_rows) >= 3 and len(distinct_videos) >= 2 else "low"
        elif key == "pause_stop_stepup":
            # Recurring questions, not confirmed lived pause/stop incidents.
            confidence = "medium-low"
        elif len(distinct_rows) >= 3 and len(distinct_videos) >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        summaries.append(
            {
                "opportunity": label,
                "evidence_strength": len(distinct_rows),
                "recurrence": f"{len(distinct_rows)} rows, {len(distinct_videos)} sources, {len(distinct_authors)} authors",
                "severity_signals": severity_hits,
                "related_process_confusion": related_count,
                "confidence": confidence,
                "ai_validated_rows": ai_validated,
            }
        )
    return summaries


_MOCK_ANALYSIS_SIGNATURE = (
    "rule-based mock output; keyword matching on a tiny vocabulary, "
    "not model-generated"
)


def insight_question_summary(df: pd.DataFrame) -> dict:
    """Counts for the four discovery questions from LIVE analyses only.

    Mock rows and unanalysed rows never count; unknown/blank values are
    excluded from distributions and remain visible as gaps. Nothing here
    infers monitoring frequency, causation or longitudinal behaviour.
    """
    live = [
        record for _, record in df.iterrows()
        if record_analysis_status(record) == "live"
    ]

    def counts_for(field: str) -> Counter:
        values = Counter(
            str(record.get(field, "")).strip() for record in live
        )
        values.pop("unknown", None)
        values.pop("", None)
        return values

    disruptions = [
        record for record in live
        if str(record.get("ai_sip_status", "")).strip() in ("paused", "stopped", "reduced")
    ]
    confirmed_reasons = [
        record for record in live
        if str(record.get("ai_discontinuation_reason", "")).strip() not in ("unknown", "")
    ]
    return {
        "live_rows": len(live),
        "motivations": counts_for("ai_motivation"),
        "behaviours": counts_for("ai_behaviour"),
        "themes": counts_for("ai_theme"),
        "disruption_rows": disruptions,
        "confirmed_discontinuation_rows": confirmed_reasons,
    }


_GENERIC_BEHAVIOUR_VALUES = ("unknown", "", "shares an opinion without stating a sip action")


def select_supporting_example(df: pd.DataFrame, field: str):
    """First live record whose analysis field AND supporting quote both carry
    real (non-'unknown', non-blank, non-boilerplate) content — else None.

    Never reuses a quote just because it exists: the record's analysed field
    must answer the question and its verbatim quote must support it.
    """
    for _, record in df.iterrows():
        if record_analysis_status(record) != "live":
            continue
        value = str(record.get(field, "")).strip()
        if value.lower() in _GENERIC_BEHAVIOUR_VALUES:
            continue
        quote = str(record.get("ai_supporting_quote", "")).strip()
        if not quote or quote.lower() == "unknown":
            continue
        return record
    return None


def evidence_log_rows(df: pd.DataFrame) -> list[dict]:
    """AI Evidence Log for matched rows only; honest validation status.

    Mock-signed rows (their ai_uncertainty carries the mock provider's
    signature) can never be labelled ai-validated, even when a supporting
    quote exists — their quotes/states are placeholders, not model output.
    """
    rows = []
    for _, record in df.iterrows():
        hits = _matched_opportunities(record.get("raw_text", ""))
        if not hits:
            continue
        validated = bool(record.get("ai_supporting_quote", "").strip())
        is_mock = _MOCK_ANALYSIS_SIGNATURE in str(record.get("ai_uncertainty", "")).lower()
        if is_mock:
            validation_status = "unverified — mock placeholder output"
        elif validated:
            validation_status = "ai-validated"
        else:
            validation_status = "unverified — no AI analysis yet"
        rows.append(
            {
                "evidence_id": record.get("evidence_id", ""),
                "source_text": record.get("raw_text", ""),
                "assigned_theme": record.get("ai_theme", "")
                or ", ".join(OPPORTUNITY_LABELS[k] for k in hits),
                "validation_status": validation_status,
                "uncertainty": record.get("ai_uncertainty", "") or "—",
                "source_url": record.get("source_url", ""),
            }
        )
    return rows
