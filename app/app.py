"""Streamlit dashboard for SIP Insight Engine.

Read-only view over data/evidence.csv: counts, filters, searchable evidence
table, and a per-record detail view. No AI provider calls, no writes.
"""

from __future__ import annotations

import html
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_data import (
    AI_FIELDS,
    FINAL_HYPOTHESES,
    OPPORTUNITY_LABELS,
    OPPORTUNITY_PATTERNS,
    record_analysis_status,
    apply_filters,
    build_new_record,
    dataset_status_counts,
    evidence_log_rows,
    insight_question_summary,
    filter_options,
    latest_run_per_source,
    load_collection_log,
    load_evidence,
    opportunity_evidence,
    opportunity_summary,
    recent_runs,
    select_supporting_example,
    source_registry_table,
    summary_stats,
)
from sip_engine.evidence_store import EvidenceStore

DATA_PATH = str(PROJECT_ROOT / "data" / "evidence.csv")
LOG_PATH = str(PROJECT_ROOT / "data" / "collection_log.csv")

THEME_CSS = """
<style>
    .stApp { background: #f6f7f9; color: #1e2230;
             font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8ebf0; padding-top: 1.2rem; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3.5rem; max-width: 1150px; }
    h1 { font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    h2, h3, h4, h5 { color: #1e2230; font-weight: 700; margin-top: 0.6rem; }
    div[data-testid="stMarkdownContainer"] p { line-height: 1.6; }
    .header-rule { height: 3px; width: 68px; background: #00d09c; border-radius: 2px; margin: 4px 0 14px 0; }
    .brand-chip { display: inline-block; background: #00d09c; color: #ffffff;
                  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
                  padding: 4px 12px; border-radius: 999px; margin-bottom: 0.4rem; }
    .subtitle { color: #5b6472; font-size: 1.0rem; margin-bottom: 0.1rem; }
    .groww-card { background: #ffffff; border: 1px solid #e8ebf0; border-radius: 14px;
                  padding: 18px 20px; box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
                  color: #1e2230; line-height: 1.55; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e8ebf0;
                               border-top: 3px solid #00d09c; border-radius: 12px;
                               padding: 14px 16px; box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05); }
    [data-testid="stMetricValue"] { color: #1e2230; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #5b6472; font-size: 0.8rem;
                                    text-transform: uppercase; letter-spacing: 0.06em; }
    .stDataFrame { border: 1px solid #e8ebf0; border-radius: 12px; overflow: hidden; }
    .stApp a { color: #00b386; }
    .empty-note { color: #3d4451; background: #f0fdf9; border: 1px solid #b7e8db;
                  border-radius: 12px; padding: 16px 18px; }
    .limitations-card { color: #4a5462; background: #f2faf7; border: 1px solid #d3ece2;
                        border-left: 4px solid #00d09c; border-radius: 12px;
                        padding: 12px 16px; font-size: 0.9rem; }
    .rank-badge { display: inline-flex; align-items: center; justify-content: center;
                  background: #e6faf5; color: #008f75; font-weight: 700;
                  width: 26px; height: 26px; border-radius: 999px; margin-right: 8px;
                  font-size: 0.85rem; }
    .chip { display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 0.76rem; font-weight: 600; margin: 2px 6px 2px 0; }
    .chip-green { background: #e6faf5; color: #008f75; border: 1px solid #b7e8db; }
    .chip-dark { background: #eef1f5; color: #3d4451; border: 1px solid #dbe0e8; }
    .quote-block { border-left: 3px solid #00d09c; background: #fbfdfc;
                   padding: 8px 14px; border-radius: 0 8px 8px 0;
                   font-style: italic; color: #3d4451; margin: 6px 0; }
    .problem-statement { font-weight: 700; color: #1e2230; }
    .muted-foot { color: #6b7280; font-size: 0.84rem; }
    .section-divider { height: 1px; background: #e8ebf0; margin: 26px 0; }
    .cq-row { display: flex; gap: 16px; padding: 8px 0;
              border-bottom: 1px solid #eef1f5; align-items: flex-start; }
    .cq-row:last-child { border-bottom: none; }
    .cq-left { flex: 0 0 46%; min-width: 0; }
    .cq-left b { line-height: 1.4; }
    .cq-left .chip { margin-top: 4px; }
    .cq-right { flex: 1 1 auto; min-width: 0; color: #4a5462;
                line-height: 1.55; font-size: 0.92rem; }
    @media (max-width: 760px) {
        .cq-row { flex-direction: column; gap: 4px; }
        .cq-left { flex: 0 0 auto; }
    }

.ai-metric {
    background: #f7fbf8;
    border: 1px solid #d8e9df;
    border-top: 3px solid #00a878;
    border-radius: 10px;
    padding: 14px 16px;
    min-height: 86px;
    overflow-wrap: anywhere;
}

.ai-metric-label {
    color: #587267;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.ai-metric-value {
    color: #173b2e;
    font-size: 1rem;
    font-weight: 650;
    line-height: 1.4;
}

.ai-quote {
    background: #f7fbf8;
    border-left: 3px solid #00a878;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    line-height: 1.65;
    color: #344b40;
}

.ai-uncertainty {
    background: #f0f7f2;
    border: 1px solid #d8e9df;
    border-radius: 8px;
    padding: 12px 16px;
    line-height: 1.6;
    color: #40574b;
}

</style>
"""


def _shorten(text, limit: int = 160) -> str:
    text = str(text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _safe_url(url) -> bool:
    return str(url or "").strip().startswith(("http://", "https://"))


def _ai_line(label: str, value) -> str:
    shown = str(value).strip() or "unknown"
    return (
        f'<p class="ai-line"><span class="ai-label">{html.escape(label)}</span>'
        f"{html.escape(shown)}</p>"
    )


def _render_detail(filtered):
    st.subheader("Evidence detail")
    if filtered.empty:
        st.markdown(
            '<div class="empty-note">No evidence matches the current filters.</div>',
            unsafe_allow_html=True,
        )
        return

    options = list(filtered.index)
    chosen = st.selectbox(
        "Select a record",
        options,
        format_func=lambda i: (
            f"{_shorten(filtered.at[i, 'evidence_id'], 8)} · "
            f"{filtered.at[i, 'platform'] or 'unknown source'} · "
            f"{filtered.at[i, 'captured_date'] or 'no date'} — "
            f"{_shorten(filtered.at[i, 'raw_text'], 60)}"
        ),
    )
    record = filtered.loc[chosen]

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Raw feedback")
        body = html.escape(str(record["raw_text"] or "")).replace("\n", "<br>")
        st.markdown(f'<div class="groww-card">{body}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("##### Record")
        st.write(f"**Evidence ID:** {record['evidence_id'] or '—'}")
        st.write(f"**Captured:** {record['captured_date'] or '—'}")
        st.write(f"**Platform:** {record['platform'] or '—'}")
        st.write(f"**Author:** {record['author_handle'] or '—'}")
        if _safe_url(record["source_url"]):
            st.markdown(f"[Open source]({record['source_url']})")
        else:
            st.write("No source URL captured")

    st.markdown("##### AI analysis")
    if any(str(record[field]).strip() for field in AI_FIELDS):
        st.markdown("**Classification**")
        cards = st.columns(4)
        fields = [
            ("Sentiment", "ai_sentiment"),
            ("Problem theme", "ai_theme"),
            ("SIP status", "ai_sip_status"),
            ("Discontinuation reason", "ai_discontinuation_reason"),
        ]
        for column, (label, field) in zip(cards, fields):
            value = str(record[field]).strip() or "Unknown"
            with column:
                st.markdown(
                    f'<div class="ai-metric"><div class="ai-metric-label">{label}</div>'
                    f'<div class="ai-metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        quote = str(record["ai_supporting_quote"]).strip()
        uncertainty = str(record["ai_uncertainty"]).strip()
        if quote and quote.lower() != "unknown":
            st.markdown("**Supporting quote**")
            st.markdown(
                f'<div class="ai-quote">{html.escape(quote)}</div>',
                unsafe_allow_html=True,
            )
        elif quote.lower() == "unknown":
            st.markdown('<div class="ai-uncertainty"><b>Supporting quote:</b> unknown — no verbatim sentence supports this field yet.</div>', unsafe_allow_html=True)
        if uncertainty:
            st.markdown("**Uncertainty / limitations**")
            st.markdown(
                f'<div class="ai-uncertainty">{html.escape(uncertainty)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="empty-note">AI analysis has not been run for this record yet. '
            "Run the analyser in mock or provider mode and save it with "
            "EvidenceStore.save_analysis(); this dashboard never calls AI providers.</div>",
            unsafe_allow_html=True,
        )


DISCOVERY_QUESTIONS = [
    ("SIP-start motivations and expectations", "Partial",
     "Motivation/expectation signals appear in analysed comments; no systematic coverage yet."),
    ("Behaviour changes after starting", "Not established",
     "One record shows an ongoing top-up habit; no before/after comparison of behaviour change."),
    ("Reasons for pausing/reducing/stopping", "Partial",
     "Two live stop reports with stated reasons (no benefit; demat-mode migration) — small but real evidence."),
    ("What investors check and how often", "Not established",
     "Nothing in the dataset describes how often investors check the app."),
    ("In-app confusion and worries", "Supported by current evidence",
     "Plan-type discovery, mandate setup and booking-time confusion are live-analysed."),
    ("Actions outside the app when unsure", "Partial",
     "Google-Pay/bank deactivation and ask-a-creator patterns noted; coverage incomplete."),
    ("Differences between continuers and drop-offs", "Not established",
     "No records compare continuers with drop-offs; no retention cohort exists."),
    ("Differences across investor segments", "Not established",
     "Only incidental segments (first-time, housewife) appear in single comments."),
    ("Consistent unmet needs", "Partial",
     "Emerging: plan-type clarity, mandate transparency and stop/step-up guidance; loss-safety misconceptions remain an unvalidated candidate."),
]


def _render_discovery_coverage():
    st.subheader("Discovery question coverage")
    st.caption(
        "Public comments cannot prove causation or actual 12-month SIP "
        "continuation — these are coverage ratings of the evidence, not "
        "outcome claims."
    )
    chip_class = {
        "Supported by current evidence": "chip-green",
        "Partial": "chip-dark",
        "Not established": "chip-dark",
    }
    rows_html = []
    for question, status, basis in DISCOVERY_QUESTIONS:
        rows_html.append(
            f'<div class="cq-row">'
            f'<div class="cq-left"><b>{html.escape(question)}</b><br>'
            f'<span class="chip {chip_class.get(status, "chip-dark")}">{html.escape(status)}</span></div>'
            f'<div class="cq-right">{html.escape(basis)}</div>'
            f'</div>'
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)


def _render_insight_questions():
    st.subheader("Insights — four discovery questions")
    st.caption(
        "Counts use live AI analyses only (mock and unanalysed rows excluded; "
        "mock rows stay visible in the AI Evidence Log). Counts are evidence "
        "status, not validated findings; values are observed field "
        "distributions, never causation, longitudinal behaviour or monitoring "
        "frequency. Intending to start a SIP is never counted as having one."
    )
    df = load_evidence(DATA_PATH)
    if df.empty:
        return
    summary = insight_question_summary(df)
    if summary["live_rows"] == 0:
        st.markdown(
            '<div class="empty-note">No live AI analyses exist yet, so the four '
            "discovery questions cannot be evidenced. Run the analysis CLI in "
            "mock or provider mode first.</div>",
            unsafe_allow_html=True,
        )
        return

    with st.container(border=True):
        st.markdown("**1. Why do investors start SIPs, and what do they expect?**")
        motivations = summary["motivations"]
        if motivations:
            chips = " ".join(
                f'<span class="chip chip-green">{_shorten(reason, 40)} ({count})</span>'
                for reason, count in motivations.most_common(4)
            )
            st.markdown(chips, unsafe_allow_html=True)
            example = select_supporting_example(df, "ai_motivation")
            if example is not None:
                quote = str(example.get("ai_supporting_quote", "")).strip()
                st.markdown(f'<div class="quote-block">\u201c{_shorten(quote, 150)}\u201d</div>', unsafe_allow_html=True)
                if _safe_url(example.get("source_url", "")):
                    st.markdown(f"[Open source]({example['source_url']})")
            else:
                st.markdown('<div class="ai-uncertainty">No suitable evidence example available yet.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ai-uncertainty">No live-analysed record states a start motivation yet.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**2. What behaviour do investors describe after starting?**")
        behaviours = summary["behaviours"]
        if behaviours:
            chips = " ".join(
                f'<span class="chip chip-green">{_shorten(desc, 40)} ({count})</span>'
                for desc, count in behaviours.most_common(4)
            )
            st.markdown(chips, unsafe_allow_html=True)
            example = select_supporting_example(df, "ai_behaviour")
            if example is not None:
                quote = str(example.get("ai_supporting_quote", "")).strip()
                st.markdown(f'<div class="quote-block">\u201c{_shorten(quote, 150)}\u201d</div>', unsafe_allow_html=True)
                if _safe_url(example.get("source_url", "")):
                    st.markdown(f"[Open source]({example['source_url']})")
            else:
                st.markdown('<div class="ai-uncertainty">No suitable evidence example available yet.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ai-uncertainty">No live-analysed record describes post-start behaviour yet.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**3. What reasons are given for pausing, reducing or stopping?**")
        disruptions = summary["disruption_rows"]
        confirmed = summary["confirmed_discontinuation_rows"]
        st.markdown(
            f'<span class="chip chip-dark">{len(disruptions)} rows with a pause/stop/reduced status claim</span>'
            f'<span class="chip chip-dark">{len(confirmed)} rows with a stated discontinuation reason</span>',
            unsafe_allow_html=True,
        )
        if not disruptions and not confirmed:
            st.markdown(
                '<div class="ai-uncertainty">Evidence gap: no live-analysed record '
                "records pausing, reducing or stopping a SIP — reasons therefore "
                "cannot be inferred. Questions about stopping exist (see the "
                "flexibility hypothesis) but describe capability doubts only.</div>",
                unsafe_allow_html=True,
            )
        else:
            example = select_supporting_example(df, "ai_discontinuation_reason")
            if example is None and disruptions:
                example = select_supporting_example(df, "ai_sip_status")
            if example is not None:
                quote = str(example.get("ai_supporting_quote", "")).strip()
                st.markdown(f'<div class="quote-block">\u201c{_shorten(quote, 150)}\u201d</div>', unsafe_allow_html=True)
                if _safe_url(example.get("source_url", "")):
                    st.markdown(f"[Open source]({example['source_url']})")
            else:
                st.markdown('<div class="ai-uncertainty">No suitable evidence example available yet.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**4. What do investors monitor, and how often?**")
        st.markdown(
            '<div class="ai-uncertainty">Evidence gap: the dataset has no field or '
            "comment describing monitoring habits or check-in frequency — nothing "
            "is inferable from public comments, which cannot establish frequency.</div>",
            unsafe_allow_html=True,
        )


def _render_add_evidence():
    st.subheader("Add evidence")
    st.caption(
        "Paste public comments you captured manually. Records are saved to "
        "data/evidence.csv only — no AI analysis or external calls on submission."
    )
    with st.form("add_evidence", clear_on_submit=True):
        platform = st.text_input(
            "Source platform", placeholder="reddit, twitter, youtube, forum…"
        )
        url = st.text_input("Source URL", placeholder="https://…")
        collected = st.date_input(
            "Date collected",
            value=None,
            min_value=date(2000, 1, 1),
            format="YYYY-MM-DD",
        )
        raw = st.text_area(
            "Raw feedback",
            height=140,
            placeholder="Paste the original comment verbatim — commas, quotes and line breaks are preserved.",
        )
        submitted = st.form_submit_button("Save evidence", type="primary", use_container_width=True)

    if not submitted:
        return
    try:
        record = build_new_record(platform, url, collected, raw)
    except ValueError as exc:
        st.error(f"Could not save the record: {exc}")
        return
    if EvidenceStore(DATA_PATH).has_source_key(record["source_key"]):
        st.error("This source URL is already captured — duplicate refused.")
        return
    try:
        stored = EvidenceStore(DATA_PATH).append(record)
    except ValueError as exc:
        st.error(f"Could not save the record: {exc}")
        return
    except OSError as exc:
        st.error(
            "Evidence could not be written to the data store "
            f"({exc.__class__.__name__}). On Streamlit Community Cloud the "
            "repository filesystem is read-only/ephemeral — evidence "
            "submissions persist only on a local checkout."
        )
        return
    st.session_state["last_saved_id"] = stored["evidence_id"]
    st.rerun()


def _render_opportunities():
    st.subheader("Candidate opportunities — 12-month SIP continuation")
    st.caption(
        "Counted live from data/evidence.csv. Keyword matches are candidate "
        "evidence, not confirmed user problems. Mandate/autopay process "
        "questions are not counted as failed payments unless the text carries "
        "a concrete failure/missed/deduction/charge/refund/reversal cue — they "
        "are shown as exploratory support instead. Public comments are not "
        "longitudinal retention data; no retention rates, churn, causality or "
        "impact can be inferred. The link to 12-month SIP continuation here remains an unproven hypothesis."
    )

    df = load_evidence(DATA_PATH)
    if df.empty:
        st.markdown(
            '<div class="empty-note">No evidence captured yet, so no opportunity '
            "counts can be shown.</div>",
            unsafe_allow_html=True,
        )
        return

    evidence_map = opportunity_evidence(df)
    all_summaries = opportunity_summary(df)
    display_keys = [key for key, _ in FINAL_HYPOTHESES]
    display_labels = [label for _, label in FINAL_HYPOTHESES]
    summaries = sorted(
        [s for s in all_summaries if s["opportunity"] in display_labels],
        key=lambda s: (s["ai_validated_rows"], s["evidence_strength"]),
        reverse=True,
    )
    confidence_by_label = {s["opportunity"]: s["confidence"] for s in summaries}
    validated_by_label = {s["opportunity"]: s["ai_validated_rows"] for s in summaries}
    payment_summary = next(
        (s for s in all_summaries if s["opportunity"] == "Failed SIP payment and unclear resolution"),
        None,
    )
    demoted_note = (
        "Payment/mandate transaction-resolution: demoted from the top three on "
        "evidence quality — one live-validated deduction incident (f6b13b84…) and "
        f'{payment_summary["related_process_confusion"] if payment_summary else 0} '
        "related process-confusion rows. Its records stay visible via the "
        "explorer's filters and the AI Evidence Log."
    )
    hypothesis_links = {
        "prime_or_plan_confusion": "Problem statement: how the first investment gets set up — plan types (regular/direct), Groww Prime and charges were surprising or opaque for first-time users.",
        "pause_stop_stepup": "Problem statement: how to change, pause, stop or step up a SIP was unclear — a flexibility/knowledge gap, including a small number of confirmed stop reports; causation remains unproven.",
        "loss_safety_misconceptions": "Problem statement: some comments suggest beginners expect guaranteed returns, insurance protection or FD-like safety from SIPs, or fear losing money — validated on 13 candidate rows; hypotheses remain unproven for 12-month continuation.",
    }
    next_steps = {
        "prime_or_plan_confusion": "Next validation step: backfill remaining unanalysed rows in this window; confirm the plan-default experience with one more direct report; then re-check app plan clarity end-to-end.",
        "pause_stop_stepup": "Next validation step: confirm at least one real pause/stop STORY exists; then map how the app presents change/stop options.",
        "loss_safety_misconceptions": "Next validation step: AI backfill on the candidate rows (provider window permitting) to test the safety misconception; do not present as findings until validated.",
    }
    unforced_note = (
        "Dataset note (not forced into these windows): the analysis also surfaces "
        "trust-erosion claims about mutual funds and early-withdrawal temptation "
        "questions — reported via the evidence log, not merged into these three."
    )

    st.markdown("##### Hypothesis ranking (by live-validated evidence, then evidence strength)")
    st.caption(unforced_note)
    st.dataframe(
        pd.DataFrame(summaries),
        hide_index=True,
        width="stretch",
        column_config={
            "opportunity": st.column_config.TextColumn("Hypothesis"),
            "evidence_strength": st.column_config.NumberColumn("Evidence strength (rows)"),
            "ai_validated_rows": st.column_config.NumberColumn("Live- analysed rows"),
            "recurrence": st.column_config.TextColumn("Recurrence"),
            "severity_signals": st.column_config.NumberColumn("Severity signals (word hits)"),
            "related_process_confusion": st.column_config.NumberColumn("Related exploratory rows"),
            "confidence": st.column_config.TextColumn("Confidence (heuristic)"),
        },
    )

    st.markdown(f'<div class="muted-foot">{demoted_note}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="limitations-card"><b>Research limitations:</b> these are '
        "hypotheses built from public comments — not proven causes of SIP "
        "discontinuation or measured 12-month continuation. Counts show candidate "
        "evidence and analysis status only.</div>",
        unsafe_allow_html=True,
    )

    rank_by_label = {s["opportunity"]: i + 1 for i, s in enumerate(summaries)}
    key_by_label = {label: key for key, label in FINAL_HYPOTHESES}
    for summary in summaries:
        label = summary["opportunity"]
        key = key_by_label[label]
        supporting = evidence_map[key]["supporting"]
        related = evidence_map[key]["related_exploratory"]
        validated_rows = [r for r in supporting if record_analysis_status(r) == "live" and r.get("ai_supporting_quote", "").strip()]
        with st.container(border=True):
            st.markdown(
                f'<h4 style="margin:0"><span class="rank-badge">{rank_by_label[label]}</span>'
                f'<span class="problem-statement">{label}</span></h4>',
                unsafe_allow_html=True,
            )
            st.caption(hypothesis_links[key])
            chips = (
                f'<span class="chip chip-green">{len(supporting)} candidate rows</span>'
                f'<span class="chip chip-dark">{validated_by_label[label]} live-validated</span>'
            )
            if related:
                chips += f'<span class="chip chip-dark">{len(related)} related exploratory</span>'
            chips += f'<span class="chip chip-dark">assessment: {confidence_by_label[label]}</span>'
            st.markdown(chips, unsafe_allow_html=True)
            for row in validated_rows[:1]:
                st.markdown(f'<div class="quote-block">\u201c{_shorten(row.get("ai_supporting_quote", ""), 140)}\u201d</div>', unsafe_allow_html=True)
                if _safe_url(row["source_url"]):
                    st.markdown(f"[Open source]({row['source_url']})")
            for row in supporting:
                if record_analysis_status(row) != "live" and row.get("raw_text", "").strip():
                    st.markdown(f'<div class="quote-block">\u201c{_shorten(row["raw_text"], 140)}\u201d</div>', unsafe_allow_html=True)
                    if _safe_url(row["source_url"]):
                        st.markdown(f"[Open source]({row['source_url']})")
                    break
            uncertainties = [
                r["ai_uncertainty"].strip() for r in validated_rows if r.get("ai_uncertainty", "").strip()
            ]
            if uncertainties or related:
                st.markdown('<div class="muted-foot"><b>Counterevidence / uncertainty:</b></div>', unsafe_allow_html=True)
                for line in uncertainties[:2]:
                    st.markdown(f'<div class="muted-foot">• {_shorten(line, 160)}</div>', unsafe_allow_html=True)
                if related:
                    st.markdown(
                        '<div class="muted-foot">• ' + f"{len(related)} related process-confusion row(s) kept out of this count — exploratory support, not confirmed problems.</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown(f'<div class="muted-foot">{next_steps[key]}</div>', unsafe_allow_html=True)

    with st.expander("AI evidence log (matched rows only)"):
        all_rows = df
        mock_count = all_rows["ai_uncertainty"].fillna("").str.lower().str.contains(
            "rule-based mock output"
        ).sum()
        live_count = len(all_rows) - mock_count
        st.caption(
            f"Analysis status across the dataset: {int(mock_count)} mock outputs · "
            f"{int(live_count)} live analyses. These are processing-status counts, "
            "not validated insight counts."
        )
        st.caption("Rows are labelled unverified until AI analysis is genuinely saved.")
        st.dataframe(
            pd.DataFrame(evidence_log_rows(df)).assign(
                source_text=lambda d: d["source_text"].map(lambda t: _shorten(t, 120))
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "evidence_id": st.column_config.TextColumn("Evidence ID"),
                "source_text": st.column_config.TextColumn("Source text", width="large"),
                "assigned_theme": st.column_config.TextColumn("Assigned theme", width="medium"),
                "validation_status": st.column_config.TextColumn("Validation status", width="medium"),
                "uncertainty": st.column_config.TextColumn("Uncertainty"),
                "source_url": st.column_config.LinkColumn("Source URL", display_text="open"),
            },
        )


def _render_collection_health(df=None):
    st.subheader("Collection health")
    st.caption(
        "Read-only view of data/collection_log.csv and the source registry. "
        "Rendering never runs collectors, makes network requests, or writes files."
    )

    registry = source_registry_table()
    if df is not None and not df.empty:
        present_sources = set(df["platform"].fillna("").astype(str).str.strip()) - {""}
        registry = registry[registry["source"].isin(present_sources)].reset_index(drop=True)
    

    log_df = load_collection_log(LOG_PATH)
    if log_df.empty:
        st.markdown(
            '<div class="empty-note">No collection runs recorded yet. Runs appear '
            "here after a collector is executed via the CLI (e.g. the Apple App "
            "Store source); this dashboard never triggers collection.</div>",
            unsafe_allow_html=True,
        )
        return

    latest = latest_run_per_source(log_df)
    st.markdown("##### Latest run per source")
    st.dataframe(
        latest.assign(error=latest["error"].map(lambda text: _shorten(text, 120)))[
            ["source", "collected_at", "status", "items_found", "items_saved", "items_skipped", "error"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "source": st.column_config.TextColumn("Source"),
            "collected_at": st.column_config.TextColumn("Last collected (UTC)"),
            "status": st.column_config.TextColumn("Outcome"),
            "items_found": st.column_config.NumberColumn("Found"),
            "items_saved": st.column_config.NumberColumn("Saved"),
            "items_skipped": st.column_config.NumberColumn("Skipped"),
            "error": st.column_config.TextColumn("Notes / errors", width="large"),
        },
    )

    recent = recent_runs(log_df)
    st.markdown("##### Recent runs (newest first)")
    st.caption(
        f"Showing the {len(recent)} most recent of {len(log_df)} recorded runs. "
        "Blank counts mean the log row had missing or malformed values — shown "
        "as-is, never guessed. Notes/errors are truncated for display only."
    )
    st.dataframe(
        recent.assign(error=recent["error"].map(lambda text: _shorten(text, 120)))[
            ["collected_at", "source", "status", "items_found", "items_saved", "items_skipped", "error"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "collected_at": st.column_config.TextColumn("Collected (UTC)"),
            "source": st.column_config.TextColumn("Source"),
            "status": st.column_config.TextColumn("Outcome"),
            "items_found": st.column_config.NumberColumn("Found"),
            "items_saved": st.column_config.NumberColumn("Saved"),
            "items_skipped": st.column_config.NumberColumn("Skipped"),
            "error": st.column_config.TextColumn("Notes / errors", width="large"),
        },
    )

    with st.expander("Full diagnostics text (recent runs)"):
        for _, run in recent.iterrows():
            detail = run["error"] if isinstance(run["error"], str) and run["error"] else "(no notes)"
            st.write(
                f"**{run['run_id']}** · {run['collected_at'] or 'no timestamp'} · "
                f"{run['source']} · {run['status']} — {detail}"
            )


def main():
    st.set_page_config(page_title="SIP Insight Engine", page_icon=None, layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown('<span class="brand-chip">SIP INSIGHT ENGINE</span>', unsafe_allow_html=True)
    st.title("Why DIP in SIP? Dashboard")
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Exploring potential barriers to first-time investors\u2019 '
                "SIP continuation — built from manually captured public conversations "
                "and approved-API collectors (YouTube, App Store, TradingQnA latest feed). "
                "<b>Groww</b> is the application being <i>studied</i>; it does not sponsor, "
                "affiliate with, or endorse this research.</p>", unsafe_allow_html=True)
    st.markdown("**Discovery method:** Collect → Validate → Discover → Prioritise → Hypothesise")
    st.caption(
        "Collect = manual capture or approved-API collectors · Validate = rule-based "
        "relevance filtering · Discover = live candidate-evidence counts · "
        "Prioritise = evidence strength on this page · Hypothesise = labelled, "
        "unverified links to continuation."
    )
    st.caption(
        "AI-driven metrics stay 0/pending until analysis is actually run and saved; "
        "a zero there does not mean zero problems."
    )

    if "last_saved_id" in st.session_state:
        saved_id = st.session_state.pop("last_saved_id")
        st.success(f"Evidence saved — record {saved_id[:8]}… now appears in the explorer.")

    try:
        df = load_evidence(DATA_PATH)
    except ValueError as exc:
        st.error(
            "The evidence file could not be read by the running code: "
            f"{exc} If the message mentions a newer schema, restart the "
            "Streamlit server (stop it and run `streamlit run app/app.py` again) "
            "or update the running code, then reload this page."
        )
        st.stop()

    if df.empty:
        st.markdown(
            '<div class="empty-note">No evidence captured yet. Use the form below to '
            "add the first record (raw_text and source_url required), optionally run "
            "analysis in mock mode later, then refresh.</div>",
            unsafe_allow_html=True,
        )
        _render_add_evidence()
        return

    with st.sidebar:
        st.subheader("Filters")
        with st.form("filters"):
            search = st.text_input(
                "Search feedback", key="filter_search", placeholder="e.g. paused, fees, market"
            )
            opportunity = st.selectbox(
                "Opportunity (candidate evidence)",
                ["All"] + list(OPPORTUNITY_LABELS.values()),
                key="filter_opportunity",
            )
            sentiment_options = filter_options(df, "sentiment")
            theme_options = filter_options(df, "problem_theme")
            status_options = filter_options(df, "sip_status")

            source = st.selectbox("Source", ["All"] + filter_options(df, "source"), key="filter_source")
            sentiment = st.selectbox("Sentiment", ["All"] + sentiment_options, key="filter_sentiment")
            if not sentiment_options:
                st.caption("Sentiment requires saved AI analysis.")
            problem_theme = st.selectbox("Problem theme", ["All"] + theme_options, key="filter_problem_theme")
            if not theme_options:
                st.caption("Problem theme requires saved AI analysis.")
            sip_status = st.selectbox("SIP status", ["All"] + status_options, key="filter_sip_status")
            if not status_options:
                st.caption("SIP status requires saved AI analysis.")
            published_since = st.date_input(
                "Published since (optional)",
                value=None,
                min_value=date(2000, 1, 1),
                format="YYYY-MM-DD",
                key="filter_published_since",
            )
            analysis_status_filter = st.selectbox(
                "Analysis status",
                ["Any", "Analysed (live)", "Analysed (mock)", "Unanalysed"],
                key="filter_analysis_status",
            )
            applied = st.form_submit_button("Apply filters", type="primary", use_container_width=True)
        if st.button("Reset filters", use_container_width=True):
            for widget_key in ("filter_search", "filter_opportunity", "filter_source",
                               "filter_sentiment", "filter_problem_theme",
                               "filter_sip_status", "filter_published_since",
                               "filter_analysis_status"):
                st.session_state.pop(widget_key, None)
            st.session_state.pop("applied_filters", None)
            st.rerun()

    if applied:
        status_values = {
            "Analysed (live)": "live",
            "Analysed (mock)": "mock",
            "Unanalysed": "unanalysed",
        }
        applied_filters = {
            "source": source,
            "sentiment": sentiment,
            "problem_theme": problem_theme,
            "sip_status": sip_status,
            "search": search,
            "opportunity": opportunity,
            "published_since": published_since,
            "analysis_status": status_values.get(analysis_status_filter, "Any"),
        }
        st.session_state["applied_filters"] = applied_filters
    else:
        applied_filters = st.session_state.get("applied_filters", {
            "source": "All", "sentiment": "All", "problem_theme": "All",
            "sip_status": "All", "search": "", "opportunity": "All",
            "published_since": None, "analysis_status": "Any",
        })

    filtered = apply_filters(df, **applied_filters)
    stats = summary_stats(df)

    status_counts = dataset_status_counts(df)
    live_analysed = status_counts["live"]
    mock_analysed = status_counts["mock"]
    unanalysed = status_counts["unanalysed"]
    hypotheses_with_evidence = sum(
        1 for s in opportunity_summary(df) if s["evidence_strength"] > 0
    )

    kpi_row_1 = st.columns(4)
    kpi_row_1[0].metric("Total evidence", stats["total"])
    kpi_row_1[1].metric("Sources", stats["sources"])
    kpi_row_1[2].metric("Analysed (live AI)", live_analysed)
    kpi_row_1[3].metric("Unanalysed", unanalysed)
    kpi_row_2 = st.columns(3)
    kpi_row_2[0].metric("Analysed (mock)", mock_analysed)
    kpi_row_2[1].metric("Candidate hypotheses (with evidence)", hypotheses_with_evidence)
    kpi_row_2[2].metric("Matching records", len(filtered))
    st.caption(
        "AI metrics are processing-status counts, not validated insight counts. "
        "Only 'Matching records' reacts to your filters; all other cards describe "
        "the whole dataset."
    )

    if stats["analysed"] == 0:
        st.markdown(
            '<div class="empty-note">AI analysis has not been run yet — sentiment, '
            "theme and status filters are empty until records are analysed and saved "
            "via EvidenceStore.save_analysis(). This dashboard never calls AI providers.</div>",
            unsafe_allow_html=True,
        )

    _render_opportunities()
    _render_discovery_coverage()
    _render_insight_questions()

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("##### Records by source")
        if stats["by_source"].empty:
            st.caption("No source information captured yet.")
        else:
            st.bar_chart(stats["by_source"], color="#00d09c")
            source_table = stats["by_source"].rename_axis("Source").reset_index(name="Records")
            st.dataframe(
                source_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Source": st.column_config.TextColumn("Source"),
                    "Records": st.column_config.NumberColumn("Records"),
                },
            )
    with chart_right:
        st.markdown("##### Records by capture date")
        if stats["by_date"].empty:
            st.caption("No capture dates recorded yet.")
        else:
            st.bar_chart(stats["by_date"], color="#00d09c")

    if not stats["by_theme"].empty:
        st.markdown("##### Problem themes")
        ranked_themes = stats["by_theme"].sort_values(ascending=False).head(8)
        import altair as alt

        themes_df = ranked_themes.rename_axis("theme").reset_index(name="count")
        bars = alt.Chart(themes_df).mark_bar(color="#00d09c").encode(
            x=alt.X("count:Q", title="Records", axis=alt.Axis(tickMinStep=1, grid=False)),
            y=alt.Y("theme:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
        )
        labels = bars.mark_text(align="left", dx=5, fontSize=12, color="#1e2230").encode(
            text="count:Q"
        )
        st.altair_chart(
            (bars + labels).properties(height=240),
            width="stretch",
        )
        top_text = " · ".join(f"{name} ({int(count)})" for name, count in stats["by_theme"].sort_values(ascending=False).head(5).items())
        st.caption(f"Top themes by count (actual values): {top_text} — the complete breakdown lives in the evidence explorer.")

    st.subheader("Evidence explorer")
    st.caption(f"Showing {len(filtered)} of {stats['total']} records")
    if filtered.empty:
        st.markdown(
            '<div class="empty-note">No evidence matches the current filters — '
            "adjust your filters or press Reset. The dataset itself still holds "
            f"{stats['total']} records.</div>",
            unsafe_allow_html=True,
        )
    table_df = filtered.assign(raw_text=filtered["raw_text"].map(lambda t: _shorten(t)))
    st.dataframe(
        table_df[
            [
                "captured_date",
                "platform",
                "raw_text",
                "source_url",
                "ai_sentiment",
                "ai_theme",
                "ai_sip_status",
                "ai_discontinuation_reason",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "captured_date": st.column_config.TextColumn("Captured"),
            "platform": st.column_config.TextColumn("Source"),
            "raw_text": st.column_config.TextColumn("Feedback", width="large"),
            "source_url": st.column_config.LinkColumn("Source URL", display_text="open"),
            "ai_sentiment": st.column_config.TextColumn("Sentiment"),
            "ai_theme": st.column_config.TextColumn("Problem theme"),
            "ai_sip_status": st.column_config.TextColumn("SIP status"),
            "ai_discontinuation_reason": st.column_config.TextColumn("Stop reason"),
        },
    )

    _render_detail(filtered)
    _render_add_evidence()
    _render_collection_health(df)


if __name__ == "__main__":
    main()
