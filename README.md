# SIP Insight Engine

Analyses publicly available conversations about SIP investing in India to surface
evidence-backed reasons why first-time mutual fund investors pause, reduce, or
stop their SIPs.

## Status

Step 1: project skeleton and CSV evidence store. No scraping, no AI calls, no
Streamlit UI yet.

## Layout

- `sip_engine/evidence_store.py` — append-only CSV evidence storage
- `data/evidence.csv` — evidence database (schema header committed)
- `tests/` — offline regression tests
- `app/` — Streamlit UI (later step)

## Traceability rules

- `raw_text` and `source_url` are written once on append and never modified.
- AI output goes to separate columns (`ai_summary`, `ai_theme`, `confidence`).
- Snippets are captured manually (paste/upload); no automatic scraping.

## Secrets

Copy `.env.example` to `.env` and set `MISTRAL_API_KEY` there. `.env` is
git-ignored and must never be committed. `SIP_MOCK_AI=1` forces mocked AI
responses for offline tests.
