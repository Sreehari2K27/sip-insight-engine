# SIP Insight Engine

Analyses publicly available conversations about SIP investing in India to surface
evidence-backed reasons why first-time mutual fund investors pause, reduce, or
stop their SIPs.

## Status

Working research dashboard: 249 evidence records (YouTube 244, TradingQnA 3,
App Store 1, Reddit 1) collected via approved sources; 33 records AI-analysed
(28 live Gemini + 5 mock placeholders) with the mock-integrity guard keeping
mock outputs out of validated counts. Dashboard: KPI cards, opportunity/
hypothesis ranking (final three + demoted payment thread), AI Evidence Log,
analysis-status filters, discovery-question coverage and the four-question
insights view. No scraping; no unauthorised collection; nothing committed yet
(single baseline commit + working tree).

## Remaining work

- Analyse the remaining unanalysed rows and the three unanalysed TradingQnA
  records (the last run hit Gemini 503 capacity errors) — no retry automation.
- Broaden source coverage: more YouTube videos, optionally TradingQnA tag
  feeds and App Store reviews (Apple throttling to be spaced out); Reddit
  remains requires_access pending API approval.
- Validate the loss-safety-misconceptions candidate hypothesis (unvalidated;
  matches are not findings).
- Real-provider verification of analytical quality; publication/licence notes
  (CC BY-NC-SA applies to TradingQnA contributions).

## Layout

- `sip_engine/evidence_store.py` — append-only CSV evidence storage
- `sip_engine/ai_client.py` — `MockAI` (deterministic, offline) and `MistralAI`
  (real API) behind one `analyse()` interface; `create_analyser()` picks
  between them
- `data/evidence.csv` — evidence database; one CSV holds original evidence and
  AI analysis side by side (schema header committed)
- `tests/` — offline regression tests
- `app/` — Streamlit UI (later step)

## Traceability rules

- `raw_text` and `source_url` are written once on append and never modified.
- AI output goes to separate columns (`ai_summary`, `ai_theme`, `confidence`).
- Snippets are captured manually (paste/upload); no automatic scraping.
- `save_analysis(evidence_id, analysis)` writes only `ai_*` columns via atomic
  file replacement; legacy 10-column CSVs migrate additively and idempotently
  on first access; any other header fails loudly without touching the file.

## Environment configuration

`create_analyser()` selects the analyser via `SIP_AI_PROVIDER=mock|mistral|gemini`:

- `mock` → `MockAI`: deterministic, offline, zero cost
- `mistral` → `MistralAI`: one JSON-mode call per item, temperature 0, at most
  500 output tokens, model `mistral-small-latest` (override `MISTRAL_MODEL`),
  key `MISTRAL_API_KEY`
- `gemini` → `GeminiAI`: same budget shape with schema-constrained JSON
  output, model `gemini-3.6-flash` (override `GEMINI_MODEL`), key
  `GEMINI_API_KEY`

Backward compatibility: without `SIP_AI_PROVIDER`, the legacy rule still
applies — `SIP_MOCK_AI=1` → `MockAI`, otherwise `MistralAI` when
`MISTRAL_API_KEY` is set.

Copy `.env.example` to `.env` for these settings. `.env` is git-ignored and
must never be committed; keys are read from the environment and never logged.
No live Gemini call has been made yet — smoke-test it exactly like Mistral
(one deliberate call on a hand-written record) before first real use.

## Dashboard

Run `streamlit run app/app.py` from the project root for a read-only evidence
dashboard: totals by source and capture date, filters for source, sentiment,
problem theme and SIP status, a searchable evidence table, and a per-record
detail view with the raw feedback and source URL. It reads `data/evidence.csv`
only — no AI provider calls, no writes, no invented numbers. An **Add Evidence**
form validates and saves new records through `EvidenceStore.append()` (no AI
analysis on submission) — every submission gets a `source_key`, and a URL
already captured is refused as a duplicate; manually captured evidence is any
platform, and `reddit` stays `requires_access` (manual capture is not
automated collection). A read-only **Collection health** section shows the
source registry with access statuses, the latest run per source, the ten most
recent runs (newest first) from `data/collection_log.csv`, and diagnostics —
blank cells mean missing/malformed log values, never guessed numbers, and
rendering the section never triggers collection or writes anything. The
visual style
is a generic green fintech theme; it is not affiliated with or endorsed by any
brokerage.

## Multi-source collection (groundwork)

The evidence schema now also carries `source_key` (a stable SHA-256 hash of
platform + normalised source URL, used for deduplication), `published_at`
(the item's original publication date when a source provides it) and
`collected_at` (collection timestamp). Older CSVs migrate additively and
idempotently, as before.

Source statuses (see `sip_engine/collectors/base.py`):

- `approved` — official API or documented feed; collection allowed
- `requires_access` — an official path exists but needs credentials or
  approval; collection waits until access is granted
- `unavailable` — no affordable or permitted method; never substituted with
  unauthorised scraping
- `not_implemented` — no adapter built yet

Dry run (no network, no LLM, no credentials, no writes):

    python -m sip_engine.collect --dry-run

The **Apple App Store** adapter (`--source appstore`) uses Apple's official
iTunes customer-reviews RSS feed. Reviews are stored verbatim; Apple's
`id.label` is a numeric review identifier (never used as a URL), and the
review permalink in `link.attributes.href` becomes `source_url` — entries
without an http(s) permalink are skipped and logged with
`link.scheme`/`link.shape` diagnostics. Relevance is a
transparent keyword gate (`sip`, `mutual fund(s)`, `systematic investment`,
`first investment/investor`, `monthly investment`); duplicates are skipped by
`source_key`; `items_skipped` in the run log covers duplicates, non-relevant
reviews and unusable entries; run `status` is `ok`, `partial_error` or `failed`.
Feed limits: newest ~500 reviews per app per country (50 per page, max 10
pages, path-style pagination), no search. Configure app ids with
`SIP_APPSTORE_APP_IDS` (comma-separated numbers from the `apps.apple.com`
app URL), optionally `SIP_APPSTORE_COUNTRY` (default `in`) and
`SIP_APPSTORE_PAGES` (1-10). A valid-but-empty feed is logged as `ok` with a
`note:` diagnostic in the error column (raw-entry counts distinguish an
empty storefront feed from rating-only entries); genuine failures keep
their full messages.

The **TradingQ&A** adapter (`--source tradingqna`) parses only Discourse's
documented RSS feeds. **Approved scope:** the `https://tradingqna.com/latest.rss`
feed (and `/tag/<name>.rss` feeds); robots.txt disallows `/c/*.rss` and
`/t/*/*.rss`, so category and topic paths must never be configured or fetched
— topic enrichment is excluded for this site.

TradingQnA source notes: user contributions are licensed CC BY-NC-SA 3.0
(non-commercial use with attribution and share-alike) — attribution is
provided structurally by the evidence schema (topic permalink, author handle,
publication date, platform, feed tag); derivative outputs inherit the same
licence. The Terms' automated-access wording has not been definitively
reviewed — "licence-only" reading so far; if a restrictive clause surfaces,
the source drops straight back to `requires_access`. Reuses the shared
relevance filter, dedup and run log. Its status is now `approved` for the
approved feed patterns only.

Real collection runs will append rows to `data/collection_log.csv`
(`run_id, collected_at, source, status, items_found, items_saved,
items_skipped, error`) so the dashboard can distinguish an empty source from
a failed collector. The CLI loads the project-root `.env` automatically
(`python-dotenv`); exported environment variables take precedence. Collection
never calls an LLM; AI analysis stays a separate explicit step
(`SIP_AI_PROVIDER=mock` for development).

## Deploying on Streamlit Community Cloud

- **Main file path:** `app/app.py`(set it in the deploy dialog; the repo root
  is the working directory, so `data/evidence.csv` and
  `data/collection_log.csv` resolve relative to it).
- **Data files must be committed** — the app reads the tracked
  `data/evidence.csv`; commit and push before deploying (untracked local
  files are invisible to the cloud).
- **Secrets:** the read-only dashboard needs no API keys or `.env` — never
  push `.env` (git-ignored). Analysis backfills (`sip_engine.analyse`) would
  need provider keys via Streamlit secrets if ever hosted — not required for
  viewing.
- **Writes:** Add Evidence submissions do **not persist** on Community Cloud
  (the repository filesystem is read-only/ephemeral) — capture evidence
  locally; the form shows a clear error when hosted writes fail.

## Live smoke test (one deliberate call)

1. `pip install -r requirements.txt`
2. Verify the SDK import (SDK 3.x moved the client class):
   `python -c "from mistralai.client import Mistral; print('Mistral SDK import successful')"`
3. `Copy-Item .env.example .env`, then set the real `MISTRAL_API_KEY` and
   `SIP_MOCK_AI=0` inside `.env`
4. Confirm the secret is protected: `git check-ignore .env` prints `.env`
5. Run one analysis on a hand-written record (nothing is saved to the CSV):
   `python -c "from dotenv import load_dotenv; load_dotenv(); import json; from sip_engine.ai_client import MistralAI; r={'evidence_id':'smoke1','source_url':'https://example.com/smoke','raw_text':'I stopped my SIP because the market fell and my salary was cut.'}; print(json.dumps(MistralAI().analyse(r), indent=2, ensure_ascii=False))"`
6. Verify every schema field is present, `sip_status` is `stopped`, and the
   supporting quote is verbatim; the Mistral dashboard should show exactly one
   small-model request.
