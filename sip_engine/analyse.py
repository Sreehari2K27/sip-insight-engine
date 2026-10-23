"""AI analysis CLI for collected evidence.

Selects rows via EvidenceStore.read_all(): analysable = non-empty raw_text and
source_url; "already analysed" = ai_theme OR ai_uncertainty non-empty. The
provider comes from create_analyser() (mock mode gives zero-cost local runs).
Successful results are saved via EvidenceStore.save_analysis(); failed records
are reported by evidence ID and left unanalysed — never marked as analysed,
never retried. Interruptions (KeyboardInterrupt/SystemExit) are not swallowed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


_load_project_env()

EVIDENCE_PATH = PROJECT_ROOT / "data" / "evidence.csv"
ANALYSIS_BACKUP_PATH = EVIDENCE_PATH.parent / "analysis_backup.json"

from sip_engine.evidence_store import ANALYSIS_TO_CSV  # noqa: E402

BACKUP_COLUMNS = tuple(ANALYSIS_TO_CSV.values())


def is_analysed(record: dict[str, str]) -> bool:
    return (
        bool(str(record.get("ai_theme", "")).strip())
        or bool(str(record.get("ai_uncertainty", "")).strip())
    )


def _resolve_id_selection(records: list[dict[str, str]], prefixes: list[str]) -> list[dict[str, str]]:
    """Every supplied ID/prefix must resolve to exactly one record, or fail safely."""
    matches = []
    for prefix in prefixes:
        hits = [
            row for row in records
            if str(row.get("evidence_id", "")).lower().startswith(prefix)
        ]
        if not hits:
            raise ValueError(f"evidence-ids: id/prefix not found: {prefix[:12]}")
        if len(hits) > 1:
            raise ValueError(
                f"evidence-ids: id/prefix is ambiguous ({len(hits)} matches): {prefix[:12]}"
            )
        matches.append(hits[0])
    return matches


def select_records(
    store: EvidenceStore,
    source: str | None = None,
    force: bool = False,
    limit: int | None = None,
    evidence_ids: str | None = None,
) -> list[dict[str, str]]:
    records = [
        row
        for row in store.read_all()
        if str(row.get("raw_text", "")).strip() and str(row.get("source_url", "")).strip()
    ]
    if evidence_ids:
        prefixes = [p.strip().lower() for p in str(evidence_ids).split(",") if p.strip()]
        if prefixes:
            records = _resolve_id_selection(records, prefixes)
    if source:
        records = [row for row in records if row.get("platform", "") == source]
    if not force:
        records = [row for row in records if not is_analysed(row)]
    if limit is not None:
        records = records[: int(limit)]
    return records


def backup_existing_analysis(records: list[dict[str, str]], backup_path: Path) -> int:
    """Snapshot the 8 mapped ai_* fields of already-analysed selected records.

    Content-free of evidence text: only analysis columns + evidence_id keys.
    Existing backup entries are merged, never discarded; writes are atomic.
    Records without an evidence_id or without any populated ai_* field are
    skipped (nothing restorable to tie them back to).
    """
    payload: dict = {"saved_at": "", "record_count": 0, "records": {}}
    if os.path.exists(backup_path):
        with open(backup_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh) or {}
            payload["records"] = existing.get("records", {})
    backed = 0
    for record in records:
        evidence_id = str(record.get("evidence_id", "")).strip()
        if not evidence_id:
            continue
        if not any(str(record.get(column, "")).strip() for column in BACKUP_COLUMNS):
            continue
        payload["records"][evidence_id] = {col: str(record.get(col, "")) for col in BACKUP_COLUMNS}
        backed += 1
    from datetime import datetime, timezone

    payload["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["record_count"] = len(payload["records"])
    temp_path = str(backup_path) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(temp_path, backup_path)
    return backed


def analyse_records(
    store: EvidenceStore,
    records: list[dict[str, str]],
    analyser,
    backup_path: Path | None = None,
) -> dict:
    """analyse one record at a time; failures are never marked as analysed."""
    backed = 0
    if backup_path is not None:
        backed = backup_existing_analysis(records, backup_path)
        if backed:
            print(f"backed up existing analysis for {backed} record(s) -> {backup_path}")
    analysed = 0
    failed = 0
    for record in records:
        evidence_id = str(record.get("evidence_id", "")) or "(no id)"
        try:
            analysis = analyser.analyse(record)
            store.save_analysis(str(record.get("evidence_id", "")), analysis)
            analysed += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {evidence_id[:8]}: {str(exc)[:300]}")
            continue
    return {"analysed": analysed, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sip-engine analyse",
        description="Run AI analysis over collected evidence (mock provider by selection).",
    )
    parser.add_argument("--limit", type=int, default=None, help="process at most N records")
    parser.add_argument("--source", default=None, help="only this platform (e.g. youtube)")
    parser.add_argument(
        "--evidence-ids",
        default=None,
        help="comma-separated evidence IDs or ID prefixes; every one must resolve "
        "to exactly one record (fails safely before any analysis)",
    )
    parser.add_argument(
        "--force", action="store_true", help="reprocess already-analysed records too"
    )
    parser.add_argument(
        "--backup-existing-analysis",
        action="store_true",
        help="preserve existing ai_* analysis fields of the selected records "
        "before reanalysis (JSON merge file)",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be 0 or more")

    from sip_engine.ai_client import create_analyser
    from sip_engine.evidence_store import EvidenceStore

    store = EvidenceStore(EVIDENCE_PATH)
    records = select_records(
        store, source=args.source, force=args.force, limit=args.limit,
        evidence_ids=args.evidence_ids,
    )
    print(
        f"records selected: {len(records)} "
        f"(source={args.source or 'all'}, force={bool(args.force)})"
    )
    if args.backup_existing_analysis and not args.force:
        print("note: --backup-existing-analysis only backs up records being "
              "reprocessed; reprocessing requires --force when ai fields exist")
    backup_path = ANALYSIS_BACKUP_PATH if args.backup_existing_analysis else None
    summary = analyse_records(store, records, create_analyser(), backup_path=backup_path)
    print(f"analysed: {summary['analysed']}, failed: {summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
