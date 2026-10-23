"""Smallest collection runner: dedup, evidence append, run-log writing.

Collection never calls an LLM. Items already stored (same source_key) are
skipped and counted; non-relevant items are counted as skipped too.
"""

from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sip_engine.collectors.base import LOG_COLUMNS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_run_log(
    log_path, source: str, status: str, found: int, saved: int, skipped: int, error: str = ""
) -> dict:
    path = Path(log_path)
    needs_header = not path.exists() or path.stat().st_size == 0
    row = {
        "run_id": uuid.uuid4().hex[:12],
        "collected_at": utc_now(),
        "source": str(source),
        "status": str(status),
        "items_found": int(found),
        "items_saved": int(saved),
        "items_skipped": int(skipped),
        "error": str(error or "")[:500],
    }
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def run_source(collector, store, log_path, now=None) -> dict:
    timestamp = str((now or utc_now)())
    result = collector.collect()
    saved = 0
    skipped_duplicates = 0
    for item in result.items:
        if store.has_source_key(item.compute_source_key()):
            skipped_duplicates += 1
            continue
        store.append(
            item.to_record(captured_date=timestamp[:10], collected_at=timestamp)
        )
        saved += 1
    skipped_total = skipped_duplicates + result.irrelevant + result.non_text_entries
    status = "ok" if not result.errors else "partial_error"
    messages = [f"note: {note}" for note in getattr(result, "notes", [])]
    if result.errors:
        messages = list(result.errors) + messages
    append_run_log(
        log_path, collector.name, status, result.found, saved, skipped_total,
        "; ".join(messages),
    )
    return {
        "found": result.found,
        "saved": saved,
        "skipped": skipped_total,
        "errors": list(result.errors),
        "notes": list(getattr(result, "notes", [])),
    }
