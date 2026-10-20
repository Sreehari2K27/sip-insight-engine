"""Append-only CSV storage for SIP conversation evidence.

Traceability contract: ``raw_text`` and ``source_url`` are written exactly as
captured (commas, quotes, Unicode and embedded newlines included) and are never
rewritten or enriched in place. AI-derived fields live in separate columns.
"""

from __future__ import annotations

import csv
import uuid
from datetime import date
from pathlib import Path

COLUMNS = (
    "evidence_id",
    "captured_date",
    "platform",
    "source_url",
    "author_handle",
    "raw_text",
    "topic_tags",
    "ai_summary",
    "ai_theme",
    "confidence",
)

REQUIRED_COLUMNS = ("raw_text", "source_url")


class EvidenceStore:
    """Minimal append/read access to the evidence CSV."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(record) - set(COLUMNS))
        if unknown:
            raise ValueError(f"Unknown column(s): {unknown}")
        missing = [c for c in REQUIRED_COLUMNS if not str(record.get(c, "")).strip()]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

        stored: dict[str, str] = dict.fromkeys(COLUMNS, "")
        stored.update({k: str(v) for k, v in record.items()})
        stored["evidence_id"] = stored["evidence_id"] or uuid.uuid4().hex
        stored["captured_date"] = stored["captured_date"] or date.today().isoformat()

        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if needs_header:
                writer.writeheader()
            writer.writerow(stored)
        return stored

    def read_all(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError("Evidence CSV header does not match the expected schema")
            return list(reader)

    def count(self) -> int:
        return len(self.read_all())
