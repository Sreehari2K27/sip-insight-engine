"""Append-only CSV storage for SIP conversation evidence.

Traceability contract: ``raw_text`` and ``source_url`` are written exactly as
captured (commas, quotes, Unicode and embedded newlines included) and are never
rewritten or enriched in place. AI-derived fields live in dedicated ``ai_*``
columns that only ``save_analysis`` may change, via atomic file replacement.

Schema compatibility: older CSVs (10-column original or 17-column without the
collection fields) are migrated additively and idempotently on first access;
any other header raises ValueError and leaves the file untouched.
"""

from __future__ import annotations

import csv
import os
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
    "ai_sentiment",
    "ai_motivation",
    "ai_behaviour",
    "ai_sip_status",
    "ai_discontinuation_reason",
    "ai_supporting_quote",
    "ai_uncertainty",
    "source_key",
    "published_at",
    "collected_at",
)

LEGACY_COLUMNS = COLUMNS[:10]

INTERIM_COLUMNS = COLUMNS[:17]

REQUIRED_COLUMNS = ("raw_text", "source_url")

ANALYSIS_TO_CSV = {
    "sentiment": "ai_sentiment",
    "motivation": "ai_motivation",
    "behaviour": "ai_behaviour",
    "problem_theme": "ai_theme",
    "sip_status": "ai_sip_status",
    "discontinuation_reason": "ai_discontinuation_reason",
    "supporting_quote": "ai_supporting_quote",
    "uncertainty": "ai_uncertainty",
}


class EvidenceStore:
    """Minimal append/read access to the evidence CSV, with analysis updates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, str]) -> dict[str, str]:
        self.migrate()
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
        self.migrate()
        if not self.path.exists():
            return []
        if self._read_header() != list(COLUMNS):
            raise ValueError("Evidence CSV header does not match the expected schema")
        return self._read_rows()

    def save_analysis(self, evidence_id: str, analysis: dict[str, str]) -> dict[str, str]:
        """Persist AI output onto an existing record without touching originals."""
        self.migrate()
        if not str(evidence_id).strip():
            raise ValueError("save_analysis requires an evidence_id")
        if "evidence_id" in analysis and str(analysis["evidence_id"]) != str(evidence_id):
            raise ValueError("analysis evidence_id does not match the target record")
        unknown_keys = sorted(set(analysis) - set(ANALYSIS_TO_CSV) - {"evidence_id"})
        if unknown_keys:
            raise ValueError(f"Unknown analysis field(s): {unknown_keys}")
        missing_keys = [k for k in ANALYSIS_TO_CSV if k not in analysis]
        if missing_keys:
            raise ValueError(f"Missing analysis field(s): {missing_keys}")

        rows = self.read_all()
        matches = [row for row in rows if row["evidence_id"] == str(evidence_id)]
        if not matches:
            raise ValueError(f"No evidence record with evidence_id={evidence_id!r}")
        if len(matches) > 1:
            raise ValueError(f"evidence_id={evidence_id!r} is not unique in the CSV")

        row = matches[0]
        for analysis_key, csv_column in ANALYSIS_TO_CSV.items():
            row[csv_column] = str(analysis[analysis_key])
        self._atomic_write(rows)
        return dict(row)

    def migrate(self) -> bool:
        """Add missing columns to legacy CSVs; True if the file changed."""
        header = self._read_header()
        if header is None:
            return False
        if header == list(COLUMNS):
            return False
        if header not in (list(LEGACY_COLUMNS), list(INTERIM_COLUMNS)):
            if self._header_is_newer(header):
                raise ValueError(
                    "The evidence CSV contains fields newer than this running "
                    f"process supports (file header has {len(header)} columns; "
                    f"the running code knows {len(COLUMNS)}). The code here is "
                    "older than the data: restart the Streamlit server or update "
                    "the running code, then reload."
                )
            raise ValueError(
                "Incompatible evidence CSV header: "
                f"{header}; expected the current schema or one of the legacy "
                f"schemas {list(LEGACY_COLUMNS)} / {list(INTERIM_COLUMNS)}"
            )
        self._atomic_write(self._read_rows())
        return True

    @staticmethod
    def _header_is_newer(header) -> bool:
        """True when the file header is a duplicate-free superset of COLUMNS."""
        return (
            len(header) == len(set(header))
            and len(header) > len(COLUMNS)
            and set(COLUMNS) <= set(header)
        )

    def count(self) -> int:
        return len(self.read_all())

    def has_source_key(self, source_key: str) -> bool:
        """True when a non-blank source_key is already stored (dedup check)."""
        if not str(source_key or "").strip():
            return False
        return any(row["source_key"] == str(source_key) for row in self.read_all())

    def _read_header(self) -> list[str] | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), None)

    def _read_rows(self) -> list[dict[str, str]]:
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [{c: row.get(c) or "" for c in COLUMNS} for row in reader]

    def _atomic_write(self, rows: list[dict[str, str]]) -> None:
        temp_path = self.path.parent / (self.path.name + ".tmp")
        try:
            with temp_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
