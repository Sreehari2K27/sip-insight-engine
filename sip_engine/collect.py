"""Collection CLI for SIP Insight Engine.

--dry-run reports configured source statuses only: no HTTP requests, no LLM,
no credentials, no writes. --source runs one approved, implemented adapter
(saving evidence and a run-log row); nothing else touches data files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from sip_engine.collectors.appstore import AppStoreAdapter
from sip_engine.collectors.base import APPROVED, SOURCES, STATUSES
from sip_engine.collectors.runner import append_run_log, run_source
from sip_engine.collectors.tradingqna import TradingQnAAdapter
from sip_engine.collectors.youtube import YoutubeCommentsAdapter
from sip_engine.evidence_store import EvidenceStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


_load_project_env()

EVIDENCE_PATH = PROJECT_ROOT / "data" / "evidence.csv"
LOG_PATH = PROJECT_ROOT / "data" / "collection_log.csv"

_ADAPTERS = {
    "appstore": lambda pages: AppStoreAdapter(pages=pages),
    "tradingqna": lambda pages: TradingQnAAdapter(),
    "youtube": lambda pages: YoutubeCommentsAdapter(max_pages=pages),
}


def dry_run() -> int:
    print(f"{'source':<12} {'status':<17} {'adapter':<10} requires")
    for name in sorted(SOURCES):
        spec = SOURCES[name]
        adapter_state = "yes" if spec.implemented else "not yet"
        requires = ", ".join(spec.requires) if spec.requires else "-"
        print(f"{spec.name:<12} {spec.status:<17} {adapter_state:<10} {requires}")
    print(
        f"\n{len(SOURCES)} sources configured, statuses limited to "
        f"{', '.join(STATUSES)}. Dry-run made no requests and wrote nothing."
    )
    return 0


def _run_source_cli(source: str, pages: int | None) -> int:
    spec = SOURCES[source]
    if not spec.implemented:
        print(f"Source '{source}' has no adapter implemented yet.")
        return 1
    if spec.status != APPROVED:
        print(f"Source '{source}' is marked '{spec.status}'; collection is not permitted.")
        return 1

    try:
        collector = _ADAPTERS[source](pages=pages)
        summary = run_source(collector, EvidenceStore(EVIDENCE_PATH), LOG_PATH)
    except ValueError as exc:
        append_run_log(LOG_PATH, source, "failed", 0, 0, 0, str(exc))
        print(f"Collection failed: {exc}")
        return 1

    print(
        f"{source}: found={summary['found']} saved={summary['saved']} "
        f"skipped={summary['skipped']} errors={len(summary['errors'])}"
    )
    for error in summary["errors"]:
        print(f"  warning: {error}")
    for note in summary.get("notes", []):
        print(f"  note: {note}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sip-engine collect",
        description="Multi-source collection CLI for SIP Insight Engine.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report configured source statuses only (no network, no LLM, no writes)",
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        help="run one approved, implemented source adapter",
    )
    parser.add_argument("--pages", type=int, default=None, help="App Store feed pages, 1-10")
    args = parser.parse_args(argv)

    if args.dry_run:
        return dry_run()
    if not args.source:
        print("Nothing to do: pass --dry-run or --source.")
        return 1
    return _run_source_cli(args.source, args.pages)


if __name__ == "__main__":
    raise SystemExit(main())
