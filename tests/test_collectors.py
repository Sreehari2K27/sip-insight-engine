"""Offline tests for collection groundwork: registry, dedup keys, dry-run CLI.

No network, no LLM calls, no files written by the CLI.
"""

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from sip_engine import collect as collect_module
from sip_engine.collect import main as collect_main
from sip_engine.collectors.base import (
    APPROVED,
    LOG_COLUMNS,
    REQUIRES_ACCESS,
    SOURCES,
    STATUSES,
    UNAVAILABLE,
    CollectedItem,
)
from sip_engine.collectors.runner import append_run_log, run_source
from sip_engine.evidence_store import COLUMNS, EvidenceStore


def test_registry_statuses_are_valid_and_honest():
    assert set(STATUSES) == {"approved", "requires_access", "unavailable", "not_implemented"}
    assert all(spec.status in STATUSES for spec in SOURCES.values())
    assert SOURCES["appstore"].status == APPROVED
    assert SOURCES["tradingqna"].status == APPROVED
    assert SOURCES["youtube"].status == APPROVED
    assert SOURCES["playstore"].status == REQUIRES_ACCESS
    assert SOURCES["reddit"].status == REQUIRES_ACCESS
    assert SOURCES["social_x"].status == UNAVAILABLE
    implemented = {name for name, spec in SOURCES.items() if spec.implemented}
    assert implemented == {"appstore", "tradingqna", "youtube"}


def test_registry_declares_required_credentials():
    assert SOURCES["reddit"].requires == ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
    assert SOURCES["youtube"].requires == ("YOUTUBE_API_KEY",)
    assert not SOURCES["appstore"].requires


def test_source_key_is_stable_and_normalised():
    baseline = CollectedItem(
        platform="appstore",
        source_url="https://apps.apple.com/in/review?id=123",
        raw_text="x",
    ).compute_source_key()

    variants = CollectedItem(
        platform="appstore",
        source_url="HTTPS://Apps.Apple.com/in/review/?id=123&utm_source=rss#top",
        raw_text="different text",
    ).compute_source_key()

    assert variants == baseline
    assert baseline == hashlib.sha256(
        "appstore|https://apps.apple.com/in/review?id=123".encode("utf-8")
    ).hexdigest()


def test_different_platform_or_url_gives_different_key():
    item = CollectedItem(platform="appstore", source_url="https://example.com/a", raw_text="x")
    other_platform = CollectedItem(platform="youtube", source_url="https://example.com/a", raw_text="x")
    other_url = CollectedItem(platform="appstore", source_url="https://example.com/b", raw_text="x")

    assert len({item.compute_source_key(), other_platform.compute_source_key(), other_url.compute_source_key()}) == 3


def test_collected_item_maps_into_evidence_schema(tmp_path):
    item = CollectedItem(
        platform="appstore",
        source_url="https://apps.apple.com/in/review?id=123",
        raw_text="SIP \u0930\u0941\u0915 \u0917\u092f\u093e — ₹5,000, felt \"heavy\".",
        published_at="2026-10-19",
        author_handle="u/saver",
        topic_tags="groww;app",
    )

    record = item.to_record(captured_date="2026-10-20", collected_at="2026-10-20T09:00:00+05:30")

    assert set(record) <= set(COLUMNS)
    stored = EvidenceStore(tmp_path / "evidence.csv").append(record)

    assert stored["source_key"] == item.compute_source_key()
    assert stored["published_at"] == "2026-10-19"
    assert stored["collected_at"] == "2026-10-20T09:00:00+05:30"
    assert stored["raw_text"] == item.raw_text


def test_dry_run_reports_statuses_and_writes_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = collect_main(["--dry-run"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "appstore" in out and "approved" in out
    assert "requires_access" in out and "unavailable" in out
    assert "no requests and wrote nothing" in out
    assert list(tmp_path.iterdir()) == []


def test_dry_run_is_deterministic(capsys):
    collect_main(["--dry-run"])
    first = capsys.readouterr().out
    collect_main(["--dry-run"])
    second = capsys.readouterr().out

    assert first == second


def test_without_dry_run_flag_refuses_and_does_nothing(capsys):
    exit_code = collect_main([])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Nothing to do: pass --dry-run or --source." in out


def test_dry_run_process_never_loads_network_or_llm_modules():
    code = (
        "import sys\n"
        "from sip_engine.collect import main\n"
        "assert main(['--dry-run']) == 0\n"
        "loaded = {m for m in sys.modules if m.split('.')[0] in "
        "{'httpx', 'httpx2', 'requests', 'mistralai', 'google', 'urllib3'}}\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr


def test_cli_loads_dotenv_from_project_root(tmp_path, monkeypatch):
    import os

    (tmp_path / ".env").write_text("SIP_APPSTORE_APP_IDS=9876543210\n", encoding="utf-8")
    monkeypatch.delenv("SIP_APPSTORE_APP_IDS", raising=False)
    monkeypatch.setattr(collect_module, "PROJECT_ROOT", tmp_path)

    collect_module._load_project_env()

    assert os.environ.get("SIP_APPSTORE_APP_IDS") == "9876543210"
    assert not (tmp_path / "data").exists()

    os.environ.pop("SIP_APPSTORE_APP_IDS", None)


def test_collection_log_header_matches_schema():
    path = Path(__file__).resolve().parents[1] / "data" / "collection_log.csv"
    with path.open("r", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), None)

    assert header == list(LOG_COLUMNS)


class FakeCollector:
    name = "fake"

    def __init__(self, items, found=None, irrelevant=0, errors=None):
        self.items = items
        self.found = found if found is not None else len(items)
        self.irrelevant = irrelevant
        self.errors = errors or []

    def collect(self):
        from sip_engine.collectors.base import CollectResult

        return CollectResult(
            found=self.found, items=self.items, irrelevant=self.irrelevant, errors=self.errors
        )


def test_has_source_key(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")

    assert store.has_source_key("abc") is False

    store.append(make_stored_item("abc"))
    store.append(make_stored_item(""))

    assert store.has_source_key("abc") is True
    assert store.has_source_key("") is False
    assert store.has_source_key("other") is False


def make_stored_item(key):
    record = CollectedItem(
        platform="appstore",
        source_url=f"https://example.com/review?id={key or 'none'}",
        raw_text="I paused my SIP.",
    ).to_record(captured_date="2026-10-20", collected_at="2026-10-20T09:00:00+00:00")
    record["source_key"] = key
    return record


def test_run_source_saves_dedups_and_logs(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    log = tmp_path / "collection_log.csv"
    item_a = CollectedItem(platform="appstore", source_url="https://example.com/a", raw_text="SIP review one")
    item_b = CollectedItem(platform="appstore", source_url="https://example.com/b", raw_text="SIP review two")
    collector = FakeCollector([item_a, item_b], found=5, irrelevant=1)

    summary = run_source(collector, store, log, now=lambda: "2026-10-20T09:00:00+00:00")

    assert summary["found"] == 5
    assert summary["saved"] == 2
    assert summary["skipped"] == 1

    again = run_source(collector, store, log, now=lambda: "2026-10-20T10:00:00+00:00")

    assert again["saved"] == 0
    assert again["skipped"] == 3

    rows = store.read_all()
    assert len(rows) == 2
    assert rows[0]["captured_date"] == "2026-10-20"
    assert rows[0]["collected_at"] == "2026-10-20T09:00:00+00:00"
    assert rows[0]["source_key"] == item_a.compute_source_key()

    with log.open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))

    assert len(logged) == 2
    assert logged[0]["source"] == "fake"
    assert logged[0]["status"] == "ok"
    assert logged[0]["items_found"] == "5"
    assert logged[0]["items_saved"] == "2"
    assert logged[1]["items_saved"] == "0"
    assert logged[1]["items_skipped"] == "3"


def test_run_source_logs_partial_errors(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    log = tmp_path / "collection_log.csv"
    item = CollectedItem(platform="appstore", source_url="https://example.com/a", raw_text="SIP review")
    collector = FakeCollector([item], errors=["111 page 1: invalid JSON"])

    summary = run_source(collector, store, log, now=lambda: "2026-10-20T09:00:00+00:00")

    assert summary["errors"] == ["111 page 1: invalid JSON"]

    with log.open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))

    assert logged[0]["status"] == "partial_error"
    assert "invalid JSON" in logged[0]["error"]


def test_append_run_log_truncates_long_errors(tmp_path):
    row = append_run_log(tmp_path / "log.csv", "appstore", "failed", 0, 0, 0, "x" * 600)

    assert len(row["error"]) == 500

    with (tmp_path / "log.csv").open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))

    assert len(logged) == 1
    assert logged[0]["run_id"]


def test_cli_refuses_unimplemented_source(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")

    exit_code = collect_main(["--source", "blogs_rss"])

    assert exit_code == 1
    assert not (tmp_path / "log.csv").exists()


def test_cli_refuses_non_approved_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")

    exit_code = collect_main(["--source", "playstore"])

    assert exit_code == 1
    assert "has no adapter implemented yet" in capsys.readouterr().out
    assert not (tmp_path / "log.csv").exists()


def test_cli_missing_app_ids_logs_failure_without_evidence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect_module, "EVIDENCE_PATH", tmp_path / "evidence.csv")
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")
    monkeypatch.delenv("SIP_APPSTORE_APP_IDS", raising=False)

    exit_code = collect_module.main(["--source", "appstore"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Collection failed" in out
    with (tmp_path / "log.csv").open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))
    assert logged[0]["status"] == "failed"
    assert "SIP_APPSTORE_APP_IDS" in logged[0]["error"]
    assert not (tmp_path / "evidence.csv").exists()


def test_cli_prints_notes_and_warnings_distinctly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect_module, "EVIDENCE_PATH", tmp_path / "evidence.csv")
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")

    class FakeAdapter:
        name = "appstore"

        def __init__(self, *args, **kwargs):
            pass

        def collect(self):
            from sip_engine.collectors.base import CollectResult

            return CollectResult(
                found=1,
                errors=["111 page 1: request failed (boom)"],
                notes=["1404871703 page 1: feed ok but no entries (raw entries: 0)"],
            )

    monkeypatch.setattr(collect_module, "AppStoreAdapter", FakeAdapter)

    exit_code = collect_module.main(["--source", "appstore"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "warning: 111 page 1: request failed (boom)" in out
    assert "note: 1404871703 page 1: feed ok but no entries" in out
    with (tmp_path / "log.csv").open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))
    assert logged[0]["status"] == "partial_error"
    assert "request failed" in logged[0]["error"]
    assert "note:" in logged[0]["error"]
