"""Offline tests for the TradingQ&A RSS collector (mocked HTTP only).

No network, no LLM calls. Live collection is blocked by the requires_access
source status; these tests also prove that refusal.
"""

import csv
import subprocess
import sys

import pytest

from sip_engine import collect as collect_module
from sip_engine.collectors.base import REQUIRES_ACCESS, SOURCES, CollectResult
from sip_engine.collectors.relevance import relevant_keywords
from sip_engine.collectors.runner import run_source
from sip_engine.collectors.tradingqna import (
    TradingQnAAdapter,
    html_to_text,
)
from sip_engine.evidence_store import EvidenceStore

FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Trading Q&amp;A</title>
    <item>
      <title>Should I pause my SIP during a market crash?</title>
      <link>https://tradingqna.com/t/pause-sip-during-crash/12345</link>
      <description>&lt;p&gt;I started my &lt;b&gt;first SIP&lt;/b&gt; last year &amp; am worried about the fall.&lt;/p&gt;&lt;p&gt;Should I stop it?&lt;/p&gt;</description>
      <dc:creator>researcher_ravi</dc:creator>
      <pubDate>Tue, 20 Oct 2026 09:00:00 +0530</pubDate>
    </item>
    <item>
      <title>Intraday margin rules?</title>
      <link>https://tradingqna.com/t/intraday-margin/12346</link>
      <description>&lt;p&gt;What are today&amp;#39;s intraday margins?&lt;/p&gt;</description>
      <dc:creator>daytrader</dc:creator>
      <pubDate>Mon, 19 Oct 2026 10:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

EMPTY_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Trading Q&amp;A</title></channel></rss>"""


def make_adapter(feeds="https://tradingqna.com/latest.rss", responses=None, **kwargs):
    calls = []

    def fake_http_get(url):
        calls.append(url)
        if callable(responses):
            return responses(url)
        if isinstance(responses, dict):
            return responses.get(url, "")
        return responses if responses is not None else FEED_XML

    adapter = TradingQnAAdapter(http_get=fake_http_get, feeds=feeds, **kwargs)
    return adapter, calls


def test_missing_config_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("SIP_TQNA_FEEDS", raising=False)
    adapter, calls = make_adapter(feeds=None)

    with pytest.raises(ValueError, match="SIP_TQNA_FEEDS"):
        adapter.collect()

    assert calls == []


def test_rss_parsing_normalises_discourse_items():
    adapter, calls = make_adapter()

    result = adapter.collect()

    assert result.found == 2
    assert len(result.items) == 1
    assert result.irrelevant == 1
    assert result.errors == []
    assert result.raw_entries == 2
    assert len(calls) == 1

    item = result.items[0]
    assert item.platform == "tradingqna"
    assert item.source_url == "https://tradingqna.com/t/pause-sip-during-crash/12345"
    assert item.raw_text == (
        "Should I pause my SIP during a market crash?\n\n"
        "I started my first SIP last year & am worried about the fall.\n"
        "Should I stop it?"
    )
    assert item.published_at == "2026-10-20T03:30:00+00:00"
    assert item.author_handle == "researcher_ravi"
    assert item.topic_tags == "feed:latest"
    assert relevant_keywords(item.raw_text) == ["sip"]


def test_html_cleanup_strips_tags_and_keeps_wording():
    assert html_to_text("<p>I started my <b>first SIP</b> &amp; paused.</p>") == (
        "I started my first SIP & paused."
    )
    assert html_to_text("<p>One</p><p>Two</p>") == "One\nTwo"
    assert html_to_text("<p>before <script>evil()</script>after</p>") == "before after"
    assert html_to_text("") == ""
    assert html_to_text(None) == ""


def test_pubdate_fallback_keeps_verbatim_on_garbage():
    adapter = TradingQnAAdapter(http_get=lambda url: FEED_XML, feeds="https://x/latest.rss")
    import xml.etree.ElementTree as ET

    root = ET.fromstring(FEED_XML)
    items, dropped, reasons = adapter.parse_feed(list(root.iter("item")), "latest")
    assert items[0].published_at == "2026-10-20T03:30:00+00:00"

    broken_pub = """<rss version="2.0"><channel><item>
        <title>T</title><link>https://tradingqna.com/t/x/1</link>
        <description>d</description><pubDate>not a date</pubDate></item></channel></rss>"""
    root2 = ET.fromstring(broken_pub)
    items2, _, _ = adapter.parse_feed(list(root2.iter("item")), "latest")
    assert items2[0].published_at == "not a date"


def test_feed_failure_isolation(tmp_path):
    good = "https://tradingqna.com/tag/mutual-funds.rss"
    bad = "https://tradingqna.com/latest.rss"
    adapter = TradingQnAAdapter(
        http_get=lambda url: FEED_XML if url == good else "<<<not xml>>>",
        feeds=f"{bad},{good}",
    )

    result = adapter.collect()

    assert result.errors == [f"{bad}: invalid RSS XML"]
    assert result.found == 2
    assert len(result.items) == 1
    assert result.items[0].topic_tags == "feed:mutual-funds"


def test_request_failure_isolation():
    def broken_http_get(url):
        if url.endswith("latest.rss"):
            raise OSError("network down")
        return FEED_XML

    adapter = TradingQnAAdapter(
        http_get=broken_http_get,
        feeds="https://tradingqna.com/latest.rss,https://tradingqna.com/tag/sip.rss",
    )

    result = adapter.collect()

    assert len(result.errors) == 1
    assert "request failed" in result.errors[0]
    assert result.found == 2


def test_empty_feed_zero_yield_note():
    adapter, _ = make_adapter(responses=EMPTY_FEED_XML)

    result = adapter.collect()

    assert result.found == 0
    assert result.raw_entries == 0
    assert result.errors == []
    assert len(result.notes) == 1
    assert "no items (raw entries: 0)" in result.notes[0]


def test_run_source_dedups_on_second_run(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    log = tmp_path / "collection_log.csv"
    adapter, _ = make_adapter()

    first = run_source(adapter, store, log, now=lambda: "2026-10-20T09:00:00+00:00")
    assert first["found"] == 2
    assert first["saved"] == 1
    assert first["skipped"] == 1

    second = run_source(adapter, store, log, now=lambda: "2026-10-20T10:00:00+00:00")
    assert second["saved"] == 0
    assert second["skipped"] == 2
    assert store.count() == 1

    rows = store.read_all()
    assert rows[0]["platform"] == "tradingqna"
    assert rows[0]["captured_date"] == "2026-10-20"
    assert rows[0]["collected_at"] == "2026-10-20T09:00:00+00:00"
    assert "SIP" in rows[0]["raw_text"]

    with log.open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))
    assert logged[0]["source"] == "tradingqna"
    assert logged[0]["status"] == "ok"
    assert len(logged) == 2


def test_registry_marks_tradingqna_approved_now():
    spec = SOURCES["tradingqna"]
    assert spec.status == "approved"
    assert spec.implemented is True
    assert "c/*.rss" in spec.method and "t/*/*.rss" in spec.method


def test_require_config_rejects_robots_disallowed_feeds():
    disallowed = "https://tradingqna.com/c/mutual-funds/32.rss"
    adapter = TradingQnAAdapter(http_get=lambda url: "", feeds=disallowed)

    with pytest.raises(ValueError, match="Disallowed"):
        adapter.require_config()

    disallowed_topic = TradingQnAAdapter(http_get=lambda url: "", feeds="https://tradingqna.com/t/some-slug/123.rss")
    with pytest.raises(ValueError, match="Disallowed"):
        disallowed_topic.require_config()

    ok = TradingQnAAdapter(http_get=lambda url: "", feeds="https://tradingqna.com/latest.rss")
    ok.require_config()


def test_help_text_never_suggests_disallowed_paths():
    adapter = TradingQnAAdapter(http_get=lambda url: "", feeds="")

    with pytest.raises(ValueError) as raised:
        adapter.require_config()

    message = str(raised.value)
    assert message.count("https://tradingqna.com/c/") == 0
    assert message.count("https://tradingqna.com/t/") == 0
    assert "https://tradingqna.com/latest.rss" in message
    assert "/tag/<name>.rss" not in message or "permitted" in message


def test_cli_requires_access_gate_refuses_restricted_sources(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")
    monkeypatch.setattr(collect_module, "EVIDENCE_PATH", tmp_path / "evidence.csv")
    from sip_engine.collectors.base import REQUIRES_ACCESS as _RA, SourceSpec

    fake_spec = SourceSpec("fake_restricted", _RA, "restricted test source", implemented=True)
    monkeypatch.setitem(collect_module.SOURCES, "fake_restricted", fake_spec)

    exit_code = collect_module.main(["--source", "fake_restricted"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "requires_access" in out
    assert "collection is not permitted" in out
    assert not (tmp_path / "log.csv").exists()
    assert not (tmp_path / "evidence.csv").exists()


def test_cli_refuses_unimplemented_playstore(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")

    exit_code = collect_module.main(["--source", "playstore"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "has no adapter implemented yet" in out
    assert not (tmp_path / "log.csv").exists()


def test_zero_llm_modules_loaded_dry_run():
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
