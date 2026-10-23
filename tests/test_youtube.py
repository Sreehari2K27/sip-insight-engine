"""Offline tests for the YouTube comments collector (mocked HTTP only).

No network, no LLM calls, and no real API key anywhere.
"""

import csv
import io
import json
import subprocess
import sys
import urllib.error

import pytest

from sip_engine import collect as collect_module
from sip_engine.collectors.base import APPROVED, SOURCES
from sip_engine.collectors.runner import run_source
from sip_engine.collectors.youtube import YoutubeCommentsAdapter
from sip_engine.evidence_store import EvidenceStore

API_KEY = "test-key-not-real"
VIDEO = "abc123"

REAL_KEY_SHAPED = "AIzaSyntheticKeyForTestsOnly123456"


def comment_thread(cid, text, author="Commenter", published="2026-10-01T10:00:00Z"):
    return {
        "id": cid,
        "snippet": {
            "topLevelComment": {
                "id": cid,
                "snippet": {
                    "textOriginal": text,
                    "authorDisplayName": author,
                    "publishedAt": published,
                    "likeCount": 3,
                },
            },
            "totalReplyCount": 0,
        },
    }


def thread_payload(*threads, next_token=None):
    payload = {"items": list(threads)}
    if next_token:
        payload["nextPageToken"] = next_token
    return payload


DEFAULT_COMMENTS = [
    comment_thread("c1", "I paused my SIP last month after the crash.", "Ravi", "2026-10-01T10:00:00Z"),
    comment_thread("c2", "Best video on mutual funds; SIP returns are clear.", "Asha", "2026-10-02T11:00:00Z"),
    comment_thread("c3", "Nice editing, subscribe everyone!", "Spammer", "2026-10-03T12:00:00Z"),
]


class FakeHttp:
    def __init__(self, page_payloads):
        self.payloads = page_payloads
        self.calls = []

    def __call__(self, url):
        index = min(len(self.calls), len(self.payloads) - 1)
        self.calls.append(url)
        current = self.payloads[index]
        if isinstance(current, Exception):
            raise current
        return current


def make_adapter(page_payloads=None, api_key=API_KEY, **kwargs):
    payloads = page_payloads if page_payloads is not None else [
        json.dumps(thread_payload(*DEFAULT_COMMENTS))
    ]
    fake = FakeHttp(payloads)
    adapter = YoutubeCommentsAdapter(http_get=fake, api_key=api_key, video_ids=VIDEO, **kwargs)
    return adapter, fake


def test_parse_maps_fields_exactly_and_filters_relevance():
    adapter, fake = make_adapter()

    result = adapter.collect()

    assert result.found == 3
    assert len(result.items) == 2
    assert result.irrelevant == 1
    assert result.errors == []
    assert result.raw_entries == 3
    assert len(fake.calls) == 1

    item = result.items[0]
    assert item.platform == "youtube"
    assert item.source_url == f"https://www.youtube.com/watch?v={VIDEO}&lc=c1"
    assert item.raw_text == "I paused my SIP last month after the crash."
    assert item.published_at == "2026-10-01T10:00:00Z"
    assert item.author_handle == "Ravi"
    assert item.topic_tags == f"video:{VIDEO}"

    request_url = fake.calls[0]
    assert "part=snippet" in request_url
    assert "replies" not in request_url
    assert f"videoId={VIDEO}" in request_url
    assert "maxResults=100" in request_url
    assert "order=time" in request_url
    assert "textFormat=plainText" in request_url


def test_pagination_follows_tokens_until_exhausted():
    page1 = json.dumps(thread_payload(
        comment_thread("c1", "My SIP paused after the crash.", "A", "2026-10-01T10:00:00Z"),
        next_token="TOKEN1",
    ))
    page2 = json.dumps(thread_payload(
        comment_thread("c2", "Mutual funds are my pick; I started a small monthly SIP.", "B", "2026-10-02T10:00:00Z"),
    ))
    adapter, fake = make_adapter([page1, page2], max_pages=5)

    result = adapter.collect()

    assert len(fake.calls) == 2
    assert "pageToken=TOKEN1" in fake.calls[1]
    assert result.found == 2
    assert len(result.items) == 2


def test_page_limit_stops_after_configured_pages():
    page1 = json.dumps(thread_payload(
        comment_thread("c1", "My SIP paused.", "A", "2026-10-01T10:00:00Z"),
        next_token="TOKEN1",
    ))
    page2 = json.dumps(thread_payload(
        comment_thread("c2", "Should not be fetched.", "B", "2026-10-02T10:00:00Z"),
    ))
    adapter, fake = make_adapter([page1, page2], max_pages=1)

    result = adapter.collect()

    assert len(fake.calls) == 1
    assert result.found == 1


def test_quota_budget_hard_stops_before_next_request():
    page1 = json.dumps(thread_payload(
        comment_thread("c1", "My SIP paused.", "A", "2026-10-01T10:00:00Z"),
        next_token="TOKEN1",
    ))
    page2 = json.dumps(thread_payload(
        comment_thread("c2", "Never fetched.", "B", "2026-10-02T10:00:00Z"),
    ))
    adapter, fake = make_adapter([page1, page2], quota_budget=1, max_pages=2)

    result = adapter.collect()

    assert len(fake.calls) == 1
    assert any("quota budget exhausted" in note for note in result.notes)
    assert result.found == 1


def test_missing_api_key_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    adapter, fake = make_adapter(api_key="")

    with pytest.raises(ValueError, match="YOUTUBE_API_KEY"):
        adapter.collect()

    assert fake.calls == []


def test_missing_video_ids_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("SIP_YOUTUBE_VIDEO_IDS", raising=False)
    adapter = YoutubeCommentsAdapter(http_get=lambda url: "", api_key=API_KEY, video_ids=None)

    with pytest.raises(ValueError, match="SIP_YOUTUBE_VIDEO_IDS"):
        adapter.collect()


def test_invalid_pages_and_budget_configuration(monkeypatch):
    monkeypatch.delenv("SIP_YOUTUBE_MAX_PAGES", raising=False)
    monkeypatch.delenv("SIP_YOUTUBE_QUOTA_BUDGET", raising=False)

    with pytest.raises(ValueError, match="SIP_YOUTUBE_MAX_PAGES"):
        YoutubeCommentsAdapter(http_get=lambda url: "", api_key=API_KEY, video_ids=VIDEO, max_pages="abc")
    with pytest.raises(ValueError, match="SIP_YOUTUBE_MAX_PAGES"):
        YoutubeCommentsAdapter(http_get=lambda url: "", api_key=API_KEY, video_ids=VIDEO, max_pages="0")
    with pytest.raises(ValueError, match="SIP_YOUTUBE_QUOTA_BUDGET"):
        YoutubeCommentsAdapter(http_get=lambda url: "", api_key=API_KEY, video_ids=VIDEO, quota_budget="abc")
    with pytest.raises(ValueError, match="SIP_YOUTUBE_QUOTA_BUDGET"):
        YoutubeCommentsAdapter(http_get=lambda url: "", api_key=API_KEY, video_ids=VIDEO, quota_budget="-5")


def http_error(code, reason):
    body = json.dumps({"error": {"code": code, "errors": [{"reason": reason}]}}).encode("utf-8")
    return urllib.error.HTTPError(f"https://www.googleapis.com/youtube/v3/commentThreads?key={REAL_KEY_SHAPED}", code, "Error", None, io.BytesIO(body))


def make_multi_video_adapter(first_behaviour, payload=None):
    calls = []
    second_video = "bbb111"
    response = payload if payload is not None else json.dumps(thread_payload(
        comment_thread("c9", "My SIP stopped after losses.", "D", "2026-10-04T10:00:00Z"),
    ))

    def fake_http_get(url):
        calls.append(url)
        if "videoId=aaa" in url:
            behaviour = first_behaviour
            if isinstance(behaviour, Exception):
                raise behaviour
            return behaviour
        return response

    adapter = YoutubeCommentsAdapter(
        http_get=fake_http_get,
        api_key=REAL_KEY_SHAPED,
        video_ids="aaa,bbb111",
    )
    return adapter, calls, second_video


def test_comments_disabled_is_a_note_and_other_videos_continue():
    payload = json.dumps(thread_payload(
        comment_thread("c9", "My SIP stopped after losses.", "D", "2026-10-04T10:00:00Z"),
    ))
    adapter, calls, second_video = make_multi_video_adapter(
        http_error(403, "commentsDisabled"), payload
    )

    result = adapter.collect()

    assert any("comments are disabled" in note for note in result.notes)
    assert result.errors == []
    assert result.found == 1
    assert any(f"videoId={second_video}" in url for url in calls)


def test_unavailable_video_is_an_error_and_other_videos_continue():
    payload = json.dumps(thread_payload(
        comment_thread("c9", "My SIP stopped after losses.", "D", "2026-10-04T10:00:00Z"),
    ))
    adapter, calls, _ = make_multi_video_adapter(
        http_error(404, "videoNotFound"), payload
    )

    result = adapter.collect()

    assert any("video unavailable (videoNotFound)" in err for err in result.errors)
    assert result.found == 1


def test_quota_exceeded_stops_entire_run():
    adapter, calls, _ = make_multi_video_adapter(
        http_error(403, "quotaExceeded")
    )

    result = adapter.collect()

    assert any("quota exceeded" in err for err in result.errors)
    assert len(calls) == 1


def test_generic_http_error_is_isolated():
    adapter, calls, _ = make_multi_video_adapter(
        http_error(500, "backendError")
    )

    result = adapter.collect()

    assert any("API error (backendError)" in err for err in result.errors)
    assert result.found == 1


def test_api_key_never_appears_in_errors_or_notes():
    leaking_failure = urllib.error.URLError(f"connection failed key={REAL_KEY_SHAPED}")
    adapter, calls, _ = make_multi_video_adapter(leaking_failure)

    result = adapter.collect()

    assert result.errors, "expected a captured error"
    assert all(REAL_KEY_SHAPED not in err for err in result.errors)
    assert all(REAL_KEY_SHAPED not in note for note in result.notes)


def test_run_source_dedups_on_second_run(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    log = tmp_path / "collection_log.csv"
    adapter, _ = make_adapter()

    first = run_source(adapter, store, log, now=lambda: "2026-10-20T09:00:00+00:00")
    assert first["found"] == 3
    assert first["saved"] == 2
    assert first["skipped"] == 1

    second = run_source(adapter, store, log, now=lambda: "2026-10-20T10:00:00+00:00")
    assert second["saved"] == 0
    assert second["skipped"] == 3
    assert store.count() == 2

    rows = store.read_all()
    assert rows[0]["platform"] == "youtube"
    assert rows[0]["source_url"].startswith(f"https://www.youtube.com/watch?v={VIDEO}&lc=")
    assert rows[0]["published_at"] == "2026-10-01T10:00:00Z"
    assert rows[0]["topic_tags"] == f"video:{VIDEO}"


def test_registry_marks_youtube_approved_and_implemented():
    spec = SOURCES["youtube"]
    assert spec.status == APPROVED
    assert spec.implemented is True


def test_cli_refuses_youtube_without_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("SIP_YOUTUBE_VIDEO_IDS", raising=False)
    monkeypatch.setattr(collect_module, "EVIDENCE_PATH", tmp_path / "evidence.csv")
    monkeypatch.setattr(collect_module, "LOG_PATH", tmp_path / "log.csv")

    exit_code = collect_module.main(["--source", "youtube"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Collection failed" in out
    assert "YOUTUBE_API_KEY" in out
    with (tmp_path / "log.csv").open("r", newline="", encoding="utf-8") as fh:
        logged = list(csv.DictReader(fh))
    assert logged[0]["status"] == "failed"
    assert not (tmp_path / "evidence.csv").exists()


def test_zero_llm_modules_loaded_with_youtube_collector():
    code = (
        "import sys\n"
        "import sip_engine.collectors.youtube\n"
        "loaded = {m for m in sys.modules if m.split('.')[0] in "
        "{'httpx', 'httpx2', 'mistralai', 'google', 'urllib3'}}\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
