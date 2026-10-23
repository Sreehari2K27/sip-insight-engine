"""Offline tests for the Apple App Store RSS collector (mocked HTTP only).

Canonical Apple structure: id.label is a numeric review identifier and the
review permalink lives in link.attributes.href.
"""

import csv
import json

import pytest

from sip_engine.collectors.appstore import (
    AppStoreAdapter,
    FeedError,
    entry_fingerprint,
    field_structure,
    parse_entries,
    relevant_keywords,
    url_scheme,
    url_shape,
    _label,
    _link_href,
)
from sip_engine.collectors.runner import run_source
from sip_engine.evidence_store import EvidenceStore

GOOD_ENTRY = {
    "author": {"name": {"label": "Ravi"}},
    "im:rating": {"label": "5"},
    "title": {"label": "Good SIP app"},
    "content": {"label": "Started my first SIP here, monthly investment is easy."},
    "id": {"label": "100"},
    "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=100"}},
    "updated": {"label": "2026-10-01T10:00:00+05:30"},
}

IRRELEVANT_ENTRY = {
    "author": {"name": {"label": "Trader"}},
    "im:rating": {"label": "2"},
    "title": {"label": "Charts lag"},
    "content": {"label": "Intraday charts are slow to load."},
    "id": {"label": "200"},
    "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=200"}},
    "updated": {"label": "2026-10-02T10:00:00+05:30"},
}

MINIMAL_ENTRY = {
    "title": {"label": "SIP stopped working"},
    "content": {"label": "My SIP payment failed twice."},
    "id": {"label": "300"},
    "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=300"}},
    "updated": {"label": "2026-10-03T10:00:00+05:30"},
}

CANONICAL_APPLE_ENTRY = {
    "author": {"name": {"label": "Reviewer Name"}, "uri": {"label": "https://itunes.apple.com/in/profile"}},
    "im:rating": {"label": "4"},
    "im:version": {"label": "1.40"},
    "id": {"label": "777"},
    "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=777&type=Purple%20Software"}},
    "title": {"label": "Works well for my SIP"},
    "content": {"attributes": {"type": "text"}, "label": "I started my first SIP and it is simple."},
    "updated": {"label": "2026-10-05T10:00:00+05:30"},
    "im:contentType": {"attributes": {"term": "Application"}, "label": "Application"},
    "im:voteCount": {"label": "0"},
    "im:voteSum": {"label": "0"},
}


def make_payload(*entries):
    return {"feed": {"entry": list(entries)}}


def make_adapter(http_text=None, entries=None, **kwargs):
    if entries is None and http_text is None:
        http_text = ""
    if entries is not None:
        http_text = json.dumps(make_payload(*entries))
    default = {"app_ids": "111", "pages": 1}
    default.update(kwargs)
    calls = []

    def fake_http_get(url):
        calls.append(url)
        if callable(http_text):
            return http_text(url)
        return http_text

    adapter = AppStoreAdapter(http_get=fake_http_get, **default)
    return adapter, calls


def test_feed_url_format_page_one_and_two():
    adapter, _ = make_adapter()
    assert adapter.feed_url("123", 1) == (
        "https://itunes.apple.com/in/rss/customerreviews/page=1/id=123/sortby=mostrecent/json"
    )
    assert adapter.feed_url("123", 2) == (
        "https://itunes.apple.com/in/rss/customerreviews/page=2/id=123/sortby=mostrecent/json"
    )


def test_link_href_extraction_is_safe():
    assert _link_href({"link": {"attributes": {"href": "https://x/y"}}}) == "https://x/y"
    assert _link_href({"link": {"attributes": {}}}) == ""
    assert _link_href({"link": {}}) == ""
    assert _link_href({"link": "not-a-dict"}) == ""
    assert _link_href({}) == ""
    assert _link_href(None) == ""


def test_parse_feed_normalises_available_fields():
    items = AppStoreAdapter.parse_feed(make_payload(GOOD_ENTRY, MINIMAL_ENTRY))

    assert len(items) == 2
    first = items[0]
    assert first.platform == "appstore"
    assert first.source_url == "https://itunes.apple.com/in/review?id=100"
    assert first.raw_text == "Good SIP app\n\nStarted my first SIP here, monthly investment is easy."
    assert first.published_at == "2026-10-01T10:00:00+05:30"
    assert first.author_handle == "Ravi"
    assert first.topic_tags == "stars=5"
    second = items[1]
    assert second.author_handle == ""
    assert second.topic_tags == ""


def test_parse_feed_handles_empty_and_single_entry_feeds():
    assert AppStoreAdapter.parse_feed({}) == []
    assert AppStoreAdapter.parse_feed({"feed": {}}) == []
    single = AppStoreAdapter.parse_feed({"feed": {"entry": dict(GOOD_ENTRY)}})
    assert len(single) == 1


def test_parse_feed_skips_non_review_and_unlinkable_entries():
    app_self_entry = {"title": {"label": "Groww"}, "id": {"label": "111"}}
    no_href_entry = dict(GOOD_ENTRY, link={"attributes": {"rel": "alternate"}})

    items = AppStoreAdapter.parse_feed(make_payload(app_self_entry, no_href_entry, GOOD_ENTRY))

    assert len(items) == 1
    assert items[0].source_url == "https://itunes.apple.com/in/review?id=100"


def test_numeric_review_id_with_permalink_parses():
    items, dropped, reasons = parse_entries([dict(CANONICAL_APPLE_ENTRY)])

    assert dropped == []
    assert reasons == []
    assert len(items) == 1
    item = items[0]
    assert item.source_url == "https://itunes.apple.com/in/review?id=777&type=Purple%20Software"
    assert item.raw_text == "Works well for my SIP\n\nI started my first SIP and it is simple."
    assert item.published_at == "2026-10-05T10:00:00+05:30"
    assert item.author_handle == "Reviewer Name"
    assert item.topic_tags == "stars=4"
    assert relevant_keywords(item.raw_text) == ["sip"]
    assert item.compute_source_key().startswith("0") is False and len(item.compute_source_key()) == 64


def test_missing_or_malformed_permalink_drops_entry():
    no_link = dict(CANONICAL_APPLE_ENTRY)
    no_link.pop("link")
    attrs_missing = dict(CANONICAL_APPLE_ENTRY)
    attrs_missing["link"] = {"rel": "alternate"}
    href_none = dict(CANONICAL_APPLE_ENTRY)
    href_none["link"] = {"attributes": {"href": None}}
    ftp_href = dict(CANONICAL_APPLE_ENTRY)
    ftp_href["link"] = {"attributes": {"href": "ftp://example.com/review"}}

    items, dropped, reasons = parse_entries([no_link, attrs_missing, href_none, ftp_href])

    assert items == []
    assert len(dropped) == 4
    assert reasons == ["invalid_permalink"] * 4


def test_canonical_entry_collects_without_notes():
    adapter, _ = make_adapter(entries=[dict(CANONICAL_APPLE_ENTRY)])

    result = adapter.collect()

    assert result.found == 1
    assert len(result.items) == 1
    assert result.notes == []
    assert result.errors == []


def test_field_structure_is_redacted_and_decisive():
    structure = field_structure(dict(CANONICAL_APPLE_ENTRY))

    assert "id{label;label:ok}" in structure
    assert "title{label;label:ok}" in structure
    assert "content{attributes,label;label:ok}" in structure
    assert "Reviewer" not in structure
    assert "SIP" not in structure
    assert "777" not in structure
    assert "apps.apple" not in structure
    assert len(structure) <= 200


def test_field_structure_flags_missing_or_nonstring_labels():
    broken = {
        "title": {"attributes": {"x": "1"}},
        "content": {"label": None},
        "id": {"label": 12345},
    }

    structure = field_structure(broken)

    assert "title{attributes;label:missing}" in structure
    assert "content{label;label:missing}" in structure
    assert "id{label;label:missing}" in structure
    assert "12345" not in structure


def test_field_structure_handles_non_dict_fields():
    assert "title:str" in field_structure({"title": "plain"})
    assert "title:NoneType" in field_structure({})


def test_dropped_entry_note_includes_field_structure(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    log = tmp_path / "collection_log.csv"
    broken_canonical = dict(CANONICAL_APPLE_ENTRY)
    broken_canonical["title"] = {"attributes": {"type": "text"}}
    broken_canonical["content"] = {"attributes": {"type": "text"}}
    adapter, _ = make_adapter(entries=[broken_canonical])

    summary = run_source(adapter, store, log, now=lambda: "2026-10-20T09:00:00+05:30")

    assert summary["found"] == 0
    assert summary["skipped"] == 1
    assert "fields: title{attributes;label:missing}" in summary["notes"][0]
    assert "Reviewer" not in summary["notes"][0]


def test_relevance_filter_is_conservative_and_explainable():
    assert relevant_keywords("I paused my SIP last month") == ["sip"]
    assert relevant_keywords("SIPs are good") == ["sips"]
    assert relevant_keywords("best mutual fund app") == ["mutual fund"]
    assert relevant_keywords("My first investment journey") == ["first investment"]
    assert relevant_keywords("Great UI for stocks") == []
    assert relevant_keywords("Gasping for air") == []
    assert relevant_keywords("") == []


def test_label_handles_labelled_plain_null_and_unexpected_types():
    assert _label({"title": {"label": "Labelled"}}, "title") == "Labelled"
    assert _label({"title": "Plain string"}, "title") == "Plain string"
    assert _label({"title": None}, "title") == ""
    assert _label({"title": 42}, "title") == ""
    assert _label({"title": ["a", "b"]}, "title") == ""
    assert _label({}, "title") == ""
    assert _label(None, "title") == ""
    assert _label("not-a-dict", "title") == ""


def test_entry_fingerprint_is_structural_only():
    entry = {
        "author": {"name": {"label": "Ravi"}},
        "content": {"label": "Private review body about my SIP"},
        "id": {"label": "999"},
        "im:rating": {"label": "5"},
        "title": {"label": "Secret title"},
        "updated": {"label": "2026-10-01T10:00:00+05:30"},
    }

    fingerprint = entry_fingerprint(entry)

    assert fingerprint.startswith("keys[")
    assert len(fingerprint) <= 200
    assert "Ravi" not in fingerprint
    assert "Secret" not in fingerprint
    assert "Private" not in fingerprint
    assert "999" not in fingerprint
    assert "itunes" not in fingerprint
    assert "title:dict" in fingerprint
    assert "im:rating:dict" in fingerprint


def test_entry_fingerprint_handles_non_dict_entries():
    assert entry_fingerprint("weird") == "non-dict:str"
    assert entry_fingerprint(None) == "non-dict:NoneType"
    assert entry_fingerprint([1, 2]) == "non-dict:list"
    assert len(entry_fingerprint({"a" * 300: "x"})) <= 200


def test_url_scheme_classifier_covers_all_classes():
    assert url_scheme("https://itunes.apple.com/in/review?id=1") == "https"
    assert url_scheme("http://itunes.apple.com/in/review?id=1") == "http"
    assert url_scheme("//itunes.apple.com/in/review?id=1") == "scheme-relative"
    assert url_scheme("itms-apps://itunes.apple.com/in/review?id=1") == "itms-apps"
    assert url_scheme("itms://itunes.apple.com/in/review?id=1") == "itms"
    assert url_scheme("ftp://example.com/x") == "other"
    assert url_scheme("unknown-scheme:x") == "other"
    assert url_scheme("HTTPS://example.com/x") == "https"
    assert url_scheme("  https://example.com/x  ") == "https"
    assert url_scheme("1404871703") == "none"
    assert url_scheme("") == "none"
    assert url_scheme(None) == "none"
    assert url_scheme("ht tp://example.com") == "none"
    assert url_scheme("//no-scheme-check") == "scheme-relative"


def test_url_shape_classifier_covers_all_classes():
    assert url_shape("83628401") == "numeric"
    assert url_shape("1404871703") == "numeric"
    assert url_shape("  83628401  ") == "numeric"
    assert url_shape("/review?id=1") == "path"
    assert url_shape("//itunes.apple.com/in/review?id=1") == "path"
    assert url_shape("Purple Software") == "text"
    assert url_shape("HTTP://example.com/x") == "text"
    assert url_scheme("HTTP://example.com/x") == "http"
    assert url_shape("") == "none"
    assert url_shape("   ") == "none"
    assert url_shape(None) == "none"


def test_permalink_drop_note_contains_scheme_and_shape_only(tmp_path):
    no_permalink = dict(CANONICAL_APPLE_ENTRY)
    no_permalink.pop("link")
    adapter, _ = make_adapter(entries=[no_permalink])

    result = adapter.collect()

    assert result.found == 0
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "link.scheme=none, link.shape=none" in note
    assert "777" not in note
    assert "Reviewer" not in note
    assert "SIP" not in note
    assert "skipped 1 entries with missing or invalid permalink" in note


def test_bad_scheme_permalink_note_reports_scheme_class(tmp_path):
    ftp_permalink = dict(CANONICAL_APPLE_ENTRY)
    ftp_permalink["link"] = {"attributes": {"href": "ftp://example.com/review"}}
    adapter, _ = make_adapter(entries=[ftp_permalink])

    result = adapter.collect()

    assert result.found == 0
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "link.scheme=other, link.shape=text" in note
    assert "ftp://" not in note
    assert "example.com" not in note


def test_missing_text_note_has_no_permalink_claim(tmp_path):
    rating_only = {
        "im:rating": {"label": "4"},
        "id": {"label": "400"},
        "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=400"}},
        "updated": {"label": "2026-10-04T00:00:00+00:00"},
    }
    adapter, _ = make_adapter(entries=[rating_only])

    result = adapter.collect()

    assert len(result.notes) == 1
    assert "link.scheme" not in result.notes[0]
    assert "fields:" in result.notes[0]


def test_collect_filters_irrelevant_and_reports_counts():
    adapter, calls = make_adapter(entries=[GOOD_ENTRY, IRRELEVANT_ENTRY, MINIMAL_ENTRY])

    result = adapter.collect()

    assert result.found == 3
    assert len(result.items) == 2
    assert result.irrelevant == 1
    assert result.errors == []
    assert calls == [adapter.feed_url("111", 1)]


def test_collect_reports_malformed_json_and_continues():
    payload = json.dumps(make_payload(GOOD_ENTRY))

    def http_get(url):
        return "not json at all" if "id=111" in url else payload

    adapter = AppStoreAdapter(http_get=http_get, app_ids="111,222", pages=1)

    result = adapter.collect()

    assert result.found == 1
    assert len(result.items) == 1
    assert result.errors == ["111 page 1: invalid JSON"]


def test_groww_labelled_adapter_collects_concrete_complaints(tmp_path):
    complaint = {
        "title": {"label": "Payment issue"},
        "content": {"label": "Payment failed twice, no support response at all."},
        "id": {"label": "441"},
        "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=441"}},
        "updated": {"label": "2026-10-06T10:00:00+05:30"},
    }
    praise = {
        "title": {"label": "Beautiful"},
        "content": {"label": "Beautiful app, love it!"},
        "id": {"label": "442"},
        "link": {"attributes": {"href": "https://itunes.apple.com/in/review?id=442"}},
        "updated": {"label": "2026-10-06T11:00:00+05:30"},
    }

    def fake_http_get(url):
        return json.dumps(make_payload(complaint, praise))

    adapter = AppStoreAdapter(http_get=fake_http_get, app_ids="1404871703:groww", pages=1)

    result = adapter.collect()

    assert result.errors == []
    assert result.found == 2
    assert len(result.items) == 1
    assert result.irrelevant == 1
    assert result.items[0].topic_tags == "app=groww"

    summary = run_source(
        adapter, EvidenceStore(tmp_path / "evidence.csv"), tmp_path / "log.csv",
        now=lambda: "2026-10-21T09:00:00+00:00",
    )
    assert summary["saved"] == 1

    rows = EvidenceStore(tmp_path / "evidence.csv").read_all()
    assert rows[0]["topic_tags"] == "app=groww"
    assert rows[0]["raw_text"] == (
        "Payment issue\n\nPayment failed twice, no support response at all."
    )


def test_collect_reports_request_failures():
    def broken_http_get(url):
        raise OSError("network down")

    adapter = AppStoreAdapter(http_get=broken_http_get, app_ids="111", pages=1)

    result = adapter.collect()

    assert result.found == 0
    assert result.items == []
    assert len(result.errors) == 1
    assert "request failed" in result.errors[0]
    assert result.notes == []


def test_collect_missing_config_raises(monkeypatch):
    monkeypatch.delenv("SIP_APPSTORE_APP_IDS", raising=False)
    adapter = AppStoreAdapter(http_get=lambda url: "")

    with pytest.raises(ValueError, match="SIP_APPSTORE_APP_IDS"):
        adapter.collect()


def test_appstore_run_saves_dedups_and_logs(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.csv")
    log = tmp_path / "collection_log.csv"
    adapter, _ = make_adapter(entries=[GOOD_ENTRY, IRRELEVANT_ENTRY, MINIMAL_ENTRY])

    summary = run_source(adapter, store, log, now=lambda: "2026-10-20T09:00:00+05:30")

    assert summary["found"] == 3
    assert summary["saved"] == 2
    assert summary["skipped"] == 1

    rows = store.read_all()
    assert rows[0]["platform"] == "appstore"
    assert rows[0]["captured_date"] == "2026-10-20"
    assert rows[0]["collected_at"] == "2026-10-20T09:00:00+05:30"
    assert rows[0]["published_at"] == "2026-10-01T10:00:00+05:30"
    assert "SIP" in rows[0]["raw_text"]

    repeat = run_source(adapter, store, log, now=lambda: "2026-10-20T10:00:00+05:30")

    assert repeat["saved"] == 0
    assert repeat["skipped"] == 3
    assert store.count() == 2


OLD_EMPTY_FEED_PAYLOAD = {"feed": {"title": {"label": "iTunes Store: Customer Reviews"}}}


def test_zero_entry_note_reports_bytes_and_top_level_keys_only():
    """A successful fetch with zero entries logs a content-free fingerprint:
    byte length + top-level key names only, never values or body text."""
    def fake_http_get(url):
        return json.dumps(OLD_EMPTY_FEED_PAYLOAD)

    adapter = AppStoreAdapter(http_get=fake_http_get, app_ids="111", pages=1)

    result = adapter.collect()

    assert result.found == 0
    assert result.errors == []
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "no entries (raw entries: 0" in note
    expected_bytes = len(json.dumps(OLD_EMPTY_FEED_PAYLOAD).encode("utf-8"))
    assert f"response {expected_bytes} bytes" in note
    assert "top-level keys: ['feed']" in note
    assert "Customer Reviews" not in note
    assert "iTunes Store" not in note
