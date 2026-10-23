"""Apple App Store reviews collector (approved source, keyless).

Uses Apple's official iTunes customer-reviews RSS/JSON feed, one app id at a
time. Reviews are stored verbatim; a conservative, explainable keyword filter
keeps only SIP / first-time-investor-relevant items. No AI, no credentials.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from urllib.error import URLError

from sip_engine.collectors.base import CollectedItem, CollectResult
from sip_engine.collectors.relevance import RELEVANCE_PATTERNS, is_relevant, relevant_keywords


class FeedError(Exception):
    """The feed could not be retrieved as usable JSON."""


def _label(mapping, key) -> str:
    node = mapping.get(key) if isinstance(mapping, dict) else None
    if isinstance(node, dict):
        return str(node.get("label", "") or "")
    if isinstance(node, str):
        return node
    return ""


def _link_href(entry) -> str:
    """Safe extraction of the review permalink from link.attributes.href."""
    link = entry.get("link") if isinstance(entry, dict) else None
    attributes = link.get("attributes") if isinstance(link, dict) else None
    href = attributes.get("href") if isinstance(attributes, dict) else None
    return href if isinstance(href, str) else ""


def _feed_entries(payload) -> list:
    entries = ((payload or {}).get("feed") or {}).get("entry", [])
    if isinstance(entries, dict):
        return [entries]
    return list(entries or [])


def entry_fingerprint(entry) -> str:
    """Redacted structural preview: top-level key names and value types only.

    Never contains reviewer names, review text, URLs, ids or any other
    content data, and is capped at 200 characters.
    """
    if not isinstance(entry, dict):
        return f"non-dict:{type(entry).__name__}"
    parts = [f"{key}:{type(entry[key]).__name__}" for key in sorted(entry)]
    return ("keys[" + ", ".join(parts) + "]")[:200]


def field_structure(entry) -> str:
    """Inner-key/label-presence structure of the decisive review fields.

    Reports only inner key names, value types and whether each label is a
    non-empty string — never any field values.
    """
    parts = []
    for name in ("title", "content", "id"):
        node = entry.get(name) if isinstance(entry, dict) else None
        if isinstance(node, dict):
            inner = ",".join(sorted(node))
            label = node.get("label")
            label_ok = isinstance(label, str) and bool(label.strip())
            parts.append(f"{name}{{{inner};label:{'ok' if label_ok else 'missing'}}}")
        else:
            parts.append(f"{name}:{type(node).__name__}")
    return ("; ".join(parts))[:200]


def url_scheme(url) -> str:
    """Classify a permalink's scheme only; never returns the URL or content."""
    value = str(url or "").strip()
    if not value:
        return "none"
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", value)
    if match:
        scheme = match.group(1).lower()
        if scheme in ("http", "https", "itms", "itms-apps"):
            return scheme
        return "other"
    if value.startswith("//"):
        return "scheme-relative"
    return "none"


def url_shape(url) -> str:
    """Classify a rejected permalink's shape only; never returns the value."""
    value = str(url or "").strip()
    if not value:
        return "none"
    if value.isdigit():
        return "numeric"
    if value.startswith("/"):
        return "path"
    return "text"


def parse_entries(entries) -> tuple[list[CollectedItem], list, list[str]]:
    """Parse feed entries; returns (items, dropped, drop_reasons aligned to dropped).

    Apple's RSS uses ``id.label`` as a numeric review identifier and carries
    the review permalink in ``link.attributes.href`` — only the permalink is
    validated as source URL; identifiers are never rewritten into URLs.
    """
    items: list[CollectedItem] = []
    dropped: list = []
    reasons: list[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            dropped.append(entry)
            reasons.append("not_a_dict")
            continue
        title = _label(entry, "title").strip()
        body = _label(entry, "content").strip()
        permalink = _link_href(entry)
        if not (title or body):
            dropped.append(entry)
            reasons.append("missing_text")
            continue
        if not permalink.startswith(("http://", "https://")):
            dropped.append(entry)
            reasons.append("invalid_permalink")
            continue
        tags = []
        rating = _label(entry, "im:rating")
        if rating:
            tags.append(f"stars={rating}")
        items.append(
            CollectedItem(
                platform="appstore",
                source_url=permalink,
                raw_text="\n\n".join(part for part in (title, body) if part),
                published_at=_label(entry, "updated"),
                author_handle=_label(entry.get("author") or {}, "name"),
                topic_tags=";".join(tags),
            )
        )
    return items, dropped, reasons


def _default_http_get(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "sip-insight-engine/0.1 (student research project)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


class AppStoreAdapter:
    """Collects App Store reviews for configured app ids via the official RSS feed."""

    name = "appstore"

    def __init__(self, http_get=None, app_ids=None, country=None, pages=None) -> None:
        self._http_get = http_get or _default_http_get
        self.last_response_bytes = 0
        raw_ids = app_ids if app_ids is not None else os.environ.get("SIP_APPSTORE_APP_IDS", "")
        self.app_ids = []
        self.app_labels = []
        for part in str(raw_ids).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                app_id, label = part.split(":", 1)
                self.app_ids.append(app_id.strip())
                self.app_labels.append(label.strip())
            else:
                self.app_ids.append(part)
                self.app_labels.append("")
        self.app_ids = tuple(self.app_ids)
        self.app_labels = tuple(self.app_labels)
        self.country = str(country or os.environ.get("SIP_APPSTORE_COUNTRY", "in")).lower()
        self.pages = max(1, min(10, int(pages or os.environ.get("SIP_APPSTORE_PAGES", "1"))))

    def require_config(self) -> None:
        if not self.app_ids:
            raise ValueError(
                "Set SIP_APPSTORE_APP_IDS to a comma-separated list of numeric App "
                "Store app ids (from the apps.apple.com/…/id<number> URL)"
            )

    def feed_url(self, app_id: str, page: int) -> str:
        return (
            f"https://itunes.apple.com/{self.country}/rss/customerreviews/"
            f"page={page}/id={app_id}/sortby=mostrecent/json"
        )

    def fetch_feed(self, app_id: str, page: int) -> dict:
        text = self._http_get(self.feed_url(app_id, page))
        self.last_response_bytes = len(text.encode("utf-8"))
        try:
            return json.loads(text)
        except (TypeError, ValueError) as exc:
            raise FeedError(f"{app_id} page {page}: invalid JSON") from exc

    @staticmethod
    def parse_feed(payload) -> list[CollectedItem]:
        items, _, _ = parse_entries(_feed_entries(payload))
        return items

    def collect(self) -> CollectResult:
        self.require_config()
        result = CollectResult()
        for app_id, app_label in zip(self.app_ids, self.app_labels):
            for page in range(1, self.pages + 1):
                try:
                    payload = self.fetch_feed(app_id, page)
                except FeedError as exc:
                    result.errors.append(str(exc))
                    continue
                except (URLError, OSError) as exc:
                    result.errors.append(f"{app_id} page {page}: request failed ({exc})")
                    continue
                entries = _feed_entries(payload)
                result.raw_entries += len(entries)
                parsed, dropped, reasons = parse_entries(entries)
                if app_label:
                    for item in parsed:
                        item.topic_tags = (
                            f"{item.topic_tags};app={app_label}"
                            if item.topic_tags
                            else f"app={app_label}"
                        )
                result.non_text_entries += len(dropped)
                result.found += len(parsed)
                for item in parsed:
                    if is_relevant(item.raw_text, context=item.topic_tags)[0]:
                        result.items.append(item)
                    else:
                        result.irrelevant += 1
                if dropped:
                    url_drops = [
                        entry for entry, reason in zip(dropped, reasons)
                        if reason == "invalid_permalink"
                    ]
                    text_drops = len(dropped) - len(url_drops)
                    if text_drops:
                        result.notes.append(
                            f"{app_id} page {page}: skipped {text_drops} unusable entries; "
                            f"fields: {field_structure(dropped[0])}; "
                            f"structure: {entry_fingerprint(dropped[0])}"
                        )
                    if url_drops:
                        href = _link_href(url_drops[0])
                        scheme = url_scheme(href)
                        shape = url_shape(href)
                        result.notes.append(
                            f"{app_id} page {page}: skipped {len(url_drops)} entries "
                            f"with missing or invalid permalink "
                            f"(link.scheme={scheme}, link.shape={shape})"
                        )
                elif not parsed:
                    if isinstance(payload, dict):
                        top_level_keys = sorted(str(k) for k in payload.keys())
                        note = (
                            f"{app_id} page {page}: feed ok but no entries "
                            f"(raw entries: 0; response {self.last_response_bytes} bytes; "
                            f"top-level keys: {top_level_keys})"
                        )
                    else:
                        note = (
                            f"{app_id} page {page}: feed ok but no entries "
                            f"(raw entries: 0; response {self.last_response_bytes} bytes; "
                            f"top-level keys: not-an-object)"
                        )
                    result.notes.append(note)
        return result
