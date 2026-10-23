"""TradingQ&A (Discourse) RSS collector — built offline, access-restricted live.

Only documented Discourse RSS feeds are eligible for collection. The source is
kept at requires_access, so the CLI refuses live runs until the site's
robots.txt and Terms of Service are verified to permit automated access.
No LLM is involved at any point.
"""

from __future__ import annotations

import os
import re
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.error import URLError
from xml.etree import ElementTree

from sip_engine.collectors.appstore import url_scheme, url_shape
from sip_engine.collectors.base import CollectedItem, CollectResult
from sip_engine.collectors.relevance import is_relevant, relevant_keywords


class FeedError(Exception):
    """The feed could not be retrieved as usable RSS XML."""


_BLOCK_TAGS = {"p", "div", "li", "blockquote", "br", "ul", "ol", "tr", "h1", "h2", "h3"}


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skipping += 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            if self._skipping:
                self._skipping -= 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data):
        if not self._skipping:
            self.chunks.append(data)


def html_to_text(html: str) -> str:
    """Strip HTML tags and unescape entities; wording is preserved verbatim."""
    extractor = _HtmlTextExtractor()
    extractor.feed(str(html or ""))
    extractor.close()
    text = "".join(extractor.chunks)
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def _normalise_pubdate(value: str) -> str:
    raw = str(value or "").strip()
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return raw
    if parsed is None:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _default_http_get(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "sip-insight-engine/0.1 (student research project)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


class TradingQnAAdapter:
    """Collects public TradingQ&A topics from configured Discourse RSS feeds."""

    name = "tradingqna"

    def __init__(self, http_get=None, feeds=None) -> None:
        self._http_get = http_get or _default_http_get
        raw_feeds = feeds if feeds is not None else os.environ.get("SIP_TQNA_FEEDS", "")
        self.feeds = tuple(f.strip() for f in str(raw_feeds).split(",") if f.strip())

    def require_config(self) -> None:
        if not self.feeds:
            raise ValueError(
                "Set SIP_TQNA_FEEDS to https://tradingqna.com/latest.rss (the "
                "approved pilot feed). robots.txt disallows /c/*.rss and "
                "/t/*/*.rss paths — never configure category or topic feeds; "
                "/tag/<name>.rss is permitted but out of this pilot's scope."
            )
        disallowed_prefixes = ("/c/", "/t/")
        for feed in self.feeds:
            lowered = feed.lower()
            for prefix in disallowed_prefixes:
                if lowered.startswith(f"https://tradingqna.com{prefix}") or lowered.startswith(f"https://www.tradingqna.com{prefix}"):
                    raise ValueError(
                        f"Disallowed feed configured (robots.txt forbids {prefix} "
                        f"paths): {feed[:60]}"
                    )

    @staticmethod
    def feed_label(feed_url: str) -> str:
        match = re.search(r"/(?:c|tag)/([a-z0-9-]+)", str(feed_url))
        return match.group(1) if match else "latest"

    def fetch_feed(self, feed_url: str) -> ElementTree.Element:
        text = self._http_get(feed_url)
        try:
            return ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            raise FeedError(f"{feed_url}: invalid RSS XML") from exc

    def parse_feed(self, entries, label: str) -> tuple[list[CollectedItem], list, list[str]]:
        items: list[CollectedItem] = []
        dropped: list = []
        reasons: list[str] = []
        for entry in entries:
            if not isinstance(entry, ElementTree.Element):
                dropped.append(entry)
                reasons.append("not_an_element")
                continue
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            description = html_to_text(entry.findtext("description") or "")
            creator = entry.find("{*}creator")
            author = (creator.text or "").strip() if creator is not None else ""
            if not (title or description):
                dropped.append(entry)
                reasons.append("missing_text")
                continue
            if not link.startswith(("http://", "https://")):
                dropped.append(entry)
                reasons.append("invalid_permalink")
                continue
            items.append(
                CollectedItem(
                    platform="tradingqna",
                    source_url=link,
                    raw_text="\n\n".join(part for part in (title, description) if part),
                    published_at=_normalise_pubdate((entry.findtext("pubDate") or "").strip()),
                    author_handle=author,
                    topic_tags=f"feed:{label}",
                )
            )
        return items, dropped, reasons

    @staticmethod
    def _element_fingerprint(item) -> str:
        return (
            "tags["
            + ", ".join(sorted({child.tag.split("}")[-1] for child in item}))
            + "]"
        )[:200]

    def collect(self) -> CollectResult:
        self.require_config()
        result = CollectResult()
        for feed_url in self.feeds:
            label = self.feed_label(feed_url)
            try:
                root = self.fetch_feed(feed_url)
            except FeedError as exc:
                result.errors.append(str(exc))
                continue
            except (URLError, OSError) as exc:
                result.errors.append(f"{feed_url}: request failed ({exc})")
                continue
            entries = list(root.iter("item"))
            result.raw_entries += len(entries)
            parsed, dropped, reasons = self.parse_feed(entries, label)
            result.non_text_entries += len(dropped)
            result.found += len(parsed)
            for item in parsed:
                if is_relevant(item.raw_text)[0]:
                    result.items.append(item)
                else:
                    result.irrelevant += 1
            if dropped:
                link_drops = [
                    entry for entry, reason in zip(dropped, reasons)
                    if reason == "invalid_permalink"
                ]
                text_drops = len(dropped) - len(link_drops)
                if text_drops:
                    result.notes.append(
                        f"{feed_url}: skipped {text_drops} items without usable text; "
                        f"structure: {self._element_fingerprint(dropped[0])}"
                    )
                if link_drops:
                    first_link = (link_drops[0].findtext("link") or "").strip()
                    result.notes.append(
                        f"{feed_url}: skipped {len(link_drops)} items with missing or "
                        f"invalid topic link (link.scheme={url_scheme(first_link)}, "
                        f"link.shape={url_shape(first_link)})"
                    )
            elif not parsed:
                result.notes.append(
                    f"{feed_url}: feed ok but no items (raw entries: 0)"
                )
        return result
