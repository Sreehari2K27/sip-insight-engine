"""Collection groundwork: collected-item shape, source registry, statuses.

Only official APIs or documented feeds are ever eligible for collection
(`approved`). Sources needing credentials or approval (`requires_access`) and
sources with no affordable/permitted method (`unavailable`) are never fetched,
and public visibility is never treated as permission to scrape.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlparse, urlunparse

APPROVED = "approved"
REQUIRES_ACCESS = "requires_access"
UNAVAILABLE = "unavailable"
NOT_IMPLEMENTED = "not_implemented"

STATUSES = (APPROVED, REQUIRES_ACCESS, UNAVAILABLE, NOT_IMPLEMENTED)

LOG_COLUMNS = (
    "run_id",
    "collected_at",
    "source",
    "status",
    "items_found",
    "items_saved",
    "items_skipped",
    "error",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    status: str
    method: str
    requires: tuple[str, ...] = ()
    implemented: bool = False


SOURCES: dict[str, SourceSpec] = {
    "appstore": SourceSpec(
        "appstore",
        APPROVED,
        "Apple iTunes customer-reviews RSS/JSON feed (official, keyless)",
        implemented=True,
    ),
    "tradingqna": SourceSpec(
        "tradingqna",
        APPROVED,
        "TradingQ&A Discourse RSS — approved scope: the documented "
        "https://tradingqna.com/latest.rss feed (and /tag/<name>.rss feeds) "
        "only; robots.txt disallows /c/*.rss and /t/*/*.rss, so category and "
        "topic paths must never be configured or fetched",
        requires=(),
        implemented=True,
    ),
    "youtube": SourceSpec(
        "youtube",
        APPROVED,
        "YouTube Data API v3 commentThreads.list",
        requires=("YOUTUBE_API_KEY",),
        implemented=True,
    ),
    "blogs_rss": SourceSpec(
        "blogs_rss",
        APPROVED,
        "Google News and personal-finance blog RSS feeds",
    ),
    "reddit": SourceSpec(
        "reddit",
        REQUIRES_ACCESS,
        "Reddit Data API OAuth free tier (non-commercial; approval required)",
        requires=("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"),
    ),
    "playstore": SourceSpec(
        "playstore",
        REQUIRES_ACCESS,
        "Google Play Developer API (only for apps you own); "
        "unofficial scraping is excluded by policy",
        requires=("GOOGLE_PLAY_SERVICE_ACCOUNT",),
    ),
    "social_x": SourceSpec(
        "social_x",
        UNAVAILABLE,
        "X API pricing exceeds a student budget; no permitted free path",
    ),
}


def canonical_url(url: str) -> str:
    """Deterministic URL normalisation for dedup keys (scheme/host lowered,
    fragment dropped, utm_* parameters dropped, trailing slash trimmed)."""
    parts = urlparse(str(url).strip())
    query = "&".join(
        f"{key}={value}"
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    )
    return urlunparse(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", query, "")
    )


@dataclass
class CollectedItem:
    """One normalised public item, ready for the evidence schema."""

    platform: str
    source_url: str
    raw_text: str
    published_at: str = ""
    author_handle: str = ""
    topic_tags: str = ""

    def compute_source_key(self) -> str:
        canonical = canonical_url(self.source_url)
        return hashlib.sha256(f"{self.platform}|{canonical}".encode("utf-8")).hexdigest()

    def to_record(self, captured_date: str = "", collected_at: str = "") -> dict[str, str]:
        return {
            "platform": self.platform,
            "source_url": self.source_url,
            "raw_text": self.raw_text,
            "author_handle": self.author_handle,
            "topic_tags": self.topic_tags,
            "published_at": self.published_at,
            "captured_date": captured_date,
            "collected_at": collected_at,
            "source_key": self.compute_source_key(),
        }


@dataclass
class CollectResult:
    """Outcome of one collection pass over a source.

    found = usable reviews parsed; items = relevant subset; irrelevant =
    parsed but not SIP-relevant; raw_entries = entries present in the feeds;
    non_text_entries = raw entries dropped for missing review text;
    errors = genuine fetch/parse failures; notes = zero-yield diagnostics
    (a successful empty feed is a note, never an error).
    """

    found: int = 0
    items: list[CollectedItem] = field(default_factory=list)
    irrelevant: int = 0
    raw_entries: int = 0
    non_text_entries: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
