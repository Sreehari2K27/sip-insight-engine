"""YouTube comments collector (approved source; API-key, quota-budgeted).

Uses the official YouTube Data API v3 ``commentThreads.list`` with
``part=snippet`` (top-level comments only) and ``textFormat=plainText`` so
comment text is preserved verbatim from ``textOriginal``. The shared
relevance filter, dedup and run-log pipeline are reused. The API key is read
from the environment only and never logged: diagnostics name the video id
and error reason, never the request URL (which carries the key).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from sip_engine.collectors.base import CollectedItem, CollectResult
from sip_engine.collectors.relevance import is_relevant, relevant_keywords

API_BASE = "https://www.googleapis.com/youtube/v3/commentThreads"


def _error_reason(exc) -> str:
    """Extract Google's error reason from an HTTPError response body, safely."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        errors = payload.get("error", {}).get("errors", [])
        if errors:
            return str(errors[0].get("reason", "") or "")
        return str(payload.get("error", {}).get("status", "") or "")
    except (AttributeError, TypeError, ValueError):
        return ""


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


class YoutubeCommentsAdapter:
    """Collects top-level comments for configured video IDs, quota-budgeted."""

    name = "youtube"

    def __init__(self, http_get=None, api_key=None, video_ids=None, max_pages=None, quota_budget=None) -> None:
        self._api_key = str(api_key if api_key is not None else os.environ.get("YOUTUBE_API_KEY", ""))
        raw_ids = video_ids if video_ids is not None else os.environ.get("SIP_YOUTUBE_VIDEO_IDS", "")
        self.video_ids = tuple(v.strip() for v in str(raw_ids).split(",") if v.strip())
        try:
            self.max_pages = int(max_pages if max_pages is not None else os.environ.get("SIP_YOUTUBE_MAX_PAGES", "1"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SIP_YOUTUBE_MAX_PAGES: {exc}") from exc
        try:
            self.quota_budget = int(quota_budget if quota_budget is not None else os.environ.get("SIP_YOUTUBE_QUOTA_BUDGET", "100"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SIP_YOUTUBE_QUOTA_BUDGET: {exc}") from exc
        if self.max_pages < 1:
            raise ValueError("SIP_YOUTUBE_MAX_PAGES must be 1 or more")
        if self.quota_budget < 1:
            raise ValueError("SIP_YOUTUBE_QUOTA_BUDGET must be 1 or more")
        self._http_get = http_get or _default_http_get

    def require_config(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Set YOUTUBE_API_KEY in your .env file; the key is read from "
                "the environment and never logged."
            )
        if not self.video_ids:
            raise ValueError(
                "Set SIP_YOUTUBE_VIDEO_IDS to a comma-separated list of YouTube video IDs"
            )

    def _request_url(self, video_id: str, page_token: str) -> str:
        params = [
            ("part", "snippet"),
            ("videoId", video_id),
            ("maxResults", "100"),
            ("order", "time"),
            ("textFormat", "plainText"),
            ("key", self._api_key),
        ]
        if page_token:
            params.append(("pageToken", page_token))
        return f"{API_BASE}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def parse_response(payload, video_id: str) -> tuple[list[CollectedItem], list, list[str]]:
        threads = (payload or {}).get("items") or []
        items: list[CollectedItem] = []
        dropped: list = []
        reasons: list[str] = []
        for thread in threads:
            if not isinstance(thread, dict):
                dropped.append(thread)
                reasons.append("not_an_object")
                continue
            comment = (thread.get("snippet") or {}).get("topLevelComment") or {}
            comment_snippet = comment.get("snippet") or {}
            comment_id = str(thread.get("id") or comment.get("id") or "").strip()
            text = str(comment_snippet.get("textOriginal", "") or "")
            if not text.strip():
                dropped.append(thread)
                reasons.append("missing_text")
                continue
            items.append(
                CollectedItem(
                    platform="youtube",
                    source_url=f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
                    raw_text=text,
                    published_at=str(comment_snippet.get("publishedAt", "") or ""),
                    author_handle=str(comment_snippet.get("authorDisplayName", "") or ""),
                    topic_tags=f"video:{video_id}",
                )
            )
        return items, dropped, reasons

    def _redact(self, text: str) -> str:
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "***")
        return text

    def collect(self) -> CollectResult:
        self.require_config()
        result = CollectResult()
        units_used = 0
        for video_id in self.video_ids:
            page_token = ""
            pages_fetched = 0
            while pages_fetched < self.max_pages:
                if units_used >= self.quota_budget:
                    result.notes.append(
                        f"quota budget exhausted ({units_used} request unit(s) used); stopping run"
                    )
                    return result
                units_used += 1
                pages_fetched += 1
                try:
                    text = self._http_get(self._request_url(video_id, page_token))
                except urllib.error.HTTPError as exc:
                    reason = _error_reason(exc)
                    if reason == "quotaExceeded":
                        result.errors.append(f"video {video_id}: YouTube quota exceeded; stopping run")
                        return result
                    if reason == "commentsDisabled":
                        result.notes.append(f"video {video_id}: comments are disabled; skipped")
                        break
                    if reason in ("videoNotFound", "videoIsPrivate"):
                        result.errors.append(f"video {video_id}: video unavailable ({reason})")
                        break
                    result.errors.append(self._redact(f"video {video_id}: API error ({reason or 'unknown'})"))
                    break
                except (urllib.error.URLError, OSError) as exc:
                    result.errors.append(self._redact(f"video {video_id}: request failed ({exc})"))
                    break
                try:
                    payload = json.loads(text)
                except (TypeError, ValueError):
                    result.errors.append(self._redact(f"video {video_id}: invalid JSON response"))
                    break
                parsed, dropped, reasons = self.parse_response(payload, video_id)
                result.raw_entries += len(parsed) + len(dropped)
                result.non_text_entries += len(dropped)
                result.found += len(parsed)
                for item in parsed:
                    if is_relevant(item.raw_text)[0]:
                        result.items.append(item)
                    else:
                        result.irrelevant += 1
                if dropped and not parsed:
                    result.notes.append(
                        f"video {video_id}: response ok but no usable comments "
                        f"(raw entries: {len(dropped)}, without text: {len(dropped)})"
                    )
                page_token = str((payload or {}).get("nextPageToken") or "")
                if not page_token:
                    break
        return result
