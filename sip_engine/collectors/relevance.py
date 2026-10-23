"""Shared conservative relevance filter for collected evidence.

Explainable by design: returns the exact keyword matches that made an item
relevant; an empty list means not relevant.
"""

from __future__ import annotations

import re

RELEVANCE_PATTERNS = (
    re.compile(r"\bsips?\b", re.IGNORECASE),
    re.compile(r"\bmutual\s+funds?\b", re.IGNORECASE),
    re.compile(r"\bsystematic\s+investment\b", re.IGNORECASE),
    re.compile(r"\bfirst\s+(?:investment|investor)\b", re.IGNORECASE),
    re.compile(r"\bmonthly\s+investment\b", re.IGNORECASE),
)


def relevant_keywords(text: str) -> list[str]:
    """Explainable relevance: the exact keywords that matched (empty = irrelevant)."""
    matched = []
    for pattern in RELEVANCE_PATTERNS:
        found = pattern.search(str(text or ""))
        if found:
            matched.append(found.group(0).lower())
    return matched


MIN_COMMENT_CHARS = 15

APP_EXPERIENCE_PATTERNS = (
    re.compile(
        r"\b(?:start(?:ed|ing|s)?|stop(?:ped|ping|s)?|paus(?:e|ed|es|ing)|quit|exit(?:ed|ing)?|"
        r"continu(?:e|es|ed|ing)|invest(?:ed|ing|s|ment)?|redeem(?:ed|ing)?|"
        r"withdraw(?:n|al|ing)?|sold|sell(?:ing)?|hold(?:ing)?|fail(?:ed|ing|s|ure|ure?s)?|payment(?:s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:los(?:t|ses|sing)|losses?|profits?|returns?|growth|savings?|mistakes?|"
        r"regret|worri(?:ed|es)|worry|afraid|fear|scared|confus(?:ed|ion)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:problems?|issues?|errors?|bugs?|crash(?:ed|ing|es)?|not\s+work(?:ing)?|"
        r"slow|hanging|hangs?|stuck|lag(?:ging|gy)?|kyc|otp|refund|charged|amount|money)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:band|rok|ruka|shuru|chalu|kar\s*diya|kr\s*di|krni|kar\s*raha|ho\s*raha|nahi)",
        re.IGNORECASE,
    ),
)


def _mentions_groww(text: str, context: str) -> bool:
    return bool(re.search(r"\bgroww\b", f"{text} {context}", re.IGNORECASE))


def has_concrete_app_experience(text: str) -> bool:
    """A concrete app experience/problem: actions, outcomes, problems, Hinglish."""
    return any(pattern.search(text) for pattern in APP_EXPERIENCE_PATTERNS)

SIGNAL_PATTERNS = (
    re.compile(
        r"\b(?:start(?:ed|ing|s)?|stop(?:ped|ping|s)?|paus(?:e|ed|es|ing)|quit|exit(?:ed|ing)?|"
        r"continu(?:e|es|ed|ing)|resum(?:e|ed|ing)|skip(?:ped|ping)?|reduc(?:e|ed|es|ing)|"
        r"lower(?:ed)?|cut(?:ting)?|invest(?:ed|ing|s|ment)?|redeem(?:ed|ing)?|"
        r"withdraw(?:n|al|ing)?|sold|sell(?:ing)?|hold(?:ing)?|fail(?:ed|ing|s)?|payment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:los(?:t|ses|sing)|losses?|profits?|returns?|growth|gains?|savings?|saved|"
        r"plan(?:ned|ning)?|mistakes?|regret|worri(?:ed|es)|worry|afraid|fear|scared|"
        r"confus(?:ed|ion)?|aide?|help|suggestion|advise?d?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:groww|zerodha|kite|autopay|mandate|upi|auto-?debit|app)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:band|rok|ruka|shuru|chalu|kar\s*diya|kr\s*di|krni|kar\s*raha|kaisa|kyun)",
        re.IGNORECASE,
    ),
)


def is_relevant(text: str, context: str = "") -> tuple[bool, str]:
    """Conservative collection gate: meaningful SIP/investing connection required.

    Returns (relevant, detail). Keyword path (unchanged): matched keyword +
    MIN_COMMENT_CHARS + a decision signal or a "?" (keyword-bearing beginner
    questions pass; vague questions without keywords were rejected by the
    keyword rule). Groww-app path (new): when there is no SIP/mutual-fund
    keyword, a review mentioning Groww — in the text or in adapter-supplied
    context tags like "app=groww" — is kept only if it describes a concrete
    app experience or problem; generic praise is rejected.
    """
    stripped = str(text or "").strip()
    matched = relevant_keywords(stripped)
    if matched:
        if len(stripped) < MIN_COMMENT_CHARS:
            return False, f"too short ({len(stripped)} chars)"
        if "?" in stripped:
            return True, "question + " + ", ".join(matched)
        if not any(signal.search(stripped) for signal in SIGNAL_PATTERNS):
            return False, "keyword-only mention (no SIP decision or app signal)"
        return True, ", ".join(matched) + " + signal"
    if _mentions_groww(stripped, str(context or "")):
        if len(stripped) < MIN_COMMENT_CHARS:
            return False, f"too short ({len(stripped)} chars)"
        if has_concrete_app_experience(stripped):
            return True, "groww app experience (no SIP/mutual-fund keyword)"
        return False, "generic praise without concrete app experience"
    return False, "no keyword match and no Groww app experience"
