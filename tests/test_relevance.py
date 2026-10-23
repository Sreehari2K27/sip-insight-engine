"""Focused offline tests for the shared relevance gate (no network, no LLM)."""

from sip_engine.collectors.relevance import (
    MIN_COMMENT_CHARS,
    is_relevant,
    relevant_keywords,
)


def test_generic_short_sip_comment_is_rejected():
    relevant, detail = is_relevant("SIP")

    assert relevant is False
    assert "too short" in detail
    assert "15" in detail or "3" in detail


def test_useful_keyword_comment_is_accepted():
    relevant, detail = is_relevant("I paused my SIP last month after the crash.")

    assert relevant is True
    assert detail.startswith("sip")


def test_short_hinglish_comment_is_accepted():
    """Hinglish stop-statements keep the Latin word 'SIP', so the shared
    keyword rule catches them; at 17 chars they clear the 15-char gate."""
    relevant, detail = is_relevant("SIP band kar diya")

    assert relevant is True
    assert detail.startswith("sip")


def test_length_boundary_behaviour():
    fourteen_chars = "SIP stopped no"
    fifteen_chars = "SIP stopped now"

    assert len(fourteen_chars) == 14
    assert len(fifteen_chars) == MIN_COMMENT_CHARS == 15
    assert is_relevant(fourteen_chars)[0] is False
    assert is_relevant(fifteen_chars)[0] is True


def test_no_keyword_match_is_rejected_regardless_of_length():
    relevant, detail = is_relevant("Nice video, please subscribe!")

    assert relevant is False
    assert "no keyword match" in detail


def test_keyword_rules_are_preserved_by_the_extractor():
    assert relevant_keywords("SIP") == ["sip"]
    assert relevant_keywords("best mutual fund app") == ["mutual fund"]
    assert relevant_keywords("Great UI for stocks") == []


def test_keyword_only_mention_is_rejected():
    relevant, detail = is_relevant("SIP, SIP, SIP, SIP, SIP!")

    assert relevant is False
    assert "keyword-only" in detail


def test_generic_praise_is_rejected():
    relevant, detail = is_relevant("Great video on SIPs, mind blowing!")

    assert relevant is False
    assert "keyword-only" in detail


def test_greeting_with_keyword_is_rejected():
    relevant, detail = is_relevant("Hello friends, SIP is in trend!")

    assert relevant is False
    assert "keyword-only" in detail


def test_beginner_question_is_accepted():
    relevant, detail = is_relevant("Which SIP should I start in Groww for long term?")

    assert relevant is True
    assert "question" in detail


def test_groww_app_experience_is_accepted():
    relevant, detail = is_relevant("Groww app SIP autopay failed twice this month.")

    assert relevant is True
    assert "autopay" not in detail
    assert "sip" in detail


def test_decision_and_outcome_comments_are_accepted():
    assert is_relevant("I stopped my SIP because of losses last year.")[0] is True
    assert is_relevant("Reduce my SIP amount before it hurts my salary.")[0] is True


def test_hinglish_still_accepted_as_before():
    """Hinglish stop/start statements keep the Latin word 'SIP', so the shared
    keyword rule catches them; at 17 chars they clear the 15-char gate."""
    assert is_relevant("SIP band kar diya")[0] is True
    assert is_relevant("Mujhe 500 ki sip start krni h")[0] is True


def test_groww_app_complaint_without_sip_is_accepted():
    relevant, detail = is_relevant("Groww app crashes whenever I try to withdraw money.")

    assert relevant is True
    assert "groww app experience" in detail


def test_detailed_hinglish_app_experience_is_accepted():
    relevant, detail = is_relevant(
        "Update ke baad Groww app hang ho raha hai, payment me problem aa rahi."
    )

    assert relevant is True
    assert "groww app experience" in detail


def test_generic_praise_with_groww_is_rejected():
    relevant, detail = is_relevant("Groww is the best app ever, love it!")

    assert relevant is False
    assert "generic praise" in detail


def test_non_groww_app_complaint_is_rejected():
    relevant, detail = is_relevant("Zomato app crashes every single day.")

    assert relevant is False
    assert "no keyword match" in detail


def test_short_groww_complaint_is_rejected_by_length():
    relevant, detail = is_relevant("Groww app bug!")

    assert relevant is False
    assert "too short" in detail


def test_context_tag_enables_groww_path_without_brand_word():
    relevant, detail = is_relevant(
        "App crashed during payment, money not credited.",
        context="stars=1;app=groww",
    )

    assert relevant is True
    assert "groww app experience" in detail
    assert is_relevant(
        "App crashed during payment, money not credited.", context="stars=1;app=other"
    )[0] is False
