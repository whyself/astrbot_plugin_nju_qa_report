from __future__ import annotations

from nju_report.aggregation import _aggregate
from nju_report.models import QuestionCandidate, StoredMessage


def test_conservative_aggregation_merges_duplicates_and_links_direct_reply() -> None:
    candidates = [
        _candidate(
            "20260712-Q001",
            "message:qq:bot:q1",
            "校园卡丢了怎么补办？",
            "南京大学校园卡丢失后如何补办？",
            "校园服务/校园卡",
            100,
        ),
        _candidate(
            "20260712-Q002",
            "message:qq:bot:q2",
            "校园卡挂失以后去哪补？",
            "南京大学校园卡丢失后如何补办",
            "校园服务/校园卡",
            120,
        ),
        _candidate(
            "20260712-Q003",
            "message:qq:bot:q3",
            "转专业怎么考？",
            "南京大学转专业需要参加哪些考核？",
            "学业与培养/转专业",
            150,
        ),
    ]
    messages = [
        _message("q1", 100, "u1", "校园卡丢了怎么补办？"),
        _message("q2", 120, "u2", "校园卡挂失以后去哪补？"),
        _message("a1", 125, "u3", "先在信息门户挂失，再去服务点。", reply="q2"),
        _message("q3", 150, "u4", "转专业怎么考？"),
    ]

    clusters = _aggregate(candidates, messages)

    assert len(clusters) == 2
    assert clusters[0].question_code == "20260712-Q001"
    assert clusters[0].occurrence_count == 2
    assert clusters[0].answers[0].direct_reply is True
    assert clusters[0].answers[0].redacted_text == "先在信息门户挂失，再去服务点。"
    assert clusters[1].question_code == "20260712-Q003"


def test_different_categories_do_not_merge_even_with_similar_text() -> None:
    candidates = [
        _candidate(
            "20260712-Q001",
            "message:qq:bot:q1",
            "怎么申请？",
            "南京大学如何申请宿舍？",
            "住宿",
            100,
        ),
        _candidate(
            "20260712-Q002",
            "message:qq:bot:q2",
            "怎么申请？",
            "南京大学如何申请宿舍？",
            "奖助",
            110,
        ),
    ]

    assert len(_aggregate(candidates, [])) == 2


def _candidate(
    code: str,
    source_key: str,
    original: str,
    canonical: str,
    category: str,
    sent_at: int,
) -> QuestionCandidate:
    return QuestionCandidate(
        question_code=code,
        source_key=source_key,
        report_date="2026-07-12",
        original_question=original,
        canonical_question=canonical,
        category=category,
        initial_decision="INCLUDE",
        final_decision="INCLUDE",
        reason="",
        confidence=0.9,
        status="RESOLVED",
        group_alias="南京大学迎新群",
        sent_at_utc=sent_at,
        created_at_utc=sent_at,
        updated_at_utc=sent_at,
    )


def _message(
    external_id: str,
    sent_at: int,
    sender: str,
    text: str,
    *,
    reply: str = "",
) -> StoredMessage:
    return StoredMessage(
        platform_id="qq",
        bot_self_id="bot",
        external_message_id=external_id,
        message_fingerprint=external_id,
        session_id="group:1",
        group_id="1",
        group_alias="南京大学迎新群",
        sender_id=sender,
        sender_name="",
        sent_at_utc=sent_at,
        text=text,
        outline="",
        reply_to_message_id=reply,
        analyzable=True,
    )
