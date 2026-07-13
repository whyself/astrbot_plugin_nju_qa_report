from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from nju_report.aggregation import QuestionAggregationService, _aggregate
from nju_report.answer_agent import AnswerDiscoveryResult, DiscoveredQuestion
from nju_report.models import (
    CommunityAnswer,
    QuestionCandidate,
    QuestionCluster,
    ScopeAssessment,
    ScopeDecision,
    StoredMessage,
)
from nju_report.storage import ReportStorage


def test_conservative_aggregation_merges_duplicates_before_agent_answer_search() -> None:
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
    assert clusters[0].answers == ()
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


def test_answer_lookup_progress_counts_completed_clusters() -> None:
    class Storage:
        def list_question_candidates(self, *, report_date: str, limit):
            del report_date, limit
            return (
                [
                    _candidate(
                        "20260712-Q001",
                        "message:qq:bot:q1",
                        "校园卡怎么补办？",
                        "南京大学校园卡如何补办？",
                        "校园卡",
                        100,
                    ),
                    _candidate(
                        "20260712-Q002",
                        "message:qq:bot:q2",
                        "转专业怎么考？",
                        "南京大学转专业需要哪些考核？",
                        "转专业",
                        120,
                    ),
                ],
                2,
            )

        def messages_in_window(self, window):
            del window
            return [
                _message("q1", 100, "u1", "校园卡怎么补办？"),
                _message("q2", 120, "u2", "转专业怎么考？"),
            ]

        def save_question_clusters(self, report_date: str, clusters):
            del report_date, clusters

    class Agent:
        async def collect(self, cluster, messages):
            from nju_report.answer_agent import AnswerDiscoveryResult

            del messages
            await asyncio.sleep(0)
            external_id = cluster.candidate_source_keys[0].rsplit(":", 1)[-1]
            return AnswerDiscoveryResult((external_id,), ())

    async def run() -> None:
        service = QuestionAggregationService(  # type: ignore[arg-type]
            Storage(),
            Agent(),
            timezone_name="Asia/Shanghai",
            concurrency=1,
        )
        result = await service.aggregate_date(date(2026, 7, 12))
        assert len(result) == 2
        assert service.progress == ("2026-07-12", 2, 2)

    asyncio.run(run())


def test_answer_agent_repairs_overinclusive_question_sources_and_uses_summary() -> None:
    class Storage:
        saved: list[QuestionCluster] = []

        def list_question_candidates(self, *, report_date: str, limit):
            del report_date, limit
            return (
                [
                    _candidate(
                        "20260712-Q001",
                        "message:qq:bot:q1",
                        "[回复 王思喆] 校园卡丢了怎么办？",
                        "南京大学校园卡丢失后如何补办？",
                        "校园卡",
                        100,
                    ),
                    _candidate(
                        "20260712-Q002",
                        "message:qq:bot:a1",
                        "陶子秋说先挂失，再去服务点办理。",
                        "南京大学校园卡丢失后如何补办？",
                        "校园卡",
                        101,
                    ),
                ],
                2,
            )

        def messages_in_window(self, window):
            del window
            return [
                _message("q1", 100, "u1", "[回复 王思喆] 校园卡丢了怎么办？"),
                _message(
                    "a1",
                    101,
                    "u2",
                    "陶子秋说先挂失，再去服务点办理。",
                    reply="q1",
                ),
            ]

        def save_question_clusters(self, report_date: str, clusters):
            del report_date
            self.saved = list(clusters)

    class Agent:
        async def collect(self, cluster, messages):
            del cluster, messages
            return AnswerDiscoveryResult(
                ("q1",),
                (
                    CommunityAnswer(
                        external_message_id="summary:test",
                        redacted_text="应先挂失，再前往服务点办理。",
                        sent_at_utc=101,
                        confidence=0.95,
                        direct_reply=True,
                    ),
                ),
            )

    async def run() -> None:
        storage = Storage()
        service = QuestionAggregationService(  # type: ignore[arg-type]
            storage,
            Agent(),
            timezone_name="Asia/Shanghai",
            concurrency=1,
        )

        clusters = await service.aggregate_date(date(2026, 7, 12))

        assert len(clusters) == 1
        assert clusters[0].candidate_source_keys == ("message:qq:bot:q1",)
        assert clusters[0].representative_questions == (
            "南京大学校园卡丢失后如何补办？",
        )
        assert clusters[0].answers[0].redacted_text == "应先挂失，再前往服务点办理。"
        assert "王思喆" not in clusters[0].representative_questions[0]
        assert "陶子秋" not in clusters[0].answers[0].redacted_text
        assert storage.saved == clusters

    asyncio.run(run())


def test_answer_agent_can_split_one_overmerged_cluster_into_two_questions() -> None:
    class Storage:
        saved: list[QuestionCluster] = []

        def list_question_candidates(self, *, report_date: str, limit):
            del report_date, limit
            return (
                [
                    _candidate(
                        "20260712-Q001",
                        "message:qq:bot:q1",
                        "陶二条件怎么样？",
                        "陶二条件如何，大二能否住上翻新宿舍？",
                        "住宿食堂",
                        100,
                    ),
                    _candidate(
                        "20260712-Q002",
                        "message:qq:bot:q2",
                        "大二能住上翻新的宿舍吗？",
                        "陶二条件如何，大二能否住上翻新宿舍？",
                        "住宿食堂",
                        101,
                    ),
                ],
                2,
            )

        def messages_in_window(self, window):
            del window
            return [
                _message("q1", 100, "u1", "陶二条件怎么样？"),
                _message("q2", 101, "u1", "大二能住上翻新的宿舍吗？"),
            ]

        def save_question_clusters(self, report_date: str, clusters):
            del report_date
            self.saved = list(clusters)

    class Agent:
        async def collect(self, cluster, messages):
            del cluster, messages
            return AnswerDiscoveryResult(
                ("q1",),
                (),
                "陶二宿舍有哪些设施",
                "住宿食堂",
                (
                    DiscoveredQuestion(
                        ("q2",),
                        (),
                        "大二学生能否住上翻新后的宿舍",
                        "住宿食堂",
                    ),
                ),
            )

    async def run() -> None:
        storage = Storage()
        service = QuestionAggregationService(  # type: ignore[arg-type]
            storage,
            Agent(),
            timezone_name="Asia/Shanghai",
            concurrency=1,
        )

        clusters = await service.aggregate_date(date(2026, 7, 12))

        assert [item.question_code for item in clusters] == [
            "20260712-Q001",
            "20260712-Q002",
        ]
        assert [item.canonical_question for item in clusters] == [
            "陶二宿舍有哪些设施",
            "大二学生能否住上翻新后的宿舍",
        ]
        assert storage.saved == clusters

    asyncio.run(run())


def test_reaggregation_can_move_candidate_between_existing_clusters(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    assessment = ScopeAssessment(
        decision=ScopeDecision.INCLUDE,
        reason="校园公共问题",
        confidence=0.95,
        canonical_question="南京大学校园卡如何补办？",
        category="校园卡",
    )
    for index in (1, 2):
        storage.upsert_scope_candidate(
            source_key=f"message:qq:bot:q{index}",
            report_date="2026-07-12",
            initial=assessment,
            original_question=f"问题 {index}",
            group_alias="南京大学迎新群",
            sent_at_utc=100 + index,
        )
    candidates, _ = storage.list_question_candidates(
        report_date="2026-07-12",
        limit=None,
    )
    first, second = candidates
    storage.save_question_clusters(
        "2026-07-12",
        [
            _cluster(first.question_code, (first.source_key,), 101, "问题一"),
            _cluster(second.question_code, (second.source_key,), 102, "问题二"),
        ],
    )

    storage.save_question_clusters(
        "2026-07-12",
        [
            _cluster(
                first.question_code,
                (first.source_key, second.source_key),
                101,
                "合并后的问题",
            )
        ],
    )

    clusters = storage.list_question_clusters("2026-07-12")
    assert len(clusters) == 1
    assert clusters[0].candidate_source_keys == (first.source_key, second.source_key)
    storage.close()


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


def _cluster(
    code: str,
    source_keys: tuple[str, ...],
    sent_at: int,
    question: str,
) -> QuestionCluster:
    return QuestionCluster(
        question_code=code,
        report_date="2026-07-12",
        canonical_question=question,
        category="校园卡",
        candidate_source_keys=source_keys,
        representative_questions=(question,),
        group_aliases=("南京大学迎新群",),
        first_sent_at_utc=sent_at,
        last_sent_at_utc=sent_at,
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
