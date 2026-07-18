from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from nju_report.answer_agent import (
    AstrBotContextAnswerAgent,
    ChatContextLookup,
    _AnswerAssessment,
)
from nju_report.models import (
    CommunityContextDegradationReason,
    QuestionCluster,
    StoredMessage,
)


class _FakeContext:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_context_window_contains_only_the_question_group() -> None:
    cluster, messages = _case()
    lookup = ChatContextLookup(cluster, messages)

    result = lookup.context_payload(cluster, limit=50)
    message_ids = [item["message_id"] for item in result["messages"]]

    assert message_ids == ["Q1", "M1", "M2"]
    assert "before" not in message_ids
    assert "other-group" not in message_ids
    discovery = lookup.discovery_result(
        _AnswerAssessment(("q1",), ("a1",), "应先在信息门户挂失，再前往服务点处理。"),
        cluster,
    )
    assert discovery.question_message_ids == ("q1",)
    assert discovery.answers[0].redacted_text == "应先在信息门户挂失，再前往服务点处理。"
    assert discovery.answers[0].direct_reply is True


def test_context_window_excludes_configured_historical_bot_before_ai() -> None:
    cluster, messages = _case()
    bot_message = _message("bot-answer", 103, "bot-qq", "机器人自动回答", reply="q1")
    lookup = ChatContextLookup(
        cluster,
        [*messages, bot_message],
        ignored_sender_ids=frozenset({"bot-qq"}),
    )

    result = lookup.context_payload(cluster, limit=50)

    assert "bot-answer" not in [item["message_id"] for item in result["messages"]]


def test_agent_finds_answer_in_first_50_messages_with_one_call() -> None:
    async def run() -> None:
        cluster, messages = _case()
        context = _FakeContext(
            [
                SimpleNamespace(
                    completion_text=(
                        '{"question_message_ids":["q1"],'
                        '"answer_message_ids":["a1"],'
                        '"answer_summary":"应先挂失，再前往服务点处理。",'
                        '"reason":"a1明确回答，water是闲聊"}'
                    ),
                ),
            ]
        )
        agent = AstrBotContextAnswerAgent(
            context,
            provider_id="provider",
        )

        discovery = await agent.collect(cluster, messages)

        assert discovery.question_message_ids == ("q1",)
        assert discovery.answers[0].redacted_text == "应先挂失，再前往服务点处理。"
        assert len(context.calls) == 1
        assert "tools" not in context.calls[0]
        assert _prompt_payload(context.calls[0])["message_limit"] == 50

    asyncio.run(run())


def test_agent_can_split_an_upstream_overmerged_question_without_an_extra_call() -> None:
    async def run() -> None:
        cluster = QuestionCluster(
            question_code="20260712-Q001",
            report_date="2026-07-12",
            canonical_question="陶二条件如何，大二能否住上翻新宿舍？",
            category="住宿食堂",
            candidate_source_keys=("message:qq:bot:q1", "message:qq:bot:q2"),
            representative_questions=("陶二条件如何，大二能否住上翻新宿舍？",),
            group_aliases=("南京大学迎新群",),
            first_sent_at_utc=100,
            last_sent_at_utc=101,
        )
        messages = [
            _message("q1", 100, "u1", "陶二条件怎么样？"),
            _message("q2", 101, "u1", "大二能住上翻新的宿舍吗？"),
            _message("a1", 102, "u2", "陶二有电梯和独立卫浴。", reply="q1"),
            _message("a2", 103, "u3", "大二宿舍安排还没有通知。", reply="q2"),
        ]
        response = {
            "question_groups": [
                {
                    "question_message_ids": ["q1"],
                    "canonical_question": "陶二宿舍有哪些设施",
                    "category": "住宿食堂",
                    "answer_message_ids": ["a1"],
                    "answer_summary": "陶二宿舍有电梯和独立卫浴。",
                    "reason": "第一项询问现有设施",
                },
                {
                    "question_message_ids": ["q2"],
                    "canonical_question": "大二学生能否住上翻新后的宿舍",
                    "category": "住宿食堂",
                    "answer_message_ids": ["a2"],
                    "answer_summary": "大二宿舍安排尚未通知。",
                    "reason": "第二项询问未来分配",
                },
            ]
        }
        context = _FakeContext(
            [SimpleNamespace(completion_text=json.dumps(response, ensure_ascii=False))]
        )
        agent = AstrBotContextAnswerAgent(context, provider_id="provider")

        discovery = await agent.collect(cluster, messages)

        assert len(discovery.questions) == 2
        assert discovery.questions[0].question_message_ids == ("q1",)
        assert discovery.questions[1].question_message_ids == ("q2",)
        assert discovery.questions[1].canonical_question == "大二学生能否住上翻新后的宿舍"
        assert len(context.calls) == 1

    asyncio.run(run())


def test_agent_expands_once_to_100_messages_then_stops() -> None:
    async def run() -> None:
        cluster, _ = _case()
        messages = [_message("q1", 0, "u0", "校园卡丢了怎么办？")]
        messages.extend(
            _message(
                f"m{index}",
                index,
                f"u{index}",
                "先去信息门户挂失" if index == 70 else f"普通消息 {index}",
                reply="q1" if index == 70 else "",
            )
            for index in range(1, 120)
        )
        context = _FakeContext(
            [
                SimpleNamespace(
                    completion_text=(
                        '{"question_message_ids":["q1"],"answer_message_ids":[],'
                        '"answer_summary":"","reason":"前50条未找到"}'
                    )
                ),
                SimpleNamespace(
                    completion_text=(
                        '{"question_message_ids":["q1"],'
                        '"answer_message_ids":["m70"],'
                        '"answer_summary":"应先在信息门户挂失。",'
                        '"reason":"扩展后找到明确回答"}'
                    )
                ),
            ]
        )
        agent = AstrBotContextAnswerAgent(context, provider_id="provider")

        discovery = await agent.collect(cluster, messages)

        assert discovery.question_message_ids == ("q1",)
        assert discovery.answers[0].redacted_text == "应先在信息门户挂失。"
        assert len(context.calls) == 2
        assert [_prompt_payload(item)["message_limit"] for item in context.calls] == [50, 100]
        assert [len(_prompt_payload(item)["messages"]) for item in context.calls] == [50, 100]

    asyncio.run(run())


def test_expanded_pass_exception_preserves_initial_validation_audit() -> None:
    async def run() -> None:
        cluster, _ = _case()
        messages = [_message("q1", 0, "u0", "校园卡丢了怎么办？")]
        messages.extend(
            _message(f"m{index}", index, f"u{index}", f"普通消息 {index}")
            for index in range(1, 120)
        )
        invalid = SimpleNamespace(
            completion_text=(
                '{"question_message_ids":["bad-id"],'
                '"answer_message_ids":[],"answer_summary":""}'
            )
        )
        repaired = SimpleNamespace(
            completion_text=(
                '{"question_message_ids":["Q1"],'
                '"answer_message_ids":[],"answer_summary":""}'
            )
        )
        context = _FakeContext([invalid, repaired, TimeoutError("provider timeout")])

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert discovery.question_message_ids == ("q1",)
        assert discovery.community_context_degraded is True
        assert discovery.community_context_degradation_reason is (
            CommunityContextDegradationReason.AGENT_EXCEPTION
        )
        audit = discovery.community_context_audit
        assert audit.initial_errors
        assert "TimeoutError" in " ".join(audit.retry_errors)
        assert audit.degraded_question_ids == ("Q1",)
        assert "EXPANDED_PASS_EXCEPTION_SAFE_FALLBACK" in audit.fallback_actions

    asyncio.run(run())


def test_agent_retries_invalid_anchor_with_allowed_short_ids() -> None:
    async def run() -> None:
        cluster, messages = _case()
        context = _FakeContext(
            [
                SimpleNamespace(
                    completion_text=(
                        '{"question_message_ids":["not-an-anchor"],'
                        '"answer_message_ids":[],"answer_summary":""}'
                    )
                ),
                SimpleNamespace(
                    completion_text=(
                        '{"question_message_ids":["Q1"],'
                        '"answer_message_ids":["M2"],'
                        '"answer_summary":"应先挂失，再前往服务点办理。"}'
                    )
                ),
            ]
        )
        agent = AstrBotContextAnswerAgent(context, provider_id="provider")

        discovery = await agent.collect(cluster, messages)

        assert discovery.question_message_ids == ("q1",)
        assert discovery.answers
        assert discovery.community_context_degraded is False
        assert len(context.calls) == 2
        correction = _prompt_payload(context.calls[1])["validation_correction"]
        assert correction["allowed_question_ids"] == ["Q1"]
        assert correction["allowed_answer_ids"] == ["M1", "M2"]
        assert correction["errors"]

    asyncio.run(run())


def test_agent_keeps_valid_group_and_degrades_only_unrepaired_group() -> None:
    async def run() -> None:
        cluster = QuestionCluster(
            question_code="20260712-Q001",
            report_date="2026-07-12",
            canonical_question="宿舍设施和分配问题",
            category="住宿",
            candidate_source_keys=("message:qq:bot:q1", "message:qq:bot:q2"),
            representative_questions=("宿舍设施和分配问题",),
            group_aliases=("测试群",),
            first_sent_at_utc=100,
            last_sent_at_utc=101,
        )
        messages = [
            _message("q1", 100, "u1", "宿舍有哪些设施？"),
            _message("q2", 101, "u1", "宿舍怎么分配？"),
            _message("a1", 102, "u2", "宿舍有空调。", reply="q1"),
        ]
        first = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "canonical_question": "宿舍有哪些设施",
                    "category": "住宿",
                    "answer_message_ids": ["M1"],
                    "answer_summary": "群聊称宿舍有空调。",
                },
                {
                    "question_message_ids": ["bad-id"],
                    "canonical_question": "宿舍怎么分配",
                    "category": "住宿",
                    "answer_message_ids": [],
                    "answer_summary": "",
                },
            ]
        }
        still_invalid = {
            "question_groups": [
                {
                    "question_message_ids": ["bad-id"],
                    "canonical_question": "宿舍怎么分配",
                    "category": "住宿",
                    "answer_message_ids": [],
                    "answer_summary": "",
                }
            ]
        }
        context = _FakeContext(
            [
                SimpleNamespace(completion_text=json.dumps(first, ensure_ascii=False)),
                SimpleNamespace(completion_text=json.dumps(still_invalid, ensure_ascii=False)),
            ]
        )
        agent = AstrBotContextAnswerAgent(context, provider_id="provider")

        discovery = await agent.collect(cluster, messages)

        assert len(discovery.questions) == 2
        assert discovery.questions[0].question_message_ids == ("q1",)
        assert discovery.questions[0].answers
        assert discovery.questions[0].community_context_degraded is False
        assert discovery.questions[1].question_message_ids == ("q2",)
        assert discovery.questions[1].answers == ()
        assert discovery.questions[1].community_context_degraded is True
        assert discovery.questions[1].community_context_degradation_reason is (
            CommunityContextDegradationReason.VALIDATION_UNRESOLVED
        )
        audit = discovery.questions[1].community_context_audit
        assert audit.initial_errors
        assert audit.retry_errors
        assert audit.retained_question_ids == ("Q1",)
        assert audit.degraded_question_ids == ("Q2",)
        assert "SAFE_QUESTION_FROM_UNCOVERED_ANCHOR" in audit.fallback_actions
        correction = _prompt_payload(context.calls[1])["validation_correction"]
        assert correction["reserved_question_ids"] == ["Q1"]
        assert correction["reserved_answer_ids"] == ["M1"]

    asyncio.run(run())


def test_agent_retries_only_the_group_that_reuses_an_answer() -> None:
    async def run() -> None:
        cluster = QuestionCluster(
            question_code="20260712-Q001",
            report_date="2026-07-12",
            canonical_question="宿舍的设施与分配",
            category="住宿",
            candidate_source_keys=("message:qq:bot:q1", "message:qq:bot:q2"),
            representative_questions=("宿舍的设施与分配",),
            group_aliases=("测试群",),
            first_sent_at_utc=100,
            last_sent_at_utc=101,
        )
        messages = [
            _message("q1", 100, "u1", "宿舍有什么设施？"),
            _message("q2", 101, "u1", "宿舍怎么分配？"),
            _message("a1", 102, "u2", "宿舍有空调。", reply="q1"),
        ]
        first = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "群聊称宿舍有空调。",
                },
                {
                    "question_message_ids": ["Q2"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "群聊称宿舍有空调。",
                },
            ]
        }
        repaired = {
            "question_message_ids": ["Q2"],
            "answer_message_ids": [],
            "answer_summary": "",
        }
        context = _FakeContext(
            [
                SimpleNamespace(completion_text=json.dumps(first, ensure_ascii=False)),
                SimpleNamespace(completion_text=json.dumps(repaired, ensure_ascii=False)),
            ]
        )

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert [item.question_message_ids for item in discovery.questions] == [
            ("q1",),
            ("q2",),
        ]
        assert discovery.questions[0].answers
        assert discovery.questions[1].answers == ()
        assert all(not item.community_context_degraded for item in discovery.questions)
        correction = _prompt_payload(context.calls[1])["validation_correction"]
        assert correction["reserved_answer_ids"] == ["M1"]
        assert any("already belong to another group" in item for item in correction["errors"])

    asyncio.run(run())


def test_successful_retry_degrades_an_anchor_the_repair_did_not_cover() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2", "q3")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
            _message("q3", 102, "u1", "问题三？"),
            _message("a1", 103, "u2", "问题一的回答。", reply="q1"),
        ]
        first = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "问题一已有回答。",
                },
                {
                    "question_message_ids": ["bad-id"],
                    "answer_message_ids": [],
                    "answer_summary": "",
                },
            ]
        }
        repaired = {
            "question_message_ids": ["Q3"],
            "answer_message_ids": [],
            "answer_summary": "",
        }
        context = _FakeContext(
            [
                SimpleNamespace(completion_text=json.dumps(first, ensure_ascii=False)),
                SimpleNamespace(completion_text=json.dumps(repaired, ensure_ascii=False)),
            ]
        )

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert [item.question_message_ids for item in discovery.questions] == [
            ("q1",),
            ("q2",),
            ("q3",),
        ]
        assert discovery.questions[1].community_context_degraded is True
        assert discovery.questions[2].community_context_degraded is False

    asyncio.run(run())


def test_cross_group_question_answer_overlap_retries_both_groups() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
            _message("a1", 102, "u2", "问题二的回答。", reply="q2"),
        ]
        conflicted = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["Q2"],
                    "answer_summary": "把第二个锚点误作回答。",
                },
                {
                    "question_message_ids": ["Q2"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "问题二已有回答。",
                },
            ]
        }
        repaired = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": [],
                    "answer_summary": "",
                },
                {
                    "question_message_ids": ["Q2"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "问题二已有回答。",
                },
            ]
        }
        context = _FakeContext(
            [
                SimpleNamespace(completion_text=json.dumps(conflicted, ensure_ascii=False)),
                SimpleNamespace(completion_text=json.dumps(repaired, ensure_ascii=False)),
            ]
        )

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert [item.question_message_ids for item in discovery.questions] == [
            ("q1",),
            ("q2",),
        ]
        assert all(not item.community_context_degraded for item in discovery.questions)
        correction = _prompt_payload(context.calls[1])["validation_correction"]
        assert correction["reserved_question_ids"] == ["Q2"]
        assert any("候选问题锚点" in item for item in correction["errors"])

    asyncio.run(run())


def test_overlap_revealed_by_repair_degrades_both_conflicting_groups() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
        ]
        first = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["Q2"],
                    "answer_summary": "把第二个锚点误作回答。",
                },
                {
                    "question_message_ids": ["Q2"],
                    "answer_message_ids": [],
                },
            ]
        }
        repaired = {
            "question_message_ids": ["Q2"],
            "answer_message_ids": [],
            "answer_summary": "",
        }
        context = _FakeContext(
            [
                SimpleNamespace(completion_text=json.dumps(first, ensure_ascii=False)),
                SimpleNamespace(completion_text=json.dumps(repaired, ensure_ascii=False)),
            ]
        )

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert [item.question_message_ids for item in discovery.questions] == [
            ("q1",),
            ("q2",),
        ]
        assert discovery.questions[0].community_context_degraded is True
        assert discovery.questions[0].community_context_degradation_reason is (
            CommunityContextDegradationReason.VALIDATION_UNRESOLVED
        )
        assert discovery.questions[1].community_context_degraded is False

    asyncio.run(run())


def test_validation_retry_budget_is_shared_by_initial_and_expanded_context() -> None:
    async def run() -> None:
        cluster, _ = _case()
        messages = [_message("q1", 0, "u0", "校园卡丢了怎么办？")]
        messages.extend(
            _message(f"m{index}", index, f"u{index}", f"普通消息 {index}")
            for index in range(1, 120)
        )
        invalid = SimpleNamespace(
            completion_text=(
                '{"question_message_ids":["bad-id"],'
                '"answer_message_ids":[],"answer_summary":""}'
            )
        )
        repaired_without_answer = SimpleNamespace(
            completion_text=(
                '{"question_message_ids":["Q1"],'
                '"answer_message_ids":[],"answer_summary":""}'
            )
        )
        context = _FakeContext([invalid, repaired_without_answer, invalid])

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert len(context.calls) == 3
        assert sum(
            "validation_correction" in _prompt_payload(call) for call in context.calls
        ) == 1
        assert discovery.community_context_degraded is True
        assert discovery.community_context_audit.initial_errors
        assert discovery.community_context_audit.degraded_question_ids == ("Q1",)

    asyncio.run(run())


def test_retry_failure_preserves_valid_groups_and_degrades_only_uncovered_anchors() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
            _message("a1", 102, "u2", "问题一的回答。", reply="q1"),
        ]
        first = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "问题一已有回答。",
                },
                {
                    "question_message_ids": ["bad-id"],
                    "answer_message_ids": [],
                    "answer_summary": "",
                },
            ]
        }
        context = _FakeContext(
            [
                SimpleNamespace(completion_text=json.dumps(first, ensure_ascii=False)),
                TimeoutError("provider timeout"),
            ]
        )

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert discovery.questions[0].question_message_ids == ("q1",)
        assert discovery.questions[0].answers
        assert discovery.questions[1].question_message_ids == ("q2",)
        assert discovery.questions[1].community_context_degraded is True
        assert discovery.questions[1].community_context_degradation_reason is (
            CommunityContextDegradationReason.RETRY_FAILED
        )
        assert "TimeoutError" in " ".join(
            discovery.questions[1].community_context_audit.retry_errors
        )

    asyncio.run(run())


def test_deterministic_fallback_sanitizes_ids_overlap_duplicates_and_summary() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
            _message("a1", 102, "u2", "问题一的可见回答。", reply="q1"),
        ]
        invalid = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["Q1", "M1", "not-visible"],
                    "answer_summary": "不可信摘要。",
                },
                {
                    "question_message_ids": ["Q2"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "",
                },
            ]
        }
        response = SimpleNamespace(completion_text=json.dumps(invalid, ensure_ascii=False))
        context = _FakeContext([response, response])

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert [item.question_message_ids for item in discovery.questions] == [
            ("q1",),
            ("q2",),
        ]
        assert discovery.questions[0].answers[0].redacted_text == "问题一的可见回答。"
        assert discovery.questions[1].answers == ()
        assert all(item.community_context_degraded for item in discovery.questions)
        audit = discovery.questions[0].community_context_audit
        assert "REBUILT_SUMMARY_FROM_VISIBLE_MESSAGES" in audit.fallback_actions
        assert "DROPPED_INVALID_ANSWERS_AND_CLEARED_SUMMARY" in audit.fallback_actions
        assert audit.degraded_question_ids == ("Q1", "Q2")

    asyncio.run(run())


def test_deterministic_fallback_rebuilds_untrusted_summary_from_visible_answers() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
            _message("a1", 102, "u2", "唯一可见的回答。", reply="q1"),
        ]
        invalid = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1", "not-visible"],
                    "answer_message_ids": ["M1"],
                    "answer_summary": "模型凭空编造的摘要。",
                }
            ]
        }
        response = SimpleNamespace(completion_text=json.dumps(invalid, ensure_ascii=False))
        context = _FakeContext([response, response])

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert discovery.questions[0].answers[0].redacted_text == "唯一可见的回答。"
        assert discovery.questions[0].community_context_degraded is True
        assert "REBUILT_SUMMARY_FROM_VISIBLE_MESSAGES" in (
            discovery.questions[0].community_context_audit.fallback_actions
        )

    asyncio.run(run())


def test_rebuilding_a_missing_summary_is_counted_as_degradation() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("a1", 101, "u2", "唯一可见的回答。", reply="q1"),
        ]
        invalid = {
            "question_message_ids": ["Q1"],
            "answer_message_ids": ["M1"],
            "answer_summary": "",
        }
        response = SimpleNamespace(completion_text=json.dumps(invalid, ensure_ascii=False))
        context = _FakeContext([response, response])

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert discovery.answers[0].redacted_text == "唯一可见的回答。"
        assert discovery.community_context_degraded is True
        assert discovery.community_context_degradation_reason is (
            CommunityContextDegradationReason.VALIDATION_UNRESOLVED
        )
        assert discovery.community_context_audit.degraded_question_ids == ("Q1",)
        assert "REBUILT_SUMMARY_FROM_VISIBLE_MESSAGES" in (
            discovery.community_context_audit.fallback_actions
        )

    asyncio.run(run())


def test_question_anchor_cannot_be_reused_as_an_answer_in_safe_fallback() -> None:
    async def run() -> None:
        cluster = _cluster_for("q1", "q2")
        messages = [
            _message("q1", 100, "u1", "问题一？"),
            _message("q2", 101, "u1", "问题二？"),
        ]
        invalid = {
            "question_groups": [
                {
                    "question_message_ids": ["Q1"],
                    "answer_message_ids": ["Q2"],
                    "answer_summary": "错误地把另一问题当回答。",
                }
            ]
        }
        response = SimpleNamespace(completion_text=json.dumps(invalid, ensure_ascii=False))
        context = _FakeContext([response, response])

        discovery = await AstrBotContextAnswerAgent(
            context, provider_id="provider"
        ).collect(cluster, messages)

        assert [item.question_message_ids for item in discovery.questions] == [
            ("q1",),
            ("q2",),
        ]
        assert all(not item.answers for item in discovery.questions)
        assert "FILTERED_INVALID_OR_DUPLICATE_ANSWER_IDS" in (
            discovery.questions[0].community_context_audit.fallback_actions
        )

    asyncio.run(run())


def _prompt_payload(call: dict) -> dict:
    return json.loads(call["prompt"].split("\n", 1)[1])


def _case() -> tuple[QuestionCluster, list[StoredMessage]]:
    cluster = QuestionCluster(
        question_code="20260712-Q001",
        report_date="2026-07-12",
        canonical_question="校园卡丢失后如何补办？",
        category="校园服务/校园卡",
        candidate_source_keys=("message:qq:bot:q1",),
        representative_questions=("校园卡丢了怎么补办？",),
        group_aliases=("南京大学迎新群",),
        first_sent_at_utc=100,
        last_sent_at_utc=100,
    )
    return cluster, [
        _message("before", 90, "u0", "刚吃完饭"),
        _message("q1", 100, "u1", "校园卡丢了怎么补办？"),
        _message("water", 101, "u2", "哈哈哈哈"),
        _message("a1", 102, "u3", "先在信息门户挂失，再去服务点。", reply="q1"),
        _message("other-group", 103, "u4", "另一个群的消息", group_id="2"),
    ]


def _cluster_for(*external_ids: str) -> QuestionCluster:
    return QuestionCluster(
        question_code="20260712-Q001",
        report_date="2026-07-12",
        canonical_question="多个测试问题",
        category="测试",
        candidate_source_keys=tuple(
            f"message:qq:bot:{external_id}" for external_id in external_ids
        ),
        representative_questions=("多个测试问题",),
        group_aliases=("测试群",),
        first_sent_at_utc=100,
        last_sent_at_utc=100 + len(external_ids) - 1,
    )


def _message(
    external_id: str,
    sent_at: int,
    sender: str,
    text: str,
    *,
    reply: str = "",
    group_id: str = "1",
) -> StoredMessage:
    return StoredMessage(
        platform_id="qq",
        bot_self_id="bot",
        external_message_id=external_id,
        message_fingerprint=external_id,
        session_id=f"group:{group_id}",
        group_id=group_id,
        group_alias="南京大学迎新群",
        sender_id=sender,
        sender_name="",
        sent_at_utc=sent_at,
        text=text,
        outline="",
        reply_to_message_id=reply,
        analyzable=True,
    )
