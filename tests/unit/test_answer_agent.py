from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from nju_report.answer_agent import (
    AstrBotContextAnswerAgent,
    ChatContextLookup,
    _AnswerAssessment,
)
from nju_report.models import QuestionCluster, StoredMessage


class _FakeContext:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_context_window_contains_only_the_question_group() -> None:
    cluster, messages = _case()
    lookup = ChatContextLookup(cluster, messages)

    result = lookup.context_payload(cluster, limit=50)
    message_ids = [item["message_id"] for item in result["messages"]]

    assert message_ids == ["q1", "water", "a1"]
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
