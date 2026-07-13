from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nju_report import answer_agent as answer_agent_module
from nju_report.answer_agent import AstrBotContextAnswerAgent, ChatContextLookup
from nju_report.models import QuestionCluster, StoredMessage


class _FakeMessage:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeFunctionTool:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeToolSet:
    def __init__(self, *, tools) -> None:
        self.tools = tools


class _FakeContext:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_context_lookup_pages_around_only_the_question_group() -> None:
    cluster, messages = _case()
    lookup = ChatContextLookup(cluster, messages, excluded_message_ids={"q1"})

    result = lookup.read(cursor_message_id="q1", before=0, after=2)

    assert '"message_id": "q1"' in result
    assert '"message_id": "water"' in result
    assert '"message_id": "a1"' in result
    assert '"message_id": "other-group"' not in result
    answers = lookup.answers_from_ids(["a1"])
    assert [item.external_message_id for item in answers] == ["a1"]
    assert answers[0].direct_reply is True


def test_agent_calls_context_tool_and_keeps_only_selected_answer(monkeypatch) -> None:
    async def run() -> None:
        cluster, messages = _case()
        context = _FakeContext(
            [
                SimpleNamespace(
                    completion_text="",
                    tools_call_name=["nju_read_chat_context"],
                    tools_call_args=[
                        {"cursor_message_id": "q1", "before": 0, "after": 2}
                    ],
                    tools_call_ids=["call-1"],
                ),
                SimpleNamespace(
                    completion_text=(
                        '{"answer_message_ids":["a1"],"reason":"a1明确回答，water是闲聊"}'
                    ),
                    tools_call_name=[],
                    tools_call_args=[],
                    tools_call_ids=[],
                ),
            ]
        )
        monkeypatch.setattr(
            answer_agent_module,
            "_astrbot_agent_types",
            lambda: (_FakeMessage, _FakeFunctionTool, _FakeToolSet),
        )
        agent = AstrBotContextAnswerAgent(
            context,
            provider_id="provider",
            max_retries=0,
        )

        answers = await agent.collect(cluster, messages, excluded_message_ids={"q1"})

        assert [item.external_message_id for item in answers] == ["a1"]
        assert len(context.calls) == 2
        second_context = context.calls[1]["contexts"]
        assert any(getattr(item, "role", "") == "tool" for item in second_context)

    asyncio.run(run())


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
