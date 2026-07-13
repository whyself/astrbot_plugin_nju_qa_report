from __future__ import annotations

import asyncio

import pytest

from nju_report.models import ScopeDecision
from nju_report.privacy import prepare_scope_input
from nju_report.scope_ai import (
    AstrBotScopeAiClient,
    ScopeAiResponseError,
    parse_scope_assessment,
)


def test_parse_scope_assessment_accepts_fenced_json() -> None:
    result = parse_scope_assessment(
        """```json
        {
          "decision": "INCLUDE",
          "reason": "属于南京大学转专业问题",
          "confidence": 0.91,
          "canonical_question": "软件学院转专业需要参加哪些考核？",
          "category": "学业与培养/转专业",
          "clarity": "CLEAR",
          "knowledge_value": "HIGH",
          "time_sensitive": true
        }
        ```"""
    )
    assert result.decision is ScopeDecision.INCLUDE
    assert result.canonical_question == "软件学院转专业需要参加哪些考核？"
    assert result.time_sensitive is True


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"decision":"INCLUDE"}',
        """{
          "decision":"DROP_LOW_CONFIDENCE",
          "reason":"x",
          "confidence":0.5,
          "canonical_question":"",
          "category":"",
          "clarity":"UNCERTAIN",
          "knowledge_value":"LOW",
          "time_sensitive":false
        }""",
    ],
)
def test_invalid_model_contract_is_rejected(payload: str) -> None:
    with pytest.raises(ScopeAiResponseError):
        parse_scope_assessment(payload)


def test_multiple_or_prefixed_json_objects_are_rejected() -> None:
    valid = """{
      "decision":"DROP",
      "reason":"无关",
      "confidence":0.9,
      "canonical_question":"",
      "category":"",
      "clarity":"CLEAR",
      "knowledge_value":"LOW",
      "time_sensitive":false
    }"""
    with pytest.raises(ScopeAiResponseError):
        parse_scope_assessment('{"decision":"INCLUDE"}\n' + valid)
    with pytest.raises(ScopeAiResponseError):
        parse_scope_assessment("模型回答：" + valid)


def test_scope_input_is_redacted_and_bounded() -> None:
    prepared = prepare_scope_input(
        "我叫张三，QQ:12345678，手机号13800138000，邮箱a@example.com，宿舍1A23",
        "身份证320101200001011234，详情https://private.example/x " + "甲" * 9000,
    )
    combined = prepared.message + prepared.context
    for sensitive in (
        "张三",
        "12345678",
        "13800138000",
        "a@example.com",
        "1A23",
        "320101200001011234",
        "private.example",
    ):
        assert sensitive not in combined
    assert "[内容已截断]" in prepared.context
    assert len(prepared.context) <= 8000


def test_ai_client_never_sends_raw_identifiers_to_provider() -> None:
    completion = """{
      "decision":"INCLUDE",
      "reason":"属于校园卡办理问题",
      "confidence":0.9,
      "canonical_question":"校园卡丢失后如何补办？",
      "category":"校园生活/校园卡",
      "clarity":"CLEAR",
      "knowledge_value":"HIGH",
      "time_sensitive":false
    }"""

    class Response:
        completion_text = completion

    class Context:
        prompt = ""

        async def llm_generate(self, **kwargs):
            self.prompt = kwargs["prompt"]
            return Response()

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=0)
    result = asyncio.run(client.classify("手机号13800138000，一卡通丢了怎么办？", "QQ:12345678"))
    assert result.decision is ScopeDecision.INCLUDE
    assert "13800138000" not in context.prompt
    assert "12345678" not in context.prompt
    assert "[手机号]" in context.prompt
    assert "[账号]" in context.prompt
