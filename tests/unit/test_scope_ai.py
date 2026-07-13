from __future__ import annotations

import asyncio
import json

import pytest

from nju_report.models import ScopeDecision
from nju_report.privacy import prepare_scope_input
from nju_report.scope_ai import (
    AstrBotScopeAiClient,
    ScopeAiError,
    ScopeAiResponseError,
    parse_final_question_gate,
    parse_scope_assessment,
    parse_scope_batch,
)
from nju_report.scope_classifier import QuestionGateCandidate, ScopeBatchMessage


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


def test_parse_scope_batch_accepts_selected_questions_only() -> None:
    question = {
        "question_message_ids": ["m1", "m2"],
        "reason": "两条消息共同表达一个校园问题",
        "confidence": 0.9,
        "canonical_question": "南京大学校园卡如何补办？",
        "category": "校园生活/校园卡",
        "clarity": "CLEAR",
        "knowledge_value": "HIGH",
        "time_sensitive": False,
    }
    payload = json.dumps(
        {
            "questions": [question],
            "uncertain_questions": [],
        },
        ensure_ascii=False,
    )
    result = parse_scope_batch(payload, ["m1", "m2", "m3"])
    assert list(result) == ["m1", "m2", "m3"]
    assert result["m1"].canonical_question == result["m2"].canonical_question
    assert result["m3"].decision is ScopeDecision.DROP
    assert "未将该消息提取" in result["m3"].reason

    unknown_result = parse_scope_batch(payload, ["m1"])
    assert unknown_result["m1"].decision is ScopeDecision.INCLUDE

    duplicate_payload = json.dumps(
        {
            "questions": [question],
            "uncertain_questions": [question],
        },
        ensure_ascii=False,
    )
    duplicate_result = parse_scope_batch(duplicate_payload, ["m1", "m2"])
    assert duplicate_result["m1"].decision is ScopeDecision.INCLUDE
    assert duplicate_result["m2"].decision is ScopeDecision.INCLUDE


def test_parse_scope_batch_ignores_bad_items_and_accepts_plain_question_array() -> None:
    valid = {
        "question_message_ids": ["m1", "not-a-target"],
        "reason": "明确的校园问题",
        "confidence": 0.9,
        "canonical_question": "南京大学校园卡如何补办？",
        "category": "校园卡",
        "clarity": "CLEAR",
        "knowledge_value": "HIGH",
        "time_sensitive": False,
    }
    result = parse_scope_batch(
        json.dumps([{"question_message_ids": ["m2"]}, valid], ensure_ascii=False),
        ["m1", "m2", "m3"],
    )

    assert result["m1"].decision is ScopeDecision.INCLUDE
    assert result["m2"].decision is ScopeDecision.DROP
    assert result["m3"].decision is ScopeDecision.DROP


def test_parse_scope_batch_still_rejects_non_json_response() -> None:
    with pytest.raises(ScopeAiResponseError):
        parse_scope_batch("不是 JSON", ["m1"])


def test_final_question_gate_can_keep_rewrite_merge_and_drop() -> None:
    payload = json.dumps(
        {
            "questions": [
                {
                    "candidate_ids": ["c1", "c2"],
                    "reason": "两个候选询问同一项宿舍分配规则，合并并补足学校名称",
                    "confidence": 0.96,
                    "canonical_question": "南京大学2026级本科生大二宿舍如何分配？",
                    "category": "住宿宿舍",
                    "knowledge_value": "HIGH",
                    "time_sensitive": True,
                }
            ]
        },
        ensure_ascii=False,
    )

    result = parse_final_question_gate(payload, ["c1", "c2", "c3"])

    assert result["c1"].decision is ScopeDecision.INCLUDE
    assert result["c2"].canonical_question == result["c1"].canonical_question
    assert result["c3"].decision is ScopeDecision.DROP
    assert "未保留" in result["c3"].reason


@pytest.mark.parametrize(
    "payload",
    [
        '{"questions":"not-an-array"}',
        '{"questions":[{"candidate_ids":["unknown"]}]}',
        json.dumps(
            {
                "questions": [
                    {
                        "candidate_ids": ["c1"],
                        "reason": "低价值却被保留",
                        "confidence": 0.9,
                        "canonical_question": "哪里最好吃？",
                        "category": "娱乐",
                        "knowledge_value": "LOW",
                        "time_sensitive": False,
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ],
)
def test_final_question_gate_rejects_invalid_contract(payload: str) -> None:
    with pytest.raises(ScopeAiResponseError):
        parse_final_question_gate(payload, ["c1"])


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
        system_prompt = ""

        async def llm_generate(self, **kwargs):
            self.prompt = kwargs["prompt"]
            self.system_prompt = kwargs["system_prompt"]
            return Response()

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=0)
    result = asyncio.run(client.classify("手机号13800138000，一卡通丢了怎么办？", "QQ:12345678"))
    assert result.decision is ScopeDecision.INCLUDE
    assert "13800138000" not in context.prompt
    assert "12345678" not in context.prompt
    assert "[手机号]" in context.prompt
    assert "[账号]" in context.prompt


def test_batch_client_sends_every_target_and_redacts_content() -> None:
    completion = json.dumps(
        {
            "questions": [],
            "uncertain_questions": [],
        },
        ensure_ascii=False,
    )

    class Response:
        completion_text = completion

    class Context:
        prompt = ""
        system_prompt = ""

        async def llm_generate(self, **kwargs):
            self.prompt = kwargs["prompt"]
            self.system_prompt = kwargs["system_prompt"]
            return Response()

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=0)
    result = asyncio.run(
        client.classify_batch(
            [
                ScopeBatchMessage(
                    "m1",
                    "手机号 13800138000，这是回答",
                    conversation_date="2026-07-12",
                )
            ],
            ["m1"],
        )
    )
    assert list(result) == ["m1"]
    assert "13800138000" not in context.prompt
    assert "[手机号]" in context.prompt
    assert '"conversation_date": "2026-07-12"' in context.prompt
    assert "筛选对象只能是“问题”，不是答案" in context.system_prompt
    assert "本阶段不得搜索、评价、摘录或生成问题的答案" in context.system_prompt
    assert "不是群聊话题档案、校园趣闻合集" in context.system_prompt
    assert "仍无法可靠还原，应直接排除" in context.system_prompt
    assert "录取通知书是否好看" in context.system_prompt
    assert "哪个窗口好吃" in context.system_prompt
    assert "社团或同好群的推荐与群号" in context.system_prompt
    assert "独立可读" in context.system_prompt


def test_final_gate_sends_only_concise_candidate_questions() -> None:
    completion = json.dumps({"questions": []}, ensure_ascii=False)

    class Response:
        completion_text = completion

    class Context:
        prompt = ""
        system_prompt = ""

        async def llm_generate(self, **kwargs):
            self.prompt = kwargs["prompt"]
            self.system_prompt = kwargs["system_prompt"]
            return Response()

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=0)
    result = asyncio.run(
        client.final_review_batch(
            [
                QuestionGateCandidate(
                    candidate_id="c1",
                    canonical_question="南京大学鼓楼校区哪些窗口比较好吃？",
                    category="住宿食堂",
                    source_count=2,
                )
            ]
        )
    )

    assert result["c1"].decision is ScopeDecision.DROP
    assert "ordered_messages" not in context.prompt
    assert "哪些窗口比较好吃" in context.prompt
    assert "最终问题编辑" in context.system_prompt
    assert "保留、改写、删除和合并" not in context.prompt
    assert "绝不搜索、生成或评价答案" in context.system_prompt
    assert "优先保留、规范化和去重" in context.system_prompt
    assert "陶二宿舍条件如何" in context.system_prompt
    assert "鼓楼宿舍是否配备马桶" in context.system_prompt
    assert "陶二、南二、仙林巴士、小百合" in context.system_prompt
    assert "小百合是什么”必须保留" in context.system_prompt
    assert "专业选修课是什么”必须保留" in context.system_prompt
    assert "标题不需要重复添加“南京大学”" in context.system_prompt
    assert "已有资料可能明确回答的问题仍应保留" in context.system_prompt
    assert "宁可少收" not in context.system_prompt


def test_final_gate_returns_format_error_to_model_for_repair() -> None:
    class Response:
        def __init__(self, text: str) -> None:
            self.completion_text = text

    class Context:
        def __init__(self) -> None:
            self.responses = [Response("不是 JSON"), Response('{"questions":[]}')]
            self.prompts: list[str] = []

        async def llm_generate(self, **kwargs):
            self.prompts.append(kwargs["prompt"])
            return self.responses.pop(0)

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=3)
    result = asyncio.run(
        client.final_review_batch(
            [QuestionGateCandidate("c1", "南京大学校园卡如何补办？")]
        )
    )

    assert result["c1"].decision is ScopeDecision.DROP
    assert len(context.prompts) == 2
    assert "具体格式错误：JSON 解析失败" in context.prompts[1]
    assert "上一次原始响应：\n不是 JSON" in context.prompts[1]
    assert "修复重试：1/3" in context.prompts[1]


def test_batch_client_returns_format_error_to_model_for_repair() -> None:
    valid = json.dumps({"questions": []}, ensure_ascii=False)

    class Response:
        def __init__(self, text: str) -> None:
            self.completion_text = text

    class Context:
        def __init__(self) -> None:
            self.responses = [Response("不是 JSON"), Response(valid)]
            self.prompts: list[str] = []

        async def llm_generate(self, **kwargs):
            self.prompts.append(kwargs["prompt"])
            return self.responses.pop(0)

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=3)

    result = asyncio.run(
        client.classify_batch([ScopeBatchMessage("m1", "普通消息")], ["m1"])
    )

    assert result["m1"].decision is ScopeDecision.DROP
    assert len(context.prompts) == 2
    assert "具体格式错误：JSON 解析失败" in context.prompts[1]
    assert "上一次原始响应：\n不是 JSON" in context.prompts[1]
    assert "修复重试：1/3" in context.prompts[1]


def test_batch_format_repair_is_capped_at_three_retries(monkeypatch) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("nju_report.scope_ai.asyncio.sleep", no_sleep)

    class Response:
        completion_text = "仍然不是 JSON"

    class Context:
        calls = 0

        async def llm_generate(self, **kwargs):
            del kwargs
            self.calls += 1
            return Response()

    context = Context()
    client = AstrBotScopeAiClient(context, provider_id="provider", max_retries=10)

    with pytest.raises(ScopeAiError):
        asyncio.run(
            client.classify_batch([ScopeBatchMessage("m1", "普通消息")], ["m1"])
        )

    assert context.calls == 4
