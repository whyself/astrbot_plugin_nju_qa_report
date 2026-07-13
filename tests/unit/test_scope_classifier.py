from __future__ import annotations

import asyncio

from nju_report.models import (
    Clarity,
    KnowledgeValue,
    ScopeAssessment,
    ScopeDecision,
)
from nju_report.scope_classifier import AutoScopeReviewService


def _assessment(decision: ScopeDecision, *, reason: str = "理由") -> ScopeAssessment:
    return ScopeAssessment(
        decision=decision,
        reason=reason,
        confidence=0.8,
        canonical_question=(
            "软件学院转专业需要参加哪些考核？" if decision is ScopeDecision.INCLUDE else ""
        ),
        category="学业与培养/转专业",
        clarity=Clarity.CLEAR,
        knowledge_value=KnowledgeValue.HIGH,
    )


class FakeAi:
    def __init__(
        self,
        primary: ScopeAssessment | Exception,
        reviews: list[ScopeAssessment | Exception] | None = None,
    ) -> None:
        self.primary = primary
        self.reviews = list(reviews or [])
        self.classify_calls = 0
        self.review_calls = 0

    async def classify(self, message: str, context: str) -> ScopeAssessment:
        del message, context
        self.classify_calls += 1
        if isinstance(self.primary, Exception):
            raise self.primary
        return self.primary

    async def review(
        self,
        message: str,
        context: str,
        *,
        round_no: int,
    ) -> ScopeAssessment:
        del message, context, round_no
        self.review_calls += 1
        result = self.reviews.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_include_and_drop_skip_review_model() -> None:
    for decision in (ScopeDecision.INCLUDE, ScopeDecision.DROP):
        ai = FakeAi(_assessment(decision))
        result = asyncio.run(AutoScopeReviewService(ai, ai).resolve("问题"))
        assert result.assessment.decision is decision
        assert ai.review_calls == 0


def test_uncertain_candidate_is_automatically_reviewed() -> None:
    ai = FakeAi(
        _assessment(ScopeDecision.AUTO_REVIEW),
        [_assessment(ScopeDecision.INCLUDE)],
    )
    result = asyncio.run(AutoScopeReviewService(ai, ai).resolve("问题", "上下文"))
    assert result.assessment.decision is ScopeDecision.INCLUDE
    assert result.review_rounds == 1
    assert result.initial_assessment is not None
    assert result.initial_assessment.decision is ScopeDecision.AUTO_REVIEW
    assert result.review_attempts == (result.assessment,)


def test_repeated_uncertainty_becomes_low_confidence_drop_without_human_queue() -> None:
    ai = FakeAi(
        _assessment(ScopeDecision.AUTO_REVIEW),
        [
            _assessment(ScopeDecision.AUTO_REVIEW),
            _assessment(ScopeDecision.AUTO_REVIEW),
        ],
    )
    result = asyncio.run(AutoScopeReviewService(ai, ai, max_rounds=2).resolve("这个咋办"))
    assert result.assessment.decision is ScopeDecision.DROP_LOW_CONFIDENCE
    assert result.review_rounds == 2
    assert ai.review_calls == 2


def test_disabled_auto_review_drops_uncertain_candidate() -> None:
    ai = FakeAi(_assessment(ScopeDecision.AUTO_REVIEW))
    result = asyncio.run(AutoScopeReviewService(ai, ai, enabled=False).resolve("这个咋办"))
    assert result.assessment.decision is ScopeDecision.DROP_LOW_CONFIDENCE
    assert result.review_rounds == 0
    assert ai.review_calls == 0


def test_technical_failure_is_not_disguised_as_low_quality() -> None:
    ai = FakeAi(
        _assessment(ScopeDecision.AUTO_REVIEW),
        [TimeoutError("provider timeout")],
    )
    result = asyncio.run(AutoScopeReviewService(ai, ai).resolve("问题"))
    assert result.assessment.decision is ScopeDecision.AUTO_REVIEW_ERROR
    assert result.retryable is True
    assert result.error_summary == "TimeoutError"
    assert "自动重试" in result.assessment.reason
