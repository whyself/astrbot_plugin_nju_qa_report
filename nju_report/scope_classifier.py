"""Automatic scope classification and AI-only review orchestration."""

from __future__ import annotations

from typing import Protocol

from .models import ScopeAssessment, ScopeDecision, ScopeResolution


class ScopeClassifier(Protocol):
    async def classify(self, message: str, context: str) -> ScopeAssessment:
        """Perform the first-pass scope decision."""


class ScopeReviewer(Protocol):
    async def review(
        self,
        message: str,
        context: str,
        *,
        round_no: int,
    ) -> ScopeAssessment:
        """Independently review an uncertain candidate."""


class ScopeAssessmentError(RuntimeError):
    """Raised when a classifier response violates the decision contract."""


class AutoScopeReviewService:
    """Resolve scope decisions without ever creating a human review queue."""

    def __init__(
        self,
        classifier: ScopeClassifier,
        reviewer: ScopeReviewer,
        *,
        enabled: bool = True,
        max_rounds: int = 2,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须大于等于 1")
        self._classifier = classifier
        self._reviewer = reviewer
        self._enabled = enabled
        self._max_rounds = max_rounds

    async def resolve(self, message: str, context: str = "") -> ScopeResolution:
        try:
            initial = _validated(await self._classifier.classify(message, context))
        except Exception as exc:
            return _error_resolution(exc, review_rounds=0)

        if initial.decision in {ScopeDecision.INCLUDE, ScopeDecision.DROP}:
            return ScopeResolution(
                initial,
                review_rounds=0,
                initial_assessment=initial,
            )
        if initial.decision is not ScopeDecision.AUTO_REVIEW:
            return _error_resolution(
                ScopeAssessmentError("初筛模型返回了不允许的决策"),
                review_rounds=0,
                initial=initial,
            )

        if not self._enabled:
            return ScopeResolution(
                ScopeAssessment(
                    decision=ScopeDecision.DROP_LOW_CONFIDENCE,
                    reason="AI 自动复核已关闭，低置信候选自动排除",
                    confidence=initial.confidence,
                    canonical_question=initial.canonical_question,
                    category=initial.category,
                    clarity=initial.clarity,
                    knowledge_value=initial.knowledge_value,
                    time_sensitive=initial.time_sensitive,
                ),
                review_rounds=0,
                initial_assessment=initial,
            )

        last = initial
        review_attempts: list[ScopeAssessment] = []
        for round_no in range(1, self._max_rounds + 1):
            try:
                reviewed = _validated(
                    await self._reviewer.review(
                        message,
                        context,
                        round_no=round_no,
                    )
                )
            except Exception as exc:
                return _error_resolution(
                    exc,
                    review_rounds=round_no,
                    initial=initial,
                    review_attempts=tuple(review_attempts),
                )
            review_attempts.append(reviewed)
            if reviewed.decision in {ScopeDecision.INCLUDE, ScopeDecision.DROP}:
                return ScopeResolution(
                    reviewed,
                    review_rounds=round_no,
                    initial_assessment=initial,
                    review_attempts=tuple(review_attempts),
                )
            if reviewed.decision is not ScopeDecision.AUTO_REVIEW:
                return _error_resolution(
                    ScopeAssessmentError("复核模型返回了不允许的决策"),
                    review_rounds=round_no,
                    initial=initial,
                    review_attempts=tuple(review_attempts),
                )
            last = reviewed

        return ScopeResolution(
            ScopeAssessment(
                decision=ScopeDecision.DROP_LOW_CONFIDENCE,
                reason="AI 自动复核后仍无法形成明确且可沉淀的问题",
                confidence=last.confidence,
                canonical_question=last.canonical_question,
                category=last.category,
                clarity=last.clarity,
                knowledge_value=last.knowledge_value,
                time_sensitive=last.time_sensitive,
            ),
            review_rounds=self._max_rounds,
            initial_assessment=initial,
            review_attempts=tuple(review_attempts),
        )


def _validated(assessment: ScopeAssessment) -> ScopeAssessment:
    if not isinstance(assessment, ScopeAssessment):
        raise ScopeAssessmentError("模型结果不是 ScopeAssessment")
    if assessment.decision not in {
        ScopeDecision.INCLUDE,
        ScopeDecision.AUTO_REVIEW,
        ScopeDecision.DROP,
    }:
        raise ScopeAssessmentError("模型决策不属于允许集合")
    if not assessment.reason.strip():
        raise ScopeAssessmentError("模型必须给出非空判断理由")
    if (
        isinstance(assessment.confidence, bool)
        or not isinstance(assessment.confidence, (int, float))
        or not 0 <= assessment.confidence <= 1
    ):
        raise ScopeAssessmentError("模型置信度必须在 0 到 1 之间")
    if assessment.decision is ScopeDecision.INCLUDE and not assessment.canonical_question.strip():
        raise ScopeAssessmentError("纳入的问题必须有清楚的聚合问题表达")
    return assessment


def _error_resolution(
    exc: Exception,
    *,
    review_rounds: int,
    initial: ScopeAssessment | None = None,
    review_attempts: tuple[ScopeAssessment, ...] = (),
) -> ScopeResolution:
    error_type = type(exc).__name__
    return ScopeResolution(
        ScopeAssessment(
            decision=ScopeDecision.AUTO_REVIEW_ERROR,
            reason="AI 范围审核发生技术错误，将由后台自动重试",
            confidence=0.0,
        ),
        review_rounds=review_rounds,
        initial_assessment=initial,
        review_attempts=review_attempts,
        retryable=True,
        error_summary=error_type,
    )
