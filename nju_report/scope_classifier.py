"""Automatic scope classification and AI-only review orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .models import ScopeAssessment, ScopeDecision, ScopeResolution


@dataclass(frozen=True, slots=True)
class ScopeBatchMessage:
    message_id: str
    content: str
    context_only: bool = False
    speaker_id: str = ""
    reply_to_id: str = ""
    conversation_date: str = ""


@dataclass(frozen=True, slots=True)
class QuestionGateCandidate:
    """One already-extracted question sent to the low-cost final AI gate."""

    candidate_id: str
    canonical_question: str
    category: str = ""
    time_sensitive: bool = False
    source_count: int = 1
    report_date: str = ""


class ScopeClassifier(Protocol):
    async def classify(self, message: str, context: str) -> ScopeAssessment:
        """Perform the first-pass scope decision."""

    async def classify_batch(
        self,
        messages: Sequence[ScopeBatchMessage],
        target_ids: Sequence[str],
    ) -> dict[str, ScopeAssessment]:
        """Classify every target in one ordered chat chunk."""


class ScopeReviewer(Protocol):
    async def review(
        self,
        message: str,
        context: str,
        *,
        round_no: int,
    ) -> ScopeAssessment:
        """Independently review an uncertain candidate."""

    async def review_batch(
        self,
        messages: Sequence[ScopeBatchMessage],
        target_ids: Sequence[str],
        *,
        round_no: int,
    ) -> dict[str, ScopeAssessment]:
        """Independently review unresolved targets in one chat chunk."""


class FinalQuestionReviewer(Protocol):
    async def final_review_batch(
        self,
        candidates: Sequence[QuestionGateCandidate],
    ) -> dict[str, ScopeAssessment]:
        """Keep, rewrite, drop, or merge extracted questions without raw chat."""


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

    async def resolve_batch(
        self,
        messages: Sequence[ScopeBatchMessage],
        target_ids: Sequence[str],
    ) -> dict[str, ScopeResolution]:
        ordered_ids = tuple(dict.fromkeys(str(item) for item in target_ids))
        if not ordered_ids:
            return {}
        try:
            initial_raw = await self._classifier.classify_batch(messages, ordered_ids)
            _require_exact_ids(initial_raw, ordered_ids)
            initial = {item: _validated(initial_raw[item]) for item in ordered_ids}
        except Exception as exc:
            return {
                item: _error_resolution(exc, review_rounds=0)
                for item in ordered_ids
            }

        resolved: dict[str, ScopeResolution] = {}
        pending: list[str] = []
        attempts: dict[str, list[ScopeAssessment]] = {item: [] for item in ordered_ids}
        for item in ordered_ids:
            assessment = initial[item]
            if assessment.decision in {ScopeDecision.INCLUDE, ScopeDecision.DROP}:
                resolved[item] = ScopeResolution(
                    assessment,
                    review_rounds=0,
                    initial_assessment=assessment,
                )
            elif assessment.decision is ScopeDecision.AUTO_REVIEW:
                pending.append(item)
            else:
                resolved[item] = _error_resolution(
                    ScopeAssessmentError("批量初筛模型返回了不允许的决策"),
                    review_rounds=0,
                    initial=assessment,
                )

        if not pending:
            return resolved
        if not self._enabled:
            for item in pending:
                assessment = initial[item]
                resolved[item] = ScopeResolution(
                    ScopeAssessment(
                        decision=ScopeDecision.DROP_LOW_CONFIDENCE,
                        reason="AI 自动复核已关闭，低置信候选自动排除",
                        confidence=assessment.confidence,
                        canonical_question=assessment.canonical_question,
                        category=assessment.category,
                        clarity=assessment.clarity,
                        knowledge_value=assessment.knowledge_value,
                        time_sensitive=assessment.time_sensitive,
                    ),
                    review_rounds=0,
                    initial_assessment=assessment,
                )
            return resolved

        for round_no in range(1, self._max_rounds + 1):
            try:
                reviewed_raw = await self._reviewer.review_batch(
                    messages,
                    pending,
                    round_no=round_no,
                )
                _require_exact_ids(reviewed_raw, pending)
                reviewed = {item: _validated(reviewed_raw[item]) for item in pending}
            except Exception as exc:
                for item in pending:
                    resolved[item] = _error_resolution(
                        exc,
                        review_rounds=round_no,
                        initial=initial[item],
                        review_attempts=tuple(attempts[item]),
                    )
                return resolved

            next_pending: list[str] = []
            for item in pending:
                assessment = reviewed[item]
                attempts[item].append(assessment)
                if assessment.decision in {ScopeDecision.INCLUDE, ScopeDecision.DROP}:
                    resolved[item] = ScopeResolution(
                        assessment,
                        review_rounds=round_no,
                        initial_assessment=initial[item],
                        review_attempts=tuple(attempts[item]),
                    )
                elif assessment.decision is ScopeDecision.AUTO_REVIEW:
                    next_pending.append(item)
                else:
                    resolved[item] = _error_resolution(
                        ScopeAssessmentError("批量复核模型返回了不允许的决策"),
                        review_rounds=round_no,
                        initial=initial[item],
                        review_attempts=tuple(attempts[item]),
                    )
            pending = next_pending
            if not pending:
                return resolved

        for item in pending:
            last = attempts[item][-1] if attempts[item] else initial[item]
            resolved[item] = ScopeResolution(
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
                initial_assessment=initial[item],
                review_attempts=tuple(attempts[item]),
            )
        return resolved


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


def _require_exact_ids(
    assessments: dict[str, ScopeAssessment],
    expected_ids: Sequence[str],
) -> None:
    if not isinstance(assessments, dict) or set(assessments) != set(expected_ids):
        raise ScopeAssessmentError("批量模型结果未精确覆盖全部目标消息 ID")


def _error_resolution(
    exc: Exception,
    *,
    review_rounds: int,
    initial: ScopeAssessment | None = None,
    review_attempts: tuple[ScopeAssessment, ...] = (),
) -> ScopeResolution:
    error_type = type(exc).__name__
    error_detail = " ".join(str(exc).split()).strip()
    error_summary = f"{error_type}: {error_detail}" if error_detail else error_type
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
        error_summary=error_summary[:1000],
    )
