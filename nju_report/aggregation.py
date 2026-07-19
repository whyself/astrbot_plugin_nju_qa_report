"""Conservative question aggregation and community-answer association."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date
from difflib import SequenceMatcher

from .answer_agent import CommunityAnswerAgent, DiscoveredQuestion
from .merge_markers import final_merge_marker
from .models import (
    CommunityContextAudit,
    CommunityContextDegradationReason,
    QuestionCandidate,
    QuestionCluster,
    StoredMessage,
)
from .storage import ReportStorage
from .time_windows import natural_day_window

logger = logging.getLogger(__name__)

_CANONICAL_RESTORED_ACTION = "CANONICAL_QUESTION_RESTORED_FROM_SCREENING"
_SCREENING_MERGE_LOCK_ACTION = "SCREENING_MERGE_LOCK_ENFORCED"
_CONTEXT_DEPENDENT_TITLE_RE = re.compile(
    r"(?:这玩意|那玩意|这东西|那东西|这个(?:呢|吗|是|怎么|如何)|"
    r"那个(?:呢|吗|是|怎么|如何)|上述|前者|后者|它(?:呢|是|怎么|如何)|"
    r"又(?:改|调整|变)(?:了|吗|没|没有)?)"
)


@dataclass(slots=True)
class _ClusterBuilder:
    candidates: list[QuestionCandidate] = field(default_factory=list)
    merge_marker: str = ""

    @property
    def representative(self) -> QuestionCandidate:
        return max(
            self.candidates, key=lambda item: (len(item.canonical_question), -item.sent_at_utc)
        )

    @property
    def merge_locked(self) -> bool:
        return bool(self.merge_marker) and len(self.candidates) > 1


class QuestionAggregationService:
    """Group equivalent canonical questions without hiding ambiguous differences."""

    def __init__(
        self,
        storage: ReportStorage,
        answer_agent: CommunityAnswerAgent,
        *,
        timezone_name: str,
        concurrency: int = 3,
    ) -> None:
        self._storage = storage
        self._answer_agent = answer_agent
        self._timezone_name = timezone_name
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._progress_date = ""
        self._progress_completed = 0
        self._progress_total = 0

    @property
    def progress(self) -> tuple[str, int, int]:
        return self._progress_date, self._progress_completed, self._progress_total

    async def aggregate_date(self, report_date: date) -> list[QuestionCluster]:
        candidates, _ = await asyncio.to_thread(
            self._storage.list_question_candidates,
            report_date=report_date.isoformat(),
            limit=None,
        )
        included = [item for item in candidates if item.final_decision == "INCLUDE"]
        window = natural_day_window(report_date, self._timezone_name)
        messages = await asyncio.to_thread(self._storage.messages_in_window, window)
        message_by_external_id = {item.external_message_id: item for item in messages}
        candidate_by_source_key = {item.source_key: item for item in included}
        clusters = _aggregate(included)
        self._progress_date = report_date.isoformat()
        self._progress_completed = 0
        self._progress_total = len(clusters)
        async def attach(cluster: QuestionCluster) -> tuple[QuestionCluster, ...]:
            async with self._semaphore:
                collection_error = ""
                try:
                    discovery = await self._answer_agent.collect(cluster, messages)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "NJU community-answer context agent failed for %s",
                        cluster.question_code,
                    )
                    collection_error = type(exc).__name__
                    discovery = None
                if discovery is None or not discovery.questions:
                    degradation_reason = (
                        discovery.community_context_degradation_reason
                        if discovery is not None
                        else CommunityContextDegradationReason.AGENT_EXCEPTION
                    )
                    audit = (
                        discovery.community_context_audit
                        if discovery is not None
                        else CommunityContextAudit(
                            initial_errors=(collection_error or "unknown agent exception",),
                            fallback_actions=("EMPTY_CLUSTER_SAFE_FALLBACK",),
                        )
                    )
                    results = (
                        replace(
                            cluster,
                            representative_questions=(cluster.canonical_question,),
                            answers=(),
                            community_context_degraded=True,
                            community_context_degradation_reason=degradation_reason,
                            community_context_audit=audit,
                        ),
                    )
                else:
                    source_by_external_id = {
                        external_id: source_key
                        for source_key in cluster.candidate_source_keys
                        if (external_id := _external_id_from_source_key(source_key))
                    }
                    split_results: list[QuestionCluster] = []
                    discovered_questions = discovery.questions
                    if (
                        cluster.screening_merge_locked
                        and len(discovered_questions) > 1
                    ):
                        discovered_questions = (
                            _collapse_screening_locked_questions(
                                cluster,
                                discovered_questions,
                            ),
                        )
                    is_true_split = len(discovered_questions) > 1
                    for question in discovered_questions:
                        refined_source_keys = tuple(
                            source_by_external_id[item]
                            for item in question.question_message_ids
                            if item in source_by_external_id
                        )
                        if cluster.screening_merge_locked:
                            refined_source_keys = cluster.candidate_source_keys
                        refined_messages = [
                            message_by_external_id[item]
                            for item in question.question_message_ids
                            if item in message_by_external_id
                        ]
                        question_candidates = [
                            candidate_by_source_key[item]
                            for item in refined_source_keys
                            if item in candidate_by_source_key
                        ]
                        question_code = (
                            min(item.question_code for item in question_candidates)
                            if question_candidates
                            else cluster.question_code
                        )
                        canonical_question, canonical_restored = (
                            _resolved_canonical_question(
                                proposed=question.canonical_question,
                                baseline=cluster.canonical_question,
                                candidates=question_candidates,
                                allow_split_refinement=is_true_split,
                            )
                        )
                        context_audit = question.community_context_audit
                        if canonical_restored:
                            logger.warning(
                                "NJU restored screening canonical question_code=%s "
                                "proposed=%r baseline=%r",
                                question_code,
                                question.canonical_question,
                                canonical_question,
                            )
                            context_audit = replace(
                                context_audit,
                                fallback_actions=tuple(
                                    dict.fromkeys(
                                        context_audit.fallback_actions
                                        + (_CANONICAL_RESTORED_ACTION,)
                                    )
                                ),
                            )
                        split_results.append(
                            replace(
                                cluster,
                                question_code=question_code,
                                canonical_question=canonical_question,
                                category=question.category or cluster.category,
                                candidate_source_keys=(
                                    refined_source_keys or cluster.candidate_source_keys
                                ),
                                representative_questions=(canonical_question,),
                                first_sent_at_utc=(
                                    min(item.sent_at_utc for item in refined_messages)
                                    if refined_messages
                                    else cluster.first_sent_at_utc
                                ),
                                last_sent_at_utc=(
                                    max(item.sent_at_utc for item in refined_messages)
                                    if refined_messages
                                    else cluster.last_sent_at_utc
                                ),
                                answers=question.answers,
                                community_context_degraded=(
                                    question.community_context_degraded
                                ),
                                community_context_degradation_reason=(
                                    question.community_context_degradation_reason
                                ),
                                community_context_audit=context_audit,
                            )
                        )
                    results = tuple(split_results)
            self._progress_completed += 1
            return results

        attached = await asyncio.gather(*(attach(item) for item in clusters))
        clusters = sorted(
            (item for group in attached for item in group),
            key=lambda item: item.question_code,
        )
        await asyncio.to_thread(
            self._storage.save_question_clusters,
            report_date.isoformat(),
            clusters,
        )
        return clusters


def _aggregate(
    candidates: list[QuestionCandidate],
    messages: list[StoredMessage] | None = None,
) -> list[QuestionCluster]:
    del messages  # Kept for compatibility; answer discovery is now Agent-driven.
    builders: list[_ClusterBuilder] = []
    unlocked: list[QuestionCandidate] = []
    locked_by_marker: dict[str, _ClusterBuilder] = {}
    for candidate in sorted(candidates, key=lambda item: (item.sent_at_utc, item.question_code)):
        marker = final_merge_marker(candidate.reason)
        if not marker:
            unlocked.append(candidate)
            continue
        builder = locked_by_marker.get(marker)
        if builder is None:
            builder = _ClusterBuilder(merge_marker=marker)
            locked_by_marker[marker] = builder
            builders.append(builder)
        builder.candidates.append(candidate)

    for candidate in unlocked:
        match = next(
            (
                builder
                for builder in builders
                if not builder.merge_marker
                and _same_question(candidate, builder.representative)
            ),
            None,
        )
        if match is None:
            builders.append(_ClusterBuilder([candidate]))
        else:
            match.candidates.append(candidate)

    result: list[QuestionCluster] = []
    for builder in builders:
        ordered = sorted(
            builder.candidates, key=lambda item: (item.sent_at_utc, item.question_code)
        )
        representative = builder.representative
        result.append(
            QuestionCluster(
                question_code=min(item.question_code for item in ordered),
                report_date=representative.report_date,
                canonical_question=representative.canonical_question,
                category=representative.category,
                candidate_source_keys=tuple(item.source_key for item in ordered),
                representative_questions=tuple(
                    dict.fromkeys(
                        item.original_question for item in ordered if item.original_question
                    )
                )[:5],
                group_aliases=tuple(
                    sorted({item.group_alias for item in ordered if item.group_alias})
                ),
                first_sent_at_utc=min(item.sent_at_utc for item in ordered),
                last_sent_at_utc=max(item.sent_at_utc for item in ordered),
                screening_merge_locked=builder.merge_locked,
            )
        )
    return sorted(result, key=lambda item: item.question_code)


def _same_question(left: QuestionCandidate, right: QuestionCandidate) -> bool:
    if left.category and right.category and left.category != right.category:
        return False
    left_text = _normalize(left.canonical_question)
    right_text = _normalize(right.canonical_question)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    ratio = SequenceMatcher(None, left_text, right_text).ratio()
    if ratio >= 0.88:
        return True
    left_terms = _bigrams(left_text)
    right_terms = _bigrams(right_text)
    union = left_terms | right_terms
    return bool(union) and len(left_terms & right_terms) / len(union) >= 0.72
def _collapse_screening_locked_questions(
    cluster: QuestionCluster,
    questions: tuple[DiscoveredQuestion, ...],
) -> DiscoveredQuestion:
    question_ids = tuple(
        dict.fromkeys(
            external_id
            for question in questions
            for external_id in question.question_message_ids
        )
    )
    answers_by_id = {
        answer.external_message_id: answer
        for question in questions
        for answer in question.answers
    }
    audits = [question.community_context_audit for question in questions]
    degradation_reason = next(
        (
            question.community_context_degradation_reason
            for question in questions
            if question.community_context_degradation_reason
            is not CommunityContextDegradationReason.NONE
        ),
        CommunityContextDegradationReason.NONE,
    )
    audit = CommunityContextAudit(
        initial_errors=tuple(
            dict.fromkeys(item for audit in audits for item in audit.initial_errors)
        ),
        retry_errors=tuple(
            dict.fromkeys(item for audit in audits for item in audit.retry_errors)
        ),
        retained_question_ids=tuple(
            dict.fromkeys(item for audit in audits for item in audit.retained_question_ids)
        ),
        degraded_question_ids=tuple(
            dict.fromkeys(item for audit in audits for item in audit.degraded_question_ids)
        ),
        fallback_actions=tuple(
            dict.fromkeys(
                (
                    *(item for audit in audits for item in audit.fallback_actions),
                    _SCREENING_MERGE_LOCK_ACTION,
                )
            )
        ),
        event_id=next((audit.event_id for audit in audits if audit.event_id), ""),
    )
    return DiscoveredQuestion(
        question_message_ids=question_ids,
        answers=tuple(answers_by_id.values()),
        canonical_question=cluster.canonical_question,
        category=cluster.category,
        community_context_degraded=any(
            question.community_context_degraded for question in questions
        ),
        community_context_degradation_reason=degradation_reason,
        community_context_audit=audit,
    )


def _resolved_canonical_question(
    *,
    proposed: str,
    baseline: str,
    candidates: list[QuestionCandidate],
    allow_split_refinement: bool,
) -> tuple[str, bool]:
    candidate_baseline = _candidate_baseline(candidates) or baseline
    proposed = proposed.strip()
    if not proposed:
        return candidate_baseline, False
    if not allow_split_refinement:
        return candidate_baseline, _normalize(proposed) != _normalize(candidate_baseline)
    if context_dependent_question_title(proposed):
        return candidate_baseline, _normalize(proposed) != _normalize(candidate_baseline)
    proposed_text = _normalize(proposed)
    baseline_text = _normalize(candidate_baseline)
    if len(proposed_text) < 6 or not (_bigrams(proposed_text) & _bigrams(baseline_text)):
        return candidate_baseline, _normalize(proposed) != _normalize(candidate_baseline)
    return proposed, False


def _candidate_baseline(candidates: list[QuestionCandidate]) -> str:
    if not candidates:
        return ""
    representative = max(
        candidates,
        key=lambda item: (len(item.canonical_question), -item.sent_at_utc),
    )
    return representative.canonical_question.strip()


def context_dependent_question_title(value: str) -> bool:
    normalized = re.sub(r"\s+", "", value)
    return bool(_CONTEXT_DEPENDENT_TITLE_RE.search(normalized))


def _external_id_from_source_key(source_key: str) -> str:
    parts = source_key.split(":", 3)
    return parts[3] if len(parts) == 4 and parts[0] == "message" else ""


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]", "", value.casefold())


def _bigrams(value: str) -> set[str]:
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}
