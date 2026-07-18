"""Idempotent daily scope screening over locally captured group messages."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from .models import (
    Clarity,
    KnowledgeValue,
    ScopeAssessment,
    ScopeDecision,
    ScopeResolution,
    StoredMessage,
)
from .privacy import redact_for_report
from .scope_classifier import (
    AutoScopeReviewService,
    FinalQuestionReviewer,
    QuestionGateCandidate,
    ScopeBatchMessage,
)
from .storage import ReportStorage
from .time_windows import natural_day_window

logger = logging.getLogger(__name__)

_BATCH_MAX_TARGETS = 200
_BATCH_MAX_TARGET_CHARS = 24_000
_BATCH_MESSAGE_CHAR_LIMIT = 1_200
_FINAL_GATE_BATCH_SIZE = 200


@dataclass(frozen=True, slots=True)
class DailyRunResult:
    report_date: str
    status: str
    messages_scanned: int = 0
    candidates_saved: int = 0
    included_count: int = 0
    dropped_count: int = 0
    error_count: int = 0
    error_summary: str = ""

    @property
    def skipped(self) -> bool:
        return self.status in {"SKIPPED_COMPLETED", "SKIPPED_REPORT_COMPLETE"}


@dataclass(frozen=True, slots=True)
class _ScreeningChunk:
    messages: tuple[ScopeBatchMessage, ...]
    targets: tuple[tuple[str, int, StoredMessage], ...]


@dataclass(frozen=True, slots=True)
class _ScreenedTarget:
    message_id: str
    index: int
    message: StoredMessage
    resolution: ScopeResolution


@dataclass(frozen=True, slots=True)
class _GateGroup:
    candidate: QuestionGateCandidate
    targets: tuple[_ScreenedTarget, ...]


class FinalGateBatchError(RuntimeError):
    """A sanitized, persistable description of one failed final-gate batch."""


class DailyQuestionProcessor:
    """Send every nonempty captured message through full-coverage AI chunks."""

    def __init__(
        self,
        storage: ReportStorage,
        scope_review_service: AutoScopeReviewService,
        *,
        final_question_reviewer: FinalQuestionReviewer | None = None,
        timezone_name: str,
        concurrency: int = 2,
        context_radius: int = 30,
        ignored_sender_ids: tuple[str, ...] = (),
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency 必须大于 0")
        if context_radius < 0:
            raise ValueError("context_radius 不能小于 0")
        self._storage = storage
        self._scope_review_service = scope_review_service
        self._final_question_reviewer = final_question_reviewer
        self._timezone_name = timezone_name
        self._concurrency = concurrency
        self._context_radius = context_radius
        self._ignored_sender_ids = frozenset(
            str(item).strip() for item in ignored_sender_ids if str(item).strip()
        )
        self._run_lock = asyncio.Lock()
        self._progress_date = ""
        self._progress_completed = 0
        self._progress_total = 0
        self._progress_phase = "消息初筛"

    @property
    def running(self) -> bool:
        return self._run_lock.locked()

    @property
    def progress(self) -> tuple[str, int, int]:
        return self._progress_date, self._progress_completed, self._progress_total

    @property
    def progress_phase(self) -> str:
        return self._progress_phase

    async def process_date(
        self,
        report_date: date,
        *,
        force: bool = False,
    ) -> DailyRunResult:
        async with self._run_lock:
            return await self._process_date(report_date, force=force)

    async def process_all_history(self, *, before_date: date) -> list[DailyRunResult]:
        """Process every local date before ``before_date`` and skip completed dates."""

        async with self._run_lock:
            raw_dates = await asyncio.to_thread(
                self._storage.message_local_dates,
                self._timezone_name,
            )
            dates = [date.fromisoformat(item) for item in raw_dates]
            return [await self._process_date(item) for item in dates if item < before_date]

    async def _process_date(
        self,
        report_date: date,
        *,
        force: bool = False,
    ) -> DailyRunResult:
        window = natural_day_window(report_date, self._timezone_name)
        run_id = uuid4().hex
        should_run = await asyncio.to_thread(
            self._storage.begin_processing_window,
            window,
            run_id=run_id,
            force=force,
        )
        if not should_run:
            existing = await asyncio.to_thread(
                self._storage.processing_window,
                report_date.isoformat(),
            )
            return DailyRunResult(
                report_date=report_date.isoformat(),
                status="SKIPPED_COMPLETED",
                messages_scanned=existing.messages_scanned if existing else 0,
                candidates_saved=existing.candidates_saved if existing else 0,
                included_count=existing.included_count if existing else 0,
                dropped_count=existing.dropped_count if existing else 0,
                error_count=existing.error_count if existing else 0,
            )

        try:
            messages = await asyncio.to_thread(
                self._storage.messages_in_window,
                window,
            )
            chunks = _screening_chunks(
                messages,
                context_radius=self._context_radius,
                conversation_date=report_date.isoformat(),
                ignored_sender_ids=self._ignored_sender_ids,
            )
            semaphore = asyncio.Semaphore(self._concurrency)

            async def screen(chunk: _ScreeningChunk) -> list[_ScreenedTarget]:
                async with semaphore:
                    return await self._screen_chunk(chunk)

            tasks = [asyncio.create_task(screen(chunk)) for chunk in chunks]
            screened_targets: list[_ScreenedTarget] = []
            self._progress_date = report_date.isoformat()
            self._progress_phase = "消息初筛"
            self._progress_total = sum(len(chunk.targets) for chunk in chunks)
            self._progress_completed = 0
            for task in asyncio.as_completed(tasks):
                chunk_targets = await task
                screened_targets.extend(chunk_targets)
                self._progress_completed += len(chunk_targets)

            screened_targets = await self._apply_final_question_gate(
                screened_targets,
                report_date=report_date.isoformat(),
            )
            screened_targets.sort(key=lambda item: item.index)
            for item in screened_targets:
                await self._save_resolution(
                    item.message,
                    item.resolution,
                    report_date=report_date.isoformat(),
                    review_run_id=f"{run_id}:{item.index}",
                )

            decisions = [item.resolution.assessment.decision for item in screened_targets]
            included_count = sum(item is ScopeDecision.INCLUDE for item in decisions)
            error_count = sum(item is ScopeDecision.AUTO_REVIEW_ERROR for item in decisions)
            dropped_count = len(decisions) - included_count - error_count
            await asyncio.to_thread(
                self._storage.complete_processing_window,
                report_date.isoformat(),
                run_id=run_id,
                messages_scanned=len(messages),
                candidates_saved=len(decisions),
                included_count=included_count,
                dropped_count=dropped_count,
                error_count=error_count,
            )
            final_status = "COMPLETED" if error_count == 0 else "RETRY_PENDING"
            return DailyRunResult(
                report_date=report_date.isoformat(),
                status=final_status,
                messages_scanned=len(messages),
                candidates_saved=len(decisions),
                included_count=included_count,
                dropped_count=dropped_count,
                error_count=error_count,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(
                self._storage.fail_processing_window,
                report_date.isoformat(),
                "CancelledError",
                run_id=run_id,
            )
            raise
        except Exception as exc:
            error_name = type(exc).__name__
            await asyncio.to_thread(
                self._storage.fail_processing_window,
                report_date.isoformat(),
                error_name,
                run_id=run_id,
            )
            return DailyRunResult(
                report_date=report_date.isoformat(),
                status="FAILED",
                error_summary=error_name,
            )

    async def _screen_chunk(
        self,
        chunk: _ScreeningChunk,
    ) -> list[_ScreenedTarget]:
        target_ids = tuple(item[0] for item in chunk.targets)
        try:
            resolutions = await self._scope_review_service.resolve_batch(
                chunk.messages,
                target_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Defensive: the service normally converts errors itself.
            resolutions = {
                message_id: ScopeResolution(
                    assessment=ScopeAssessment(
                        decision=ScopeDecision.AUTO_REVIEW_ERROR,
                        reason="AI 范围审核发生技术错误，已保留本地记录",
                        confidence=0.0,
                    ),
                    review_rounds=0,
                    retryable=True,
                    error_summary=type(exc).__name__,
                )
                for message_id in target_ids
            }

        return [
            _ScreenedTarget(message_id, index, message, resolutions[message_id])
            for message_id, index, message in chunk.targets
        ]

    async def _apply_final_question_gate(
        self,
        targets: list[_ScreenedTarget],
        *,
        report_date: str,
    ) -> list[_ScreenedTarget]:
        if self._final_question_reviewer is None:
            return targets
        groups = _final_gate_groups(targets, report_date=report_date)
        if not groups:
            return targets

        self._progress_phase = "最终问题复核"
        self._progress_completed = 0
        self._progress_total = len(groups)
        final_assessments: dict[str, ScopeAssessment] = {}
        batches = [
            groups[start : start + _FINAL_GATE_BATCH_SIZE]
            for start in range(0, len(groups), _FINAL_GATE_BATCH_SIZE)
        ]
        for batch_no, batch in enumerate(batches, start=1):
            started_at = time.monotonic()
            logger.info(
                "NJU final gate batch started report_date=%s batch=%d/%d "
                "batch_candidates=%d total_candidates=%d",
                report_date,
                batch_no,
                len(batches),
                len(batch),
                len(groups),
            )
            try:
                reviewed = await self._final_question_reviewer.final_review_batch(
                    [item.candidate for item in batch]
                )
                expected = {item.candidate.candidate_id for item in batch}
                if set(reviewed) != expected:
                    raise RuntimeError("最终问题 AI 闸门未完整返回候选 ID")
                final_assessments.update(reviewed)
                self._progress_completed += len(batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                detail = " ".join(str(exc).split()).strip()
                audit_error = FinalGateBatchError(
                    f"report_date={report_date}; batch={batch_no}/{len(batches)}; "
                    f"batch_candidates={len(batch)}; total_candidates={len(groups)}; "
                    f"elapsed_seconds={time.monotonic() - started_at:.3f}; "
                    f"cause={type(exc).__name__}"
                    + (f": {detail}" if detail else "")
                )
                logger.warning("NJU final gate batch failed: %s", audit_error)
                return _final_gate_error_targets(targets, groups, audit_error)
            logger.info(
                "NJU final gate batch completed report_date=%s batch=%d/%d "
                "batch_candidates=%d total_candidates=%d elapsed_seconds=%.3f",
                report_date,
                batch_no,
                len(batches),
                len(batch),
                len(groups),
                time.monotonic() - started_at,
            )

        by_message_id: dict[str, ScopeAssessment] = {}
        for group in groups:
            assessment = final_assessments[group.candidate.candidate_id]
            if not _valid_final_assessment(assessment):
                return _final_gate_error_targets(
                    targets,
                    groups,
                    RuntimeError("最终问题 AI 闸门返回了无效终态"),
                )
            for target in group.targets:
                by_message_id[target.message_id] = assessment

        return [
            _with_final_assessment(item, by_message_id[item.message_id])
            if item.message_id in by_message_id
            else item
            for item in targets
        ]

    async def _save_resolution(
        self,
        message: StoredMessage,
        resolution: ScopeResolution,
        *,
        report_date: str,
        review_run_id: str,
    ) -> None:
        content = message.text.strip() or message.outline.strip()

        source_key = ":".join(
            (
                "message",
                message.platform_id,
                message.bot_self_id,
                message.external_message_id,
            )
        )
        await asyncio.to_thread(
            self._storage.save_scope_resolution,
            source_key=source_key,
            report_date=report_date,
            review_run_id=review_run_id,
            resolution=resolution,
            original_question=redact_for_report(content),
            group_alias=message.group_alias,
            sent_at_utc=message.sent_at_utc,
        )


def _final_gate_groups(
    targets: list[_ScreenedTarget],
    *,
    report_date: str,
) -> list[_GateGroup]:
    included = sorted(
        (
            item
            for item in targets
            if item.resolution.assessment.decision is ScopeDecision.INCLUDE
        ),
        key=lambda item: item.index,
    )
    grouped: dict[str, list[_ScreenedTarget]] = {}
    for item in included:
        question = " ".join(item.resolution.assessment.canonical_question.split()).strip()
        if not question:
            continue
        grouped.setdefault(question.casefold(), []).append(item)

    result: list[_GateGroup] = []
    for number, items in enumerate(grouped.values(), start=1):
        assessment = items[0].resolution.assessment
        result.append(
            _GateGroup(
                candidate=QuestionGateCandidate(
                    candidate_id=f"c{number}",
                    canonical_question=assessment.canonical_question,
                    category=assessment.category,
                    time_sensitive=assessment.time_sensitive,
                    source_count=len(items),
                    report_date=report_date,
                ),
                targets=tuple(items),
            )
        )
    return result


def _with_final_assessment(
    target: _ScreenedTarget,
    final: ScopeAssessment,
) -> _ScreenedTarget:
    previous = target.resolution
    if final.decision is ScopeDecision.DROP and not final.canonical_question:
        prior = previous.assessment
        final = ScopeAssessment(
            decision=ScopeDecision.DROP,
            reason=final.reason,
            confidence=final.confidence,
            canonical_question=prior.canonical_question,
            category=prior.category,
            clarity=final.clarity,
            knowledge_value=final.knowledge_value,
            time_sensitive=prior.time_sensitive,
        )
    return _ScreenedTarget(
        target.message_id,
        target.index,
        target.message,
        ScopeResolution(
            assessment=final,
            review_rounds=previous.review_rounds + 1,
            initial_assessment=previous.initial_assessment or previous.assessment,
            review_attempts=previous.review_attempts + (final,),
        ),
    )


def _valid_final_assessment(assessment: ScopeAssessment) -> bool:
    if not isinstance(assessment, ScopeAssessment):
        return False
    if assessment.decision is ScopeDecision.DROP:
        return bool(assessment.reason.strip())
    return bool(
        assessment.decision is ScopeDecision.INCLUDE
        and assessment.reason.strip()
        and assessment.canonical_question.strip()
        and assessment.clarity is Clarity.CLEAR
        and assessment.knowledge_value in {KnowledgeValue.HIGH, KnowledgeValue.MEDIUM}
        and 0 <= assessment.confidence <= 1
    )


def _final_gate_error_targets(
    targets: list[_ScreenedTarget],
    groups: list[_GateGroup],
    exc: Exception,
) -> list[_ScreenedTarget]:
    affected = {
        target.message_id
        for group in groups
        for target in group.targets
    }
    error_name = type(exc).__name__
    detail = " ".join(str(exc).split()).strip()
    summary = f"{error_name}: {detail}" if detail else error_name
    result: list[_ScreenedTarget] = []
    for target in targets:
        if target.message_id not in affected:
            result.append(target)
            continue
        previous = target.resolution
        prior = previous.assessment
        result.append(
            _ScreenedTarget(
                target.message_id,
                target.index,
                target.message,
                ScopeResolution(
                    assessment=ScopeAssessment(
                        decision=ScopeDecision.AUTO_REVIEW_ERROR,
                        reason="最终问题 AI 闸门发生技术错误，将由后台自动重试",
                        confidence=0.0,
                        canonical_question=prior.canonical_question,
                        category=prior.category,
                        clarity=prior.clarity,
                        knowledge_value=prior.knowledge_value,
                        time_sensitive=prior.time_sensitive,
                    ),
                    review_rounds=previous.review_rounds + 1,
                    initial_assessment=previous.initial_assessment or prior,
                    review_attempts=previous.review_attempts,
                    retryable=True,
                    error_summary=summary[:1000],
                ),
            )
        )
    return result


def _screening_chunks(
    messages: list[StoredMessage],
    *,
    context_radius: int,
    conversation_date: str = "",
    ignored_sender_ids: frozenset[str] | None = None,
) -> list[_ScreeningChunk]:
    groups: dict[str, list[tuple[int, StoredMessage]]] = {}
    for index, item in enumerate(messages):
        if item.sender_id in (ignored_sender_ids or ()) or (
            item.sender_id and item.sender_id == item.bot_self_id
        ):
            continue
        if not (item.text.strip() or item.outline.strip()):
            continue
        group_key = item.group_id or item.session_id
        groups.setdefault(group_key, []).append((index, item))

    chunks: list[_ScreeningChunk] = []
    for group_entries in groups.values():
        external_to_internal = {
            message.external_message_id: f"m{global_index}"
            for global_index, message in group_entries
            if message.external_message_id
        }
        speaker_aliases: dict[str, str] = {}
        for global_index, message in group_entries:
            speaker_key = message.sender_id or message.sender_name or f"unknown:{global_index}"
            speaker_aliases.setdefault(speaker_key, f"u{len(speaker_aliases) + 1}")
        start = 0
        while start < len(group_entries):
            end = start
            target_chars = 0
            while end < len(group_entries) and end - start < _BATCH_MAX_TARGETS:
                content = _message_content(group_entries[end][1])
                next_chars = min(len(content), _BATCH_MESSAGE_CHAR_LIMIT) + 80
                if end > start and target_chars + next_chars > _BATCH_MAX_TARGET_CHARS:
                    break
                target_chars += next_chars
                end += 1

            context_start = max(0, start - context_radius)
            context_end = min(len(group_entries), end + context_radius)
            batch_messages: list[ScopeBatchMessage] = []
            targets: list[tuple[str, int, StoredMessage]] = []
            for local_index in range(context_start, context_end):
                global_index, message = group_entries[local_index]
                message_id = f"m{global_index}"
                speaker_key = (
                    message.sender_id or message.sender_name or f"unknown:{global_index}"
                )
                is_target = start <= local_index < end
                batch_messages.append(
                    ScopeBatchMessage(
                        message_id=message_id,
                        content=_message_content(message),
                        context_only=not is_target,
                        speaker_id=speaker_aliases[speaker_key],
                        reply_to_id=external_to_internal.get(message.reply_to_message_id, ""),
                        conversation_date=conversation_date,
                    )
                )
                if is_target:
                    targets.append((message_id, global_index, message))
            chunks.append(_ScreeningChunk(tuple(batch_messages), tuple(targets)))
            start = end
    return chunks


def _message_content(message: StoredMessage) -> str:
    return message.text.strip() or message.outline.strip()


def today_in_timezone(timezone_name: str) -> date:
    return datetime.now(tz=ZoneInfo(timezone_name)).date()
