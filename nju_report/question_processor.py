"""Idempotent daily scope screening over locally captured group messages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from .models import ScopeAssessment, ScopeDecision, ScopeResolution, StoredMessage
from .privacy import redact_for_report
from .scope_classifier import AutoScopeReviewService, ScopeBatchMessage
from .storage import ReportStorage
from .time_windows import natural_day_window

_BATCH_MAX_TARGETS = 200
_BATCH_MAX_TARGET_CHARS = 24_000
_BATCH_MESSAGE_CHAR_LIMIT = 1_200


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


class DailyQuestionProcessor:
    """Send every nonempty captured message through full-coverage AI chunks."""

    def __init__(
        self,
        storage: ReportStorage,
        scope_review_service: AutoScopeReviewService,
        *,
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

    @property
    def running(self) -> bool:
        return self._run_lock.locked()

    @property
    def progress(self) -> tuple[str, int, int]:
        return self._progress_date, self._progress_completed, self._progress_total

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

            async def screen(chunk: _ScreeningChunk) -> list[ScopeDecision]:
                async with semaphore:
                    return await self._screen_chunk(
                        chunk,
                        report_date=report_date.isoformat(),
                        review_run_id=run_id,
                    )

            tasks = [asyncio.create_task(screen(chunk)) for chunk in chunks]
            decisions: list[ScopeDecision] = []
            self._progress_date = report_date.isoformat()
            self._progress_total = sum(len(chunk.targets) for chunk in chunks)
            self._progress_completed = 0
            for task in asyncio.as_completed(tasks):
                chunk_decisions = await task
                decisions.extend(chunk_decisions)
                self._progress_completed += len(chunk_decisions)
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
        *,
        report_date: str,
        review_run_id: str,
    ) -> list[ScopeDecision]:
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

        decisions: list[ScopeDecision] = []
        for message_id, index, message in chunk.targets:
            resolution = resolutions[message_id]
            await self._save_resolution(
                message,
                resolution,
                report_date=report_date,
                review_run_id=f"{review_run_id}:{index}",
            )
            decisions.append(resolution.assessment.decision)
        return decisions

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
