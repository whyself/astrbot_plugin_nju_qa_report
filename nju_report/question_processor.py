"""Idempotent daily scope screening over locally captured group messages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from .models import ScopeAssessment, ScopeDecision, ScopeResolution, StoredMessage
from .privacy import redact_for_report
from .scope_classifier import AutoScopeReviewService
from .storage import ReportStorage
from .time_windows import natural_day_window


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


class DailyQuestionProcessor:
    """Send every nonempty captured message through AI scope screening."""

    def __init__(
        self,
        storage: ReportStorage,
        scope_review_service: AutoScopeReviewService,
        *,
        timezone_name: str,
        concurrency: int = 2,
        context_radius: int = 5,
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
            screenable = [
                (index, item)
                for index, item in enumerate(messages)
                if item.text.strip() or item.outline.strip()
            ]
            semaphore = asyncio.Semaphore(self._concurrency)

            async def screen(index: int, message: StoredMessage) -> ScopeDecision:
                async with semaphore:
                    return await self._screen_one(
                        message,
                        context=_nearby_context(messages, index, self._context_radius),
                        report_date=report_date.isoformat(),
                        review_run_id=f"{run_id}:{index}",
                    )

            tasks = [asyncio.create_task(screen(index, item)) for index, item in screenable]
            decisions: list[ScopeDecision] = []
            self._progress_date = report_date.isoformat()
            self._progress_total = len(tasks)
            self._progress_completed = 0
            for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
                decisions.append(await task)
                self._progress_completed = completed
            included_count = sum(item is ScopeDecision.INCLUDE for item in decisions)
            error_count = sum(item is ScopeDecision.AUTO_REVIEW_ERROR for item in decisions)
            dropped_count = len(decisions) - included_count - error_count
            await asyncio.to_thread(
                self._storage.complete_processing_window,
                report_date.isoformat(),
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
            )
            raise
        except Exception as exc:
            error_name = type(exc).__name__
            await asyncio.to_thread(
                self._storage.fail_processing_window,
                report_date.isoformat(),
                error_name,
            )
            return DailyRunResult(
                report_date=report_date.isoformat(),
                status="FAILED",
                error_summary=error_name,
            )

    async def _screen_one(
        self,
        message: StoredMessage,
        *,
        context: str,
        report_date: str,
        review_run_id: str,
    ) -> ScopeDecision:
        content = message.text.strip() or message.outline.strip()
        try:
            resolution = await self._scope_review_service.resolve(content, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # Defensive: the service normally converts errors itself.
            resolution = ScopeResolution(
                assessment=ScopeAssessment(
                    decision=ScopeDecision.AUTO_REVIEW_ERROR,
                    reason="AI 范围审核发生技术错误，已保留本地记录",
                    confidence=0.0,
                ),
                review_rounds=0,
                retryable=True,
                error_summary=type(exc).__name__,
            )

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
        return resolution.assessment.decision


def _nearby_context(messages: list[StoredMessage], index: int, radius: int) -> str:
    if radius == 0:
        return ""
    target = messages[index]
    lines: list[str] = []
    before: list[StoredMessage] = []
    for nearby_index in range(index - 1, -1, -1):
        item = messages[nearby_index]
        if item.group_id != target.group_id:
            continue
        before.append(item)
        if len(before) >= radius:
            break
    after: list[StoredMessage] = []
    for nearby_index in range(index + 1, len(messages)):
        item = messages[nearby_index]
        if item.group_id != target.group_id:
            continue
        after.append(item)
        if len(after) >= radius:
            break
    for relation, item in [
        *(("此前", item) for item in reversed(before)),
        *(("此后", item) for item in after),
    ]:
        content = item.text.strip() or item.outline.strip()
        if not content:
            continue
        lines.append(f"{relation}群聊消息：{content}")
    return "\n".join(lines)


def today_in_timezone(timezone_name: str) -> date:
    return datetime.now(tz=ZoneInfo(timezone_name)).date()
