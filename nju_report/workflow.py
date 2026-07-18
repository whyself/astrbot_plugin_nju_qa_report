"""End-to-end daily report workflow and restart-safe scheduler."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .aggregation import QuestionAggregationService
from .investigation import InvestigationService
from .knowledge import KnowledgeService
from .models import ReportArtifact, ScheduledReportRun
from .question_processor import DailyQuestionProcessor, DailyRunResult
from .reporting import DeliverySummary, ReportService, recipient_hash
from .storage import ReportStorage
from .token_usage import TokenUsage, TokenUsageTracker

logger = logging.getLogger(__name__)

_RUNNING_STALE_AFTER_SECONDS = 6 * 60 * 60


@dataclass(frozen=True, slots=True)
class FullReportRunResult:
    screening: DailyRunResult
    cluster_count: int
    report: ReportArtifact | None
    delivery: DeliverySummary | None = None
    token_usage: TokenUsage = TokenUsage()


@dataclass(frozen=True, slots=True)
class WorkflowProgress:
    running: bool
    stage: str
    report_date: str
    date_index: int
    date_total: int
    screening_phase: str
    screening_completed: int
    screening_total: int
    aggregation_completed: int
    aggregation_total: int
    investigation_completed: int
    investigation_total: int
    token_usage: TokenUsage


class DailyReportWorkflow:
    """Run all report stages in order under one process-wide lock."""

    def __init__(
        self,
        storage: ReportStorage,
        question_processor: DailyQuestionProcessor,
        aggregation: QuestionAggregationService,
        investigation: InvestigationService,
        reports: ReportService,
        knowledge: KnowledgeService,
        *,
        timezone_name: str,
        token_usage: TokenUsageTracker | None = None,
    ) -> None:
        self._storage = storage
        self._question_processor = question_processor
        self._aggregation = aggregation
        self._investigation = investigation
        self._reports = reports
        self._knowledge = knowledge
        self._timezone_name = timezone_name
        self._token_usage = token_usage or TokenUsageTracker()
        self._usage_baseline = self._token_usage.snapshot()
        self._lock = asyncio.Lock()
        self._stage = "空闲"
        self._current_date = ""
        self._date_index = 0
        self._date_total = 0

    @property
    def running(self) -> bool:
        return self._lock.locked()

    @property
    def busy(self) -> bool:
        """Whether a manual or scheduled stage currently owns shared report resources."""

        return self.running or self._knowledge.syncing

    def progress(self) -> WorkflowProgress:
        screening_date, screening_completed, screening_total = self._question_processor.progress
        if screening_date != self._current_date:
            screening_completed = screening_total = 0
        aggregation_date, aggregation_completed, aggregation_total = self._aggregation.progress
        if aggregation_date != self._current_date:
            aggregation_completed = aggregation_total = 0
        investigation_date, investigation_completed, investigation_total = (
            self._investigation.progress
        )
        if investigation_date != self._current_date:
            investigation_completed = investigation_total = 0
        return WorkflowProgress(
            running=self.running or self._knowledge.syncing,
            stage=self._stage,
            report_date=self._current_date,
            date_index=self._date_index,
            date_total=self._date_total,
            screening_phase=self._question_processor.progress_phase,
            screening_completed=screening_completed,
            screening_total=screening_total,
            aggregation_completed=aggregation_completed,
            aggregation_total=aggregation_total,
            investigation_completed=investigation_completed,
            investigation_total=investigation_total,
            token_usage=self._token_usage.snapshot().since(self._usage_baseline),
        )

    async def sync_knowledge(self) -> None:
        async with self._lock:
            try:
                await self._sync_knowledge_locked()
            finally:
                self._stage = "空闲"

    async def run_date(
        self,
        report_date: date,
        *,
        deliver: bool = False,
        force: bool = False,
        sync_knowledge: bool = False,
    ) -> FullReportRunResult:
        async with self._lock:
            self._usage_baseline = self._token_usage.snapshot()
            self._date_index = 1
            self._date_total = 1
            try:
                if sync_knowledge:
                    await self._sync_knowledge_locked()
                return await self._run_date_locked(
                    report_date,
                    deliver=deliver,
                    force=force,
                )
            finally:
                self._stage = "空闲"

    async def run_all_history(
        self,
        *,
        before_date: date,
        deliver: bool = False,
        force: bool = False,
        sync_knowledge: bool = False,
    ) -> list[FullReportRunResult]:
        dates = await self.history_dates(before_date=before_date)
        async with self._lock:
            self._usage_baseline = self._token_usage.snapshot()
            self._date_total = len(dates)
            result: list[FullReportRunResult] = []
            try:
                if sync_knowledge:
                    await self._sync_knowledge_locked()
                for index, current in enumerate(dates, start=1):
                    self._date_index = index
                    result.append(
                        await self._run_date_locked(
                            current,
                            deliver=deliver,
                            force=force,
                        )
                    )
                return result
            finally:
                self._stage = "空闲"

    async def _sync_knowledge_locked(self) -> None:
        self._stage = "同步语雀知识库"
        await self._knowledge.sync_all()

    async def history_dates(self, *, before_date: date) -> list[date]:
        raw_dates = await asyncio.to_thread(
            self._storage.message_local_dates,
            self._timezone_name,
        )
        return [
            date.fromisoformat(item) for item in raw_dates if date.fromisoformat(item) < before_date
        ]

    async def is_report_complete(self, report_date: date) -> bool:
        return await self._completed_result(report_date) is not None

    async def _run_date_locked(
        self,
        report_date: date,
        *,
        deliver: bool,
        force: bool,
    ) -> FullReportRunResult:
        usage_before = self._token_usage.snapshot()
        self._current_date = report_date.isoformat()
        if not force:
            completed = await self._completed_result(report_date)
            if completed is not None:
                if deliver and completed.report is not None:
                    self._stage = "发送邮件"
                    delivery = await self._reports.deliver(completed.report)
                    return FullReportRunResult(
                        completed.screening,
                        completed.cluster_count,
                        completed.report,
                        delivery,
                        self._token_usage.snapshot().since(usage_before),
                    )
                return completed
        self._stage = "AI 筛选与自动复核"
        screening = await self._question_processor.process_date(report_date, force=force)
        if screening.status != "COMPLETED":
            return FullReportRunResult(
                screening,
                0,
                None,
                token_usage=self._token_usage.snapshot().since(usage_before),
            )
        self._stage = "问题与回答划分及脱敏摘要"
        clusters = await self._aggregation.aggregate_date(report_date)
        self._stage = "知识库调查"
        await self._investigation.investigate_date(report_date.isoformat())
        self._stage = "生成 HTML 日报"
        report = await self._reports.build(report_date.isoformat())
        self._stage = "发送邮件" if deliver else "完成"
        delivery = await self._reports.deliver(report) if deliver else None
        return FullReportRunResult(
            screening,
            len(clusters),
            report,
            delivery,
            self._token_usage.snapshot().since(usage_before),
        )

    async def _completed_result(self, report_date: date) -> FullReportRunResult | None:
        raw_date = report_date.isoformat()
        window = await asyncio.to_thread(self._storage.processing_window, raw_date)
        if window is None or window.status != "COMPLETED":
            return None
        report = await asyncio.to_thread(self._storage.latest_report, raw_date)
        if report is None:
            return None
        clusters = await asyncio.to_thread(self._storage.list_question_clusters, raw_date)
        screening = DailyRunResult(
            report_date=raw_date,
            status="SKIPPED_REPORT_COMPLETE",
            messages_scanned=window.messages_scanned,
            candidates_saved=window.candidates_saved,
            included_count=window.included_count,
            dropped_count=window.dropped_count,
            error_count=window.error_count,
        )
        return FullReportRunResult(screening, len(clusters), report)

    async def deliver_latest(self, report_date: str) -> DeliverySummary:
        window = await asyncio.to_thread(self._storage.processing_window, report_date)
        if window is None or window.status != "COMPLETED":
            status = window.status if window is not None else "NOT_RUN"
            raise RuntimeError(f"该日期筛选状态为 {status}，不能发送旧的或不完整的日报")
        report = await asyncio.to_thread(self._storage.latest_report, report_date)
        if report is None:
            raise RuntimeError("该日期尚未生成日报")
        return await self._reports.deliver(report)


class DailyScheduler:
    """Minute-resolution scheduler that safely stops on plugin unload."""

    def __init__(
        self,
        workflow: DailyReportWorkflow,
        storage: ReportStorage,
        *,
        mail_recipients: tuple[str, ...],
        timezone_name: str,
        report_time: str,
        enabled: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._workflow = workflow
        self._storage = storage
        self._recipient_hashes = tuple(recipient_hash(item) for item in mail_recipients)
        self._timezone = ZoneInfo(timezone_name)
        hour, minute = (int(item) for item in report_time.split(":", 1))
        self._time = time(hour, minute)
        self._enabled = enabled
        self._clock = clock or (lambda: datetime.now(self._timezone))
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._enabled and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._run(), name="nju-report-scheduler")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            now = self._clock()
            try:
                await self._tick(now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("NJU daily scheduler tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=60)
            except TimeoutError:
                continue

    async def _tick(self, now: datetime) -> None:
        """Process one deterministic scheduler tick using persistent state."""

        if not self._enabled:
            return
        if bool(getattr(self._workflow, "busy", False)):
            return
        now_utc = int(now.timestamp())
        state = await asyncio.to_thread(
            self._storage.oldest_unfinished_scheduled_report_run,
        )
        if state is None:
            scheduled = datetime.combine(now.date(), self._time, self._timezone)
            if now < scheduled:
                return
            scheduled_date = now.date().isoformat()
            report_date = (now.date() - timedelta(days=1)).isoformat()
            state = await asyncio.to_thread(
                self._storage.scheduled_report_run,
                scheduled_date,
            )
            if state is not None and state.status == "SENT":
                return
        else:
            scheduled_date = state.scheduled_date
            report_date = state.report_date

        already_delivered = await asyncio.to_thread(
            self._storage.report_date_delivered_to,
            report_date,
            self._recipient_hashes,
        )
        if already_delivered:
            await asyncio.to_thread(
                self._storage.bootstrap_scheduled_report_sent,
                scheduled_date,
                report_date,
                now_utc=now_utc,
            )
            return

        if state is not None:
            if (
                state.status == "RETRY_PENDING"
                and state.next_retry_at_utc is not None
                and state.next_retry_at_utc > now_utc
            ):
                return
            stale_before_utc = now_utc - _RUNNING_STALE_AFTER_SECONDS
            if state.status == "RUNNING" and state.updated_at_utc > stale_before_utc:
                return
        else:
            stale_before_utc = now_utc - _RUNNING_STALE_AFTER_SECONDS

        claim_task = asyncio.create_task(
            asyncio.to_thread(
                self._storage.begin_scheduled_report_run,
                scheduled_date,
                report_date,
                now_utc=now_utc,
                stale_before_utc=stale_before_utc,
            )
        )
        try:
            claim_token = await asyncio.shield(claim_task)
        except asyncio.CancelledError:
            claim_token = await claim_task
            if claim_token is not None:
                cancelled_utc = int(self._clock().timestamp())
                await asyncio.shield(
                    asyncio.to_thread(
                        self._storage.fail_scheduled_report_run,
                        scheduled_date,
                        error_summary="CancelledError: scheduler stopped while claiming",
                        next_retry_at_utc=cancelled_utc,
                        now_utc=cancelled_utc,
                        claim_token=claim_token,
                    )
                )
            raise
        if claim_token is None:
            return

        try:
            result = await self._workflow.run_date(
                date.fromisoformat(report_date),
                deliver=True,
                sync_knowledge=True,
            )
            if result.report is None:
                raise RuntimeError("daily report was not generated")
            if result.delivery is None:
                raise RuntimeError("daily report delivery was not attempted")
            if result.delivery.failed:
                raise RuntimeError("one or more report recipients failed")
        except asyncio.CancelledError:
            cancelled_at = self._clock()
            cancelled_utc = int(cancelled_at.timestamp())
            await asyncio.shield(
                asyncio.to_thread(
                    self._storage.fail_scheduled_report_run,
                    scheduled_date,
                    error_summary="CancelledError: scheduler stopped",
                    next_retry_at_utc=cancelled_utc,
                    now_utc=cancelled_utc,
                    claim_token=claim_token,
                )
            )
            raise
        except Exception as exc:
            failed_at = self._clock()
            failed_utc = int(failed_at.timestamp())
            retry_at = int((failed_at + timedelta(minutes=15)).timestamp())
            await asyncio.to_thread(
                self._storage.fail_scheduled_report_run,
                scheduled_date,
                error_summary=f"{type(exc).__name__}: {exc}",
                next_retry_at_utc=retry_at,
                now_utc=failed_utc,
                claim_token=claim_token,
            )
            logger.exception("NJU scheduled daily report failed")
            return

        completed_utc = int(self._clock().timestamp())
        await asyncio.to_thread(
            self._storage.complete_scheduled_report_run,
            scheduled_date,
            now_utc=completed_utc,
            claim_token=claim_token,
        )


def format_scheduled_report_status(state: ScheduledReportRun) -> str:
    status_labels = {
        "RUNNING": "运行中",
        "RETRY_PENDING": "等待自动重试",
        "SENT": "已发送",
    }
    lines = [
        f"零点自动日报：{status_labels.get(state.status, state.status)}",
        f"调度日期：{state.scheduled_date}",
        f"处理日期：{state.report_date}",
        f"调度尝试：{state.attempts}",
    ]
    if state.error_summary:
        lines.append(f"上次错误：{state.error_summary}")
    return "\n".join(lines)
