"""End-to-end daily report workflow and restart-safe scheduler."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .aggregation import QuestionAggregationService
from .investigation import InvestigationService
from .knowledge import KnowledgeService
from .models import ReportArtifact
from .question_processor import DailyQuestionProcessor, DailyRunResult
from .reporting import DeliverySummary, ReportService
from .storage import ReportStorage
from .token_usage import TokenUsage, TokenUsageTracker

logger = logging.getLogger(__name__)


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
            screening_completed=screening_completed,
            screening_total=screening_total,
            aggregation_completed=aggregation_completed,
            aggregation_total=aggregation_total,
            investigation_completed=investigation_completed,
            investigation_total=investigation_total,
            token_usage=self._token_usage.snapshot().since(self._usage_baseline),
        )

    async def sync_knowledge(self) -> None:
        self._stage = "同步语雀知识库"
        try:
            await self._knowledge.sync_all()
        finally:
            if not self.running:
                self._stage = "空闲"

    async def run_date(
        self,
        report_date: date,
        *,
        deliver: bool = False,
        force: bool = False,
    ) -> FullReportRunResult:
        async with self._lock:
            self._usage_baseline = self._token_usage.snapshot()
            self._date_index = 1
            self._date_total = 1
            try:
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
    ) -> list[FullReportRunResult]:
        dates = await self.history_dates(before_date=before_date)
        async with self._lock:
            self._usage_baseline = self._token_usage.snapshot()
            self._date_total = len(dates)
            result: list[FullReportRunResult] = []
            try:
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
        if screening.status == "FAILED":
            return FullReportRunResult(
                screening,
                0,
                None,
                token_usage=self._token_usage.snapshot().since(usage_before),
            )
        self._stage = "问题聚合与群友回答上下文判断"
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
        report = await asyncio.to_thread(self._storage.latest_report, raw_date)
        if window is None or window.status != "COMPLETED" or report is None:
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
        report = await asyncio.to_thread(self._storage.latest_report, report_date)
        if report is None:
            raise RuntimeError("该日期尚未生成日报")
        return await self._reports.deliver(report)


class DailyScheduler:
    """Minute-resolution scheduler that safely stops on plugin unload."""

    def __init__(
        self,
        workflow: DailyReportWorkflow,
        *,
        timezone_name: str,
        report_time: str,
        enabled: bool,
    ) -> None:
        self._workflow = workflow
        self._timezone = ZoneInfo(timezone_name)
        hour, minute = (int(item) for item in report_time.split(":", 1))
        self._time = time(hour, minute)
        self._enabled = enabled
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
        last_run: date | None = None
        retry_not_before: datetime | None = None
        while not self._stop.is_set():
            now = datetime.now(self._timezone)
            scheduled = datetime.combine(now.date(), self._time, self._timezone)
            retry_due = retry_not_before is None or now >= retry_not_before
            if now >= scheduled and last_run != now.date() and retry_due:
                report_date = now.date() - timedelta(days=1)
                try:
                    await self._workflow.sync_knowledge()
                    result = await self._workflow.run_date(report_date, deliver=True)
                    if result.report is None:
                        raise RuntimeError("daily report was not generated")
                    if result.delivery is not None and result.delivery.failed:
                        raise RuntimeError("one or more report recipients failed")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The workflow persists stage state; the next manual or scheduled run can retry.
                    logger.exception("NJU scheduled daily report failed")
                    retry_not_before = now + timedelta(minutes=15)
                else:
                    last_run = now.date()
                    retry_not_before = None
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=60)
            except TimeoutError:
                continue
