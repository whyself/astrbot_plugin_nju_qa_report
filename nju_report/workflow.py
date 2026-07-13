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

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FullReportRunResult:
    screening: DailyRunResult
    cluster_count: int
    report: ReportArtifact | None
    delivery: DeliverySummary | None = None


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
    ) -> None:
        self._storage = storage
        self._question_processor = question_processor
        self._aggregation = aggregation
        self._investigation = investigation
        self._reports = reports
        self._knowledge = knowledge
        self._timezone_name = timezone_name
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._lock.locked()

    async def sync_knowledge(self) -> None:
        await self._knowledge.sync_all()

    async def run_date(
        self,
        report_date: date,
        *,
        deliver: bool = False,
    ) -> FullReportRunResult:
        async with self._lock:
            screening = await self._question_processor.process_date(report_date)
            if screening.status == "FAILED":
                return FullReportRunResult(screening, 0, None)
            clusters = await self._aggregation.aggregate_date(report_date)
            await self._investigation.investigate_date(report_date.isoformat())
            report = await self._reports.build(report_date.isoformat())
            delivery = await self._reports.deliver(report) if deliver else None
            return FullReportRunResult(screening, len(clusters), report, delivery)

    async def run_all_history(
        self,
        *,
        before_date: date,
        deliver: bool = False,
    ) -> list[FullReportRunResult]:
        dates = await asyncio.to_thread(
            self._storage.message_local_dates,
            self._timezone_name,
        )
        result: list[FullReportRunResult] = []
        for raw in dates:
            current = date.fromisoformat(raw)
            if current < before_date:
                result.append(await self.run_date(current, deliver=deliver))
        return result

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
