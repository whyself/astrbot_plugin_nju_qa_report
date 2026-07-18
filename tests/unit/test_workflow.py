from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nju_report.models import ReportArtifact, StoredMessage
from nju_report.question_processor import DailyRunResult
from nju_report.reporting import DeliverySummary, recipient_hash
from nju_report.storage import ReportStorage
from nju_report.time_windows import natural_day_window
from nju_report.token_usage import TokenUsageTracker
from nju_report.workflow import (
    DailyReportWorkflow,
    DailyScheduler,
    format_scheduled_report_status,
)


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[date, bool]] = []
        self.progress = ("", 0, 0)

    async def process_date(self, report_date: date, *, force: bool = False):
        self.calls.append((report_date, force))
        return DailyRunResult(report_date.isoformat(), "COMPLETED")


class FakeAggregation:
    progress = ("", 0, 0)

    async def aggregate_date(self, report_date: date):
        return []


class FakeInvestigation:
    progress = ("", 0, 0)

    async def investigate_date(self, report_date: str):
        return []


class FakeReports:
    def __init__(self, report: ReportArtifact) -> None:
        self.report = report

    async def build(self, report_date: str):
        return self.report

    async def deliver(self, report: ReportArtifact):
        raise AssertionError("本测试不应发送邮件")


class FakeKnowledge:
    syncing = False

    async def sync_all(self):
        return None


class FakeScheduledWorkflow:
    def __init__(
        self,
        *,
        failures: int = 0,
        on_run: Callable[[], None] | None = None,
    ) -> None:
        self.failures = failures
        self.on_run = on_run
        self.sync_calls = 0
        self.run_calls: list[date] = []

    async def sync_knowledge(self) -> None:
        self.sync_calls += 1

    async def run_date(
        self,
        report_date: date,
        *,
        deliver: bool = False,
        sync_knowledge: bool = False,
    ):
        assert deliver is True
        assert sync_knowledge is True
        self.sync_calls += 1
        self.run_calls.append(report_date)
        if self.on_run is not None:
            self.on_run()
        if self.failures:
            self.failures -= 1
            raise RuntimeError("smtp failed")
        return SimpleNamespace(
            report=object(),
            delivery=DeliverySummary(sent=1),
        )


def _scheduler(
    workflow: FakeScheduledWorkflow,
    storage: ReportStorage,
    *,
    recipients: tuple[str, ...] = ("reader@example.com",),
    report_time: str = "00:00",
    clock: Callable[[], datetime] | None = None,
) -> DailyScheduler:
    return DailyScheduler(  # type: ignore[arg-type]
        workflow,
        storage,
        mail_recipients=recipients,
        timezone_name="Asia/Shanghai",
        report_time=report_time,
        enabled=True,
        clock=clock,
    )


def test_scheduler_catches_up_once_and_persists_success_across_reload(tmp_path: Path) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        now = datetime(2026, 7, 14, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        first_workflow = FakeScheduledWorkflow()

        await _scheduler(first_workflow, storage)._tick(now)

        assert first_workflow.run_calls == [date(2026, 7, 13)]
        persisted = storage.scheduled_report_run("2026-07-14")
        assert persisted is not None
        assert persisted.status == "SENT"

        reloaded_workflow = FakeScheduledWorkflow()
        await _scheduler(reloaded_workflow, storage)._tick(now + timedelta(hours=1))
        assert reloaded_workflow.run_calls == []
        storage.close()

    asyncio.run(run())


def test_scheduler_waits_while_manual_report_or_knowledge_sync_is_busy(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        now = datetime(2026, 7, 19, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        workflow = FakeScheduledWorkflow()
        workflow.busy = True

        await _scheduler(workflow, storage)._tick(now)

        assert workflow.sync_calls == 0
        assert workflow.run_calls == []
        assert storage.scheduled_report_run("2026-07-19") is None
        storage.close()

    asyncio.run(run())


def test_scheduled_running_state_has_operator_status_text(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    claim = storage.begin_scheduled_report_run(
        "2026-07-19",
        "2026-07-18",
        now_utc=100,
        stale_before_utc=0,
    )
    assert claim is not None
    state = storage.oldest_unfinished_scheduled_report_run()
    assert state is not None

    text = format_scheduled_report_status(state)

    assert "零点自动日报：运行中" in text
    assert "调度日期：2026-07-19" in text
    assert "处理日期：2026-07-18" in text
    assert "调度尝试：1" in text
    storage.close()


def test_scheduler_persists_retry_deadline_and_retries_after_15_minutes(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        now = datetime(2026, 7, 14, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        failing = FakeScheduledWorkflow(failures=1)

        await _scheduler(failing, storage, clock=lambda: now)._tick(now)

        pending = storage.scheduled_report_run("2026-07-14")
        assert pending is not None
        assert pending.status == "RETRY_PENDING"
        assert pending.next_retry_at_utc == int((now + timedelta(minutes=15)).timestamp())

        reloaded = FakeScheduledWorkflow()
        scheduler = _scheduler(reloaded, storage)
        await scheduler._tick(now + timedelta(minutes=14, seconds=59))
        assert reloaded.run_calls == []

        await scheduler._tick(now + timedelta(minutes=15))
        assert reloaded.run_calls == [date(2026, 7, 13)]
        assert storage.scheduled_report_run("2026-07-14").status == "SENT"
        storage.close()

    asyncio.run(run())


def test_scheduler_bootstraps_success_from_existing_recipient_deliveries(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        first = storage.save_report(
            report_date="2026-07-13",
            subject="first",
            html_path=str(tmp_path / "first.html"),
            summary_json="{}",
        )
        second = storage.save_report(
            report_date="2026-07-13",
            subject="second",
            html_path=str(tmp_path / "second.html"),
            summary_json='{"version":2}',
        )
        recipients = ("first@example.com", "second@example.com")
        for report, recipient in zip((first, second), recipients, strict=True):
            hashed = recipient_hash(recipient)
            assert storage.begin_mail_delivery(report.report_id, hashed)
            storage.complete_mail_delivery(report.report_id, hashed)

        workflow = FakeScheduledWorkflow()
        scheduler = _scheduler(workflow, storage, recipients=recipients)
        now = datetime(2026, 7, 14, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

        await scheduler._tick(now)

        assert workflow.run_calls == []
        persisted = storage.scheduled_report_run("2026-07-14")
        assert persisted is not None
        assert persisted.status == "SENT"
        assert persisted.attempts == 0
        storage.close()

    asyncio.run(run())


def test_scheduler_reclaims_a_running_attempt_left_by_reload(tmp_path: Path) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        claim = storage.begin_scheduled_report_run(
            "2026-07-14",
            "2026-07-13",
            now_utc=1,
            stale_before_utc=0,
        )
        assert claim is not None
        workflow = FakeScheduledWorkflow()

        await _scheduler(workflow, storage)._tick(
            datetime(2026, 7, 14, 1, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        )

        assert workflow.run_calls == [date(2026, 7, 13)]
        persisted = storage.scheduled_report_run("2026-07-14")
        assert persisted is not None
        assert persisted.status == "SENT"
        assert persisted.attempts == 2
        storage.close()

    asyncio.run(run())


def test_scheduler_immediately_resumes_run_recovered_during_reload(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        now = datetime(2026, 7, 19, 0, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        claim = storage.begin_scheduled_report_run(
            "2026-07-19",
            "2026-07-18",
            now_utc=int((now - timedelta(minutes=30)).timestamp()),
            stale_before_utc=0,
        )
        assert claim is not None
        assert storage.recover_running_scheduled_report_runs(
            now_utc=int(now.timestamp())
        ) == 1
        workflow = FakeScheduledWorkflow()

        await _scheduler(workflow, storage)._tick(now)

        assert workflow.sync_calls == 1
        assert workflow.run_calls == [date(2026, 7, 18)]
        persisted = storage.scheduled_report_run("2026-07-19")
        assert persisted is not None
        assert persisted.status == "SENT"
        assert persisted.attempts == 2
        storage.close()

    asyncio.run(run())


def test_scheduler_sets_retry_deadline_from_failure_time(tmp_path: Path) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        started = datetime(2026, 7, 14, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        failed_at = started + timedelta(minutes=20)
        current = [started]
        workflow = FakeScheduledWorkflow(
            failures=1,
            on_run=lambda: current.__setitem__(0, failed_at),
        )

        await _scheduler(workflow, storage, clock=lambda: current[0])._tick(started)

        pending = storage.scheduled_report_run("2026-07-14")
        assert pending is not None
        assert pending.updated_at_utc == int(failed_at.timestamp())
        assert pending.next_retry_at_utc == int(
            (failed_at + timedelta(minutes=15)).timestamp()
        )
        storage.close()

    asyncio.run(run())


def test_scheduler_resumes_due_retry_after_midnight_before_new_day(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        storage = ReportStorage(tmp_path / "report.sqlite3")
        storage.initialize()
        first_tick = datetime(2026, 7, 14, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))
        current = [first_tick]
        failing = FakeScheduledWorkflow(failures=1)
        await _scheduler(failing, storage, clock=lambda: current[0])._tick(first_tick)
        pending = storage.scheduled_report_run("2026-07-14")
        assert pending is not None

        current[0] = first_tick + timedelta(minutes=15)
        resumed = FakeScheduledWorkflow()
        await _scheduler(resumed, storage, clock=lambda: current[0])._tick(current[0])

        assert resumed.run_calls == [date(2026, 7, 13)]
        assert storage.scheduled_report_run("2026-07-14").status == "SENT"
        assert storage.scheduled_report_run("2026-07-15") is None
        storage.close()

    asyncio.run(run())


def test_normal_history_run_skips_complete_report_but_force_recomputes(tmp_path: Path) -> None:
    async def run() -> None:
        report_date = date(2026, 7, 12)
        storage = _complete_report_storage(tmp_path, report_date)
        existing = storage.latest_report(report_date.isoformat())
        assert existing is not None
        processor = FakeProcessor()
        workflow = DailyReportWorkflow(  # type: ignore[arg-type]
            storage,
            processor,
            FakeAggregation(),
            FakeInvestigation(),
            FakeReports(existing),
            FakeKnowledge(),
            timezone_name="Asia/Shanghai",
        )

        normal = await workflow.run_all_history(before_date=date(2026, 7, 13))
        assert normal[0].screening.status == "SKIPPED_REPORT_COMPLETE"
        assert processor.calls == []

        forced = await workflow.run_date(report_date, force=True)
        assert forced.screening.status == "COMPLETED"
        assert processor.calls == [(report_date, True)]
        storage.close()

    asyncio.run(run())


def test_incomplete_retry_does_not_read_an_older_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        report_date = date(2026, 7, 12)
        storage = _complete_report_storage(tmp_path, report_date)
        existing = storage.latest_report(report_date.isoformat())
        assert existing is not None
        window = natural_day_window(report_date, "Asia/Shanghai")
        assert storage.begin_processing_window(window, run_id="retry", force=True)
        storage.complete_processing_window(
            report_date.isoformat(),
            run_id="retry",
            messages_scanned=1,
            candidates_saved=1,
            included_count=0,
            dropped_count=0,
            error_count=1,
        )

        def reject_old_report_read(_report_date: str):
            raise AssertionError("未完成的重跑不得读取旧报告")

        monkeypatch.setattr(storage, "latest_report", reject_old_report_read)
        workflow = DailyReportWorkflow(  # type: ignore[arg-type]
            storage,
            FakeProcessor(),
            FakeAggregation(),
            FakeInvestigation(),
            FakeReports(existing),
            FakeKnowledge(),
            timezone_name="Asia/Shanghai",
        )

        assert await workflow._completed_result(report_date) is None
        storage.close()

    asyncio.run(run())


def test_force_all_history_recomputes_every_available_date(tmp_path: Path) -> None:
    async def run() -> None:
        first_date = date(2026, 7, 11)
        second_date = date(2026, 7, 12)
        storage = _complete_report_storage(tmp_path, first_date)
        window = natural_day_window(second_date, "Asia/Shanghai")
        storage.insert_message(
            StoredMessage(
                platform_id="qq",
                bot_self_id="bot",
                external_message_id="m2",
                message_fingerprint="fp2",
                session_id="group:1",
                group_id="1",
                group_alias="测试群",
                sender_id="u2",
                sender_name="",
                sent_at_utc=window.start_timestamp + 1,
                text="宿舍什么时候开放？",
                outline="",
                reply_to_message_id="",
                analyzable=True,
            )
        )
        existing = storage.latest_report(first_date.isoformat())
        assert existing is not None
        processor = FakeProcessor()
        workflow = DailyReportWorkflow(  # type: ignore[arg-type]
            storage,
            processor,
            FakeAggregation(),
            FakeInvestigation(),
            FakeReports(existing),
            FakeKnowledge(),
            timezone_name="Asia/Shanghai",
        )

        results = await workflow.run_all_history(
            before_date=date(2026, 7, 13),
            force=True,
        )

        assert len(results) == 2
        assert processor.calls == [(first_date, True), (second_date, True)]
        storage.close()

    asyncio.run(run())


def test_workflow_returns_only_this_run_provider_token_usage(tmp_path: Path) -> None:
    class UsageProcessor(FakeProcessor):
        def __init__(self, tracker: TokenUsageTracker) -> None:
            super().__init__()
            self.tracker = tracker

        async def process_date(self, report_date: date, *, force: bool = False):
            self.tracker.record(
                SimpleNamespace(
                    raw_completion=SimpleNamespace(
                        usage={
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        }
                    )
                )
            )
            return await super().process_date(report_date, force=force)

    async def run() -> None:
        report_date = date(2026, 7, 12)
        storage = _complete_report_storage(tmp_path, report_date)
        existing = storage.latest_report(report_date.isoformat())
        assert existing is not None
        tracker = TokenUsageTracker()
        tracker.record({"usage": {"prompt_tokens": 999, "completion_tokens": 1}})
        workflow = DailyReportWorkflow(  # type: ignore[arg-type]
            storage,
            UsageProcessor(tracker),
            FakeAggregation(),
            FakeInvestigation(),
            FakeReports(existing),
            FakeKnowledge(),
            timezone_name="Asia/Shanghai",
            token_usage=tracker,
        )

        result = await workflow.run_date(report_date, force=True)

        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 20
        assert result.token_usage.total_tokens == 120
        assert result.token_usage.calls == result.token_usage.reported_calls == 1
        storage.close()

    asyncio.run(run())


def _complete_report_storage(tmp_path: Path, report_date: date) -> ReportStorage:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(
        StoredMessage(
            platform_id="qq",
            bot_self_id="bot",
            external_message_id="m1",
            message_fingerprint="fp1",
            session_id="group:1",
            group_id="1",
            group_alias="测试群",
            sender_id="u1",
            sender_name="",
            sent_at_utc=window.start_timestamp + 1,
            text="校园卡丢了怎么办？",
            outline="",
            reply_to_message_id="",
            analyzable=True,
        )
    )
    assert storage.begin_processing_window(window, run_id="first") is True
    storage.complete_processing_window(
        report_date.isoformat(),
        run_id="first",
        messages_scanned=1,
        candidates_saved=1,
        included_count=1,
        dropped_count=0,
        error_count=0,
    )
    storage.save_report(
        report_date=report_date.isoformat(),
        subject="测试日报",
        html_path=str(tmp_path / "report.html"),
        summary_json="{}",
    )
    return storage
