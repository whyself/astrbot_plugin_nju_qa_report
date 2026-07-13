from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from nju_report.models import ReportArtifact, StoredMessage
from nju_report.question_processor import DailyRunResult
from nju_report.storage import ReportStorage
from nju_report.time_windows import natural_day_window
from nju_report.token_usage import TokenUsageTracker
from nju_report.workflow import DailyReportWorkflow


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
