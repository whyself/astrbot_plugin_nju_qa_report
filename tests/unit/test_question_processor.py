from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
from pathlib import Path

from nju_report.models import (
    ScopeAssessment,
    ScopeDecision,
    ScopeResolution,
    StoredMessage,
)
from nju_report.question_export import QuestionCsvExporter
from nju_report.question_processor import DailyQuestionProcessor
from nju_report.storage import ReportStorage
from nju_report.time_windows import natural_day_window


class FakeScopeService:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def resolve(self, message: str, context: str = "") -> ScopeResolution:
        del context
        self.seen.append(message)
        if "纳入" in message:
            assessment = ScopeAssessment(
                decision=ScopeDecision.INCLUDE,
                reason="属于南京大学校园公共知识",
                confidence=0.95,
                canonical_question="南京大学校园卡丢失后如何补办？",
                category="校园生活/校园卡",
            )
            return ScopeResolution(assessment, review_rounds=0)
        if "错误" in message:
            assessment = ScopeAssessment(
                decision=ScopeDecision.AUTO_REVIEW_ERROR,
                reason="AI 范围审核发生技术错误",
                confidence=0.0,
            )
            return ScopeResolution(
                assessment,
                review_rounds=1,
                retryable=True,
                error_summary="FakeProviderError",
            )
        assessment = ScopeAssessment(
            decision=ScopeDecision.DROP,
            reason="无关闲聊",
            confidence=0.9,
        )
        return ScopeResolution(assessment, review_rounds=0)


def _stored(message_id: str, text: str, timestamp: int) -> StoredMessage:
    return StoredMessage(
        platform_id="aiocqhttp:default",
        bot_self_id="999",
        external_message_id=message_id,
        message_fingerprint=f"fingerprint-{message_id}",
        session_id="group-session",
        group_id="826811581",
        group_alias="南京大学迎新群",
        sender_id="10001",
        sender_name="测试用户",
        sent_at_utc=timestamp,
        text=text,
        outline=text,
        reply_to_message_id="",
        analyzable=True,
    )


def test_daily_processor_retains_include_drop_and_error_and_skips_rerun(
    tmp_path: Path,
) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(_stored("m-1", "纳入：校园卡怎么补办？", window.start_timestamp + 1))
    storage.insert_message(_stored("m-2", "排除：今晚吃什么？", window.start_timestamp + 2))
    storage.insert_message(_stored("m-3", "错误：模型超时", window.start_timestamp + 3))
    processor = DailyQuestionProcessor(
        storage,
        FakeScopeService(),  # type: ignore[arg-type]
        timezone_name="Asia/Shanghai",
        concurrency=2,
    )

    first = asyncio.run(processor.process_date(report_date))
    assert first.status == "RETRY_PENDING"
    assert first.candidates_saved == 3
    assert first.included_count == 1
    assert first.dropped_count == 1
    assert first.error_count == 1

    candidates, total = storage.list_question_candidates(limit=None)
    assert total == 3
    assert [item.question_code for item in candidates] == [
        "20260712-Q001",
        "20260712-Q002",
        "20260712-Q003",
    ]
    assert {item.final_decision for item in candidates} == {
        "INCLUDE",
        "DROP",
        "AUTO_REVIEW_ERROR",
    }
    assert all(item.original_question for item in candidates)

    second = asyncio.run(processor.process_date(report_date))
    assert second.status == "RETRY_PENDING"
    assert storage.question_candidate_count() == 3
    storage.close()


def test_all_history_reports_completed_dates_as_skipped(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(_stored("m-1", "排除：闲聊", window.start_timestamp + 1))
    processor = DailyQuestionProcessor(
        storage,
        FakeScopeService(),  # type: ignore[arg-type]
        timezone_name="Asia/Shanghai",
    )

    asyncio.run(processor.process_date(report_date))
    results = asyncio.run(processor.process_all_history(before_date=date(2026, 7, 14)))
    assert len(results) == 1
    assert results[0].status == "SKIPPED_COMPLETED"
    assert results[0].candidates_saved == 1
    forced = asyncio.run(processor.process_date(report_date, force=True))
    assert forced.status == "COMPLETED"
    assert forced.skipped is False
    storage.close()


def test_cumulative_csv_contains_every_decision(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(_stored("m-1", "纳入：一卡通丢了怎么办？", window.start_timestamp + 1))
    storage.insert_message(_stored("m-2", "排除：闲聊", window.start_timestamp + 2))
    processor = DailyQuestionProcessor(
        storage,
        FakeScopeService(),  # type: ignore[arg-type]
        timezone_name="Asia/Shanghai",
    )
    asyncio.run(processor.process_date(report_date))

    exporter = QuestionCsvExporter(
        storage,
        tmp_path / "exports",
        timezone_name="Asia/Shanghai",
    )
    output, count = exporter.export_all()
    content = output.read_text(encoding="utf-8-sig")
    assert count == 2
    assert "20260712-Q001" in content
    assert "20260712-Q002" in content
    assert "已纳入" in content
    assert "已排除" in content
    assert not output.with_suffix(".csv.tmp").exists()
    storage.close()


def test_every_nonempty_message_reaches_ai_without_keyword_prefilter(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(_stored("m-1", "哈哈哈", window.start_timestamp + 1))
    storage.insert_message(
        replace(
            _stored("m-2", "", window.start_timestamp + 2),
            outline="[图片]",
            analyzable=False,
        )
    )
    scope = FakeScopeService()
    processor = DailyQuestionProcessor(
        storage,
        scope,  # type: ignore[arg-type]
        timezone_name="Asia/Shanghai",
        concurrency=1,
    )

    result = asyncio.run(processor.process_date(report_date))

    assert result.candidates_saved == 2
    assert scope.seen == ["哈哈哈", "[图片]"]
    storage.close()
