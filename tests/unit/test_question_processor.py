from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

from nju_report.models import (
    Clarity,
    KnowledgeValue,
    ScopeAssessment,
    ScopeDecision,
    ScopeResolution,
    StoredMessage,
)
from nju_report.question_export import QuestionCsvExporter
from nju_report.question_processor import DailyQuestionProcessor, _screening_chunks
from nju_report.scope_classifier import QuestionGateCandidate, ScopeBatchMessage
from nju_report.storage import ReportStorage
from nju_report.time_windows import natural_day_window


class FakeScopeService:
    def __init__(self) -> None:
        self.seen: list[str] = []
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
        self.batch_message_sizes: list[int] = []

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

    async def resolve_batch(
        self,
        messages: list[ScopeBatchMessage] | tuple[ScopeBatchMessage, ...],
        target_ids: list[str] | tuple[str, ...],
    ) -> dict[str, ScopeResolution]:
        self.batch_calls += 1
        self.batch_sizes.append(len(target_ids))
        self.batch_message_sizes.append(len(messages))
        by_id = {item.message_id: item for item in messages}
        return {
            message_id: await self.resolve(by_id[message_id].content)
            for message_id in target_ids
        }


class GateScopeService:
    async def resolve_batch(
        self,
        messages: list[ScopeBatchMessage] | tuple[ScopeBatchMessage, ...],
        target_ids: list[str] | tuple[str, ...],
    ) -> dict[str, ScopeResolution]:
        by_id = {item.message_id: item for item in messages}
        return {
            message_id: ScopeResolution(
                ScopeAssessment(
                    decision=ScopeDecision.INCLUDE,
                    reason="初筛纳入",
                    confidence=0.9,
                    canonical_question=by_id[message_id].content,
                    category="测试分类",
                    clarity=Clarity.CLEAR,
                    knowledge_value=KnowledgeValue.HIGH,
                ),
                review_rounds=0,
            )
            for message_id in target_ids
        }


class FakeFinalQuestionGate:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.seen: list[QuestionGateCandidate] = []

    async def final_review_batch(
        self,
        candidates: list[QuestionGateCandidate],
    ) -> dict[str, ScopeAssessment]:
        self.seen.extend(candidates)
        if self.fail:
            raise TimeoutError("final gate timeout")
        result: dict[str, ScopeAssessment] = {}
        for item in candidates:
            if "好吃" in item.canonical_question:
                result[item.candidate_id] = ScopeAssessment(
                    decision=ScopeDecision.DROP,
                    reason="主观口味问题不进入知识库",
                    confidence=0.98,
                    clarity=Clarity.CLEAR,
                    knowledge_value=KnowledgeValue.LOW,
                )
            else:
                result[item.candidate_id] = ScopeAssessment(
                    decision=ScopeDecision.INCLUDE,
                    reason="合并为同一项宿舍分配规则",
                    confidence=0.96,
                    canonical_question="南京大学2026级本科生大二宿舍如何分配？",
                    category="住宿宿舍",
                    clarity=Clarity.CLEAR,
                    knowledge_value=KnowledgeValue.HIGH,
                    time_sensitive=True,
                )
        return result


class BatchAwareFinalQuestionGate:
    def __init__(self, *, failing_batch: int) -> None:
        self.failing_batch = failing_batch
        self.batch_sizes: list[int] = []

    async def final_review_batch(
        self,
        candidates: list[QuestionGateCandidate],
    ) -> dict[str, ScopeAssessment]:
        self.batch_sizes.append(len(candidates))
        if len(self.batch_sizes) == self.failing_batch:
            raise TimeoutError("selected final gate batch timeout")
        return {
            item.candidate_id: ScopeAssessment(
                decision=ScopeDecision.INCLUDE,
                reason="问题清晰且适合进入知识库",
                confidence=0.95,
                canonical_question=item.canonical_question,
                category=item.category,
                clarity=Clarity.CLEAR,
                knowledge_value=KnowledgeValue.HIGH,
                time_sensitive=item.time_sensitive,
            )
            for item in candidates
        }


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


def test_screening_chunks_exclude_configured_bot_before_ai() -> None:
    user = _stored("user", "校园卡怎么补办？", 100)
    bot = replace(_stored("bot", "机器人自动回答", 101), sender_id="bot-qq")

    chunks = _screening_chunks(
        [user, bot],
        context_radius=30,
        ignored_sender_ids=frozenset({"bot-qq"}),
    )

    assert [item[2].external_message_id for item in chunks[0].targets] == ["user"]
    assert [item.message_id for item in chunks[0].messages] == ["m0"]


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
    scope = FakeScopeService()
    processor = DailyQuestionProcessor(
        storage,
        scope,  # type: ignore[arg-type]
        timezone_name="Asia/Shanghai",
        concurrency=2,
    )

    first = asyncio.run(processor.process_date(report_date))
    assert first.status == "RETRY_PENDING"
    assert first.candidates_saved == 3
    assert first.included_count == 1
    assert first.dropped_count == 1
    assert first.error_count == 1
    assert scope.batch_calls == 1

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


def test_final_question_gate_drops_subjective_and_merges_duplicate_questions(
    tmp_path: Path,
) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(
        _stored("m-1", "南京大学鼓楼校区哪些窗口比较好吃？", window.start_timestamp + 1)
    )
    storage.insert_message(
        _stored("m-2", "2026级大二宿舍怎么分配？", window.start_timestamp + 2)
    )
    storage.insert_message(
        _stored("m-3", "26级大二会继承23级宿舍吗？", window.start_timestamp + 3)
    )
    gate = FakeFinalQuestionGate()
    processor = DailyQuestionProcessor(
        storage,
        GateScopeService(),  # type: ignore[arg-type]
        final_question_reviewer=gate,
        timezone_name="Asia/Shanghai",
    )

    result = asyncio.run(processor.process_date(report_date))

    assert result.status == "COMPLETED"
    assert result.included_count == 2
    assert result.dropped_count == 1
    assert len(gate.seen) == 3
    assert {item.report_date for item in gate.seen} == {"2026-07-12"}
    candidates, total = storage.list_question_candidates(limit=None)
    assert total == 3
    assert candidates[0].final_decision == "DROP"
    assert candidates[1].canonical_question == candidates[2].canonical_question
    assert candidates[1].canonical_question == "南京大学2026级本科生大二宿舍如何分配？"
    storage.close()


def test_final_question_gate_failure_marks_only_selected_questions_retryable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "report.sqlite3"
    storage = ReportStorage(database_path)
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    storage.insert_message(_stored("m-1", "校园卡如何补办？", window.start_timestamp + 1))
    processor = DailyQuestionProcessor(
        storage,
        GateScopeService(),  # type: ignore[arg-type]
        final_question_reviewer=FakeFinalQuestionGate(fail=True),
        timezone_name="Asia/Shanghai",
    )

    result = asyncio.run(processor.process_date(report_date))

    assert result.status == "RETRY_PENDING"
    assert result.error_count == 1
    candidates, total = storage.list_question_candidates(limit=None)
    assert total == 1
    assert candidates[0].final_decision == "AUTO_REVIEW_ERROR"
    assert "最终问题 AI 闸门" in candidates[0].reason
    storage.close()
    with sqlite3.connect(database_path) as connection:
        error_summary = connection.execute(
            "SELECT error_summary FROM scope_review_runs WHERE error_summary != ''"
        ).fetchone()[0]
    assert "FinalGateBatchError" in error_summary
    assert "report_date=2026-07-12" in error_summary
    assert "batch=1/1" in error_summary
    assert "batch_candidates=1" in error_summary
    assert "total_candidates=1" in error_summary
    assert "cause=TimeoutError: final gate timeout" in error_summary


def test_final_question_gate_batches_progress_and_isolates_one_failed_batch(
    tmp_path: Path,
) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    for index in range(45):
        storage.insert_message(
            _stored(
                f"m-{index}",
                f"第 {index} 个不同的校园问题是什么？",
                window.start_timestamp + index,
            )
        )
    gate = BatchAwareFinalQuestionGate(failing_batch=2)
    processor = DailyQuestionProcessor(
        storage,
        GateScopeService(),  # type: ignore[arg-type]
        final_question_reviewer=gate,
        timezone_name="Asia/Shanghai",
    )

    result = asyncio.run(processor.process_date(report_date))

    assert gate.batch_sizes == [20, 20, 5]
    assert result.status == "RETRY_PENDING"
    assert result.included_count == 25
    assert result.error_count == 20
    assert processor.progress == ("2026-07-12", 45, 45)
    candidates, total = storage.list_question_candidates(limit=None)
    assert total == 45
    assert sum(item.final_decision == "INCLUDE" for item in candidates) == 25
    assert sum(item.final_decision == "AUTO_REVIEW_ERROR" for item in candidates) == 20
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
    assert scope.batch_calls == 1
    storage.close()


def test_large_day_is_split_into_full_coverage_batches(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = date(2026, 7, 12)
    window = natural_day_window(report_date, "Asia/Shanghai")
    for index in range(425):
        storage.insert_message(
            _stored(
                f"m-{index}",
                f"纳入：第 {index} 个校园问题？",
                window.start_timestamp + index,
            )
        )
    scope = FakeScopeService()
    processor = DailyQuestionProcessor(
        storage,
        scope,  # type: ignore[arg-type]
        timezone_name="Asia/Shanghai",
        concurrency=2,
    )

    result = asyncio.run(processor.process_date(report_date))

    assert result.status == "COMPLETED"
    assert result.candidates_saved == 425
    assert result.included_count == 425
    assert scope.batch_calls == 3
    assert scope.batch_sizes == [200, 200, 25]
    assert sorted(scope.batch_message_sizes) == [55, 230, 255]
    assert len(scope.seen) == 425
    assert len(set(scope.seen)) == 425
    storage.close()
