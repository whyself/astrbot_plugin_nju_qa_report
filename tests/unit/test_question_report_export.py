from __future__ import annotations

from pathlib import Path

from nju_report.models import (
    CommunityAnswer,
    CoverageStatus,
    EvidenceItem,
    InvestigationResult,
    QuestionCluster,
)
from nju_report.question_export import QuestionCsvExporter
from nju_report.storage import ReportStorage


def _cluster(code: str, question: str) -> QuestionCluster:
    return QuestionCluster(
        question_code=code,
        report_date="2026-07-12",
        canonical_question=question,
        category="住宿食堂",
        candidate_source_keys=(),
        representative_questions=(question,),
        group_aliases=("测试群",),
        first_sent_at_utc=1,
        last_sent_at_utc=2,
        answers=(
            CommunityAnswer(
                external_message_id="answer-1",
                redacted_text="南二有部分套间，具体安排以学校通知为准。",
                sent_at_utc=2,
                confidence=0.9,
                direct_reply=True,
            ),
        ),
    )


def test_report_csv_can_export_only_missing_knowledge_questions(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    missing = _cluster("20260712-Q001", "南园二舍是否有套间？")
    answerable = _cluster("20260712-Q002", "校园卡在哪里补办？")
    storage.save_question_clusters("2026-07-12", [missing, answerable])
    storage.save_investigation(
        InvestigationResult(
            question_code=missing.question_code,
            status=CoverageStatus.NO_USABLE_EVIDENCE,
            summary="知识库中没有可用于回答套间情况的信息。",
            missing_information="缺少当前宿舍结构说明。",
            recommendation="补充经核实的宿舍结构。",
        )
    )
    storage.save_investigation(
        InvestigationResult(
            question_code=answerable.question_code,
            status=CoverageStatus.ANSWERABLE,
            summary="已有明确补办说明。",
            missing_information="无。",
            recommendation="无需修改。",
            evidence=(
                EvidenceItem(
                    namespace="test/repo",
                    document_id="doc-1",
                    title="校园卡指南",
                    source_url="https://example.com/card",
                    updated_at="2026-01-01",
                    excerpt="补办说明",
                ),
            ),
        )
    )
    exporter = QuestionCsvExporter(
        storage,
        tmp_path / "exports",
        timezone_name="Asia/Shanghai",
    )

    output, count = exporter.export_report_questions(
        status=CoverageStatus.NO_USABLE_EVIDENCE,
    )

    content = output.read_text(encoding="utf-8-sig")
    assert count == 1
    assert "20260712-Q001" in content
    assert "未找到可用信息" in content
    assert "南二有部分套间" in content
    assert "20260712-Q002" not in content
    storage.close()
