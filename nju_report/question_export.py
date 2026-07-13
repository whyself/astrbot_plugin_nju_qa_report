"""Cumulative concise-question CSV export."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import CoverageStatus, InvestigationResult, QuestionCandidate, QuestionCluster
from .storage import ReportStorage

DECISION_LABELS = {
    "INCLUDE": "已纳入：与南哪知识库维护范围相关",
    "DROP": "已排除：低质量或无关",
    "DROP_LOW_CONFIDENCE": "已排除：AI 自动复核后仍不明确",
    "AUTO_REVIEW": "自动复核中",
    "AUTO_REVIEW_ERROR": "技术错误：已留档",
}


class QuestionCsvExporter:
    """Write one atomically replaced file containing every retained candidate."""

    def __init__(self, storage: ReportStorage, export_dir: Path, *, timezone_name: str) -> None:
        self._storage = storage
        self._export_dir = Path(export_dir)
        self._timezone = ZoneInfo(timezone_name)

    @property
    def output_path(self) -> Path:
        return self._export_dir / "全部问题简表.csv"

    def export_all(self) -> tuple[Path, int]:
        candidates, total = self._storage.list_question_candidates(limit=None)
        self._export_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_path
        temporary = output.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "问题编号",
                        "日期",
                        "筛选结果（含中文解释）",
                        "AI 聚合问题",
                        "原始问题（已脱敏）",
                        "分类",
                        "筛选理由",
                        "群聊（别名）",
                        "消息时间",
                    )
                )
                for candidate in candidates:
                    writer.writerow(self._row(candidate))
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output.resolve(), total

    def export_report_questions(
        self,
        *,
        report_date: str | None = None,
        status: CoverageStatus | None = None,
    ) -> tuple[Path, int]:
        """Export aggregated questions filtered by date and knowledge coverage."""

        clusters = self._storage.list_question_clusters(report_date)
        investigations = self._storage.investigations_for_date(report_date)
        selected = [
            cluster
            for cluster in clusters
            if status is None
            or _public_status(investigations.get(cluster.question_code)) is status
        ]
        self._export_dir.mkdir(parents=True, exist_ok=True)
        date_part = report_date or "全部日期"
        status_part = _STATUS_FILE_LABELS.get(status, "全部状态")
        output = self._export_dir / f"日报问题-{date_part}-{status_part}.csv"
        temporary = output.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    (
                        "问题编号",
                        "日期",
                        "知识库状态",
                        "聚合问题",
                        "分类",
                        "出现次数",
                        "群内回答（AI筛选并脱敏）",
                        "调查结论",
                        "仍缺少",
                        "维护建议",
                        "知识库引用",
                    )
                )
                for cluster in selected:
                    writer.writerow(
                        _report_row(cluster, investigations.get(cluster.question_code))
                    )
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        return output.resolve(), len(selected)

    def _row(self, candidate: QuestionCandidate) -> tuple[str, ...]:
        sent_at = ""
        if candidate.sent_at_utc > 0:
            sent_at = datetime.fromtimestamp(
                candidate.sent_at_utc,
                tz=self._timezone,
            ).strftime("%Y-%m-%d %H:%M:%S")
        return (
            candidate.question_code,
            candidate.report_date,
            DECISION_LABELS.get(candidate.final_decision, candidate.final_decision),
            candidate.canonical_question,
            candidate.original_question,
            candidate.category,
            candidate.reason,
            candidate.group_alias,
            sent_at,
        )


_STATUS_FILE_LABELS = {
    CoverageStatus.ANSWERABLE: "明确回答",
    CoverageStatus.PARTIAL: "部分覆盖",
    CoverageStatus.NO_USABLE_EVIDENCE: "未找到可用信息",
    CoverageStatus.ERROR: "程序执行异常",
}


def _public_status(result: InvestigationResult | None) -> CoverageStatus:
    if result is None or result.status in {CoverageStatus.INCOMPLETE, CoverageStatus.ERROR}:
        return CoverageStatus.ERROR
    return result.status


def _report_row(
    cluster: QuestionCluster,
    investigation: InvestigationResult | None,
) -> tuple[str, ...]:
    status = _public_status(investigation)
    answers = "｜".join(answer.redacted_text for answer in cluster.answers)
    evidence = (
        "｜".join(f"{item.title} {item.source_url}" for item in investigation.evidence)
        if investigation
        else ""
    )
    return (
        cluster.question_code,
        cluster.report_date,
        _STATUS_FILE_LABELS[status],
        cluster.canonical_question,
        cluster.category,
        str(cluster.occurrence_count),
        answers,
        investigation.summary if investigation else "调查结果不存在",
        investigation.missing_information if investigation else "调查未正常完成",
        investigation.recommendation if investigation else "请重新调查",
        evidence,
    )
