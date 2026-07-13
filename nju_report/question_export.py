"""Cumulative concise-question CSV export."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import QuestionCandidate
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
