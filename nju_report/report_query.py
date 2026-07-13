"""Parse nontechnical report-list query arguments."""

from __future__ import annotations

from datetime import date

from .models import CoverageStatus

_STATUS_ALIASES = {
    "all": None,
    "全部": None,
    "answerable": CoverageStatus.ANSWERABLE,
    "明确": CoverageStatus.ANSWERABLE,
    "明确回答": CoverageStatus.ANSWERABLE,
    "可找到明确回答": CoverageStatus.ANSWERABLE,
    "partial": CoverageStatus.PARTIAL,
    "部分": CoverageStatus.PARTIAL,
    "部分覆盖": CoverageStatus.PARTIAL,
    "找到部分相关资料": CoverageStatus.PARTIAL,
    "missing": CoverageStatus.NO_USABLE_EVIDENCE,
    "no_usable_evidence": CoverageStatus.NO_USABLE_EVIDENCE,
    "未找到": CoverageStatus.NO_USABLE_EVIDENCE,
    "无可用信息": CoverageStatus.NO_USABLE_EVIDENCE,
    "知识库未找到可用信息": CoverageStatus.NO_USABLE_EVIDENCE,
    "error": CoverageStatus.ERROR,
    "异常": CoverageStatus.ERROR,
    "程序异常": CoverageStatus.ERROR,
    "程序执行异常": CoverageStatus.ERROR,
}


def parse_list_arguments(
    tail: str,
) -> tuple[str | None, CoverageStatus | None, int]:
    parts = tail.split()
    if len(parts) > 3:
        raise ValueError(
            "用法：/南哪日报 列表 [YYYY-MM-DD|all] "
            "[answerable|partial|missing|error|all] [页码]"
        )
    report_date: str | None = None
    status: CoverageStatus | None = None
    page = 1
    if parts and parts[0] not in {"全部", "all"}:
        try:
            report_date = date.fromisoformat(parts[0]).isoformat()
        except ValueError as exc:
            raise ValueError("日期必须使用 YYYY-MM-DD，或填写 all。") from exc
    if len(parts) >= 2 and not parts[1].isdigit():
        key = parts[1].casefold()
        if key not in _STATUS_ALIASES:
            raise ValueError(
                "状态必须填写 answerable、partial、missing、error 或 all。"
            )
        status = _STATUS_ALIASES[key]
    if len(parts) == 3 and parts[1].isdigit():
        raise ValueError("三参数格式的第二个参数必须是状态，第三个参数才是页码。")
    page_part = ""
    if len(parts) == 2 and parts[1].isdigit():
        page_part = parts[1]
    elif len(parts) == 3:
        page_part = parts[2]
    if page_part:
        try:
            page = int(page_part)
        except ValueError as exc:
            raise ValueError("页码必须是正整数。") from exc
        if page < 1:
            raise ValueError("页码必须是正整数。")
    return report_date, status, page


def parse_export_arguments(tail: str) -> tuple[str | None, CoverageStatus | None]:
    """Parse optional date and coverage filters for a report-question CSV."""

    parts = tail.split()
    if len(parts) > 2:
        raise ValueError(
            "用法：/南哪日报 导出 [YYYY-MM-DD|all] "
            "[answerable|partial|missing|error|all]"
        )
    if not parts:
        return None, None

    report_date: str | None = None
    status: CoverageStatus | None = None
    first = parts[0].casefold()
    if first in _STATUS_ALIASES and len(parts) == 1:
        return None, _STATUS_ALIASES[first]
    if first not in {"全部", "all"}:
        try:
            report_date = date.fromisoformat(parts[0]).isoformat()
        except ValueError as exc:
            raise ValueError(
                "第一个参数必须是 YYYY-MM-DD、all，或单独填写一个状态。"
            ) from exc
    if len(parts) == 2:
        key = parts[1].casefold()
        if key not in _STATUS_ALIASES:
            raise ValueError(
                "状态必须填写 answerable、partial、missing、error 或 all。"
            )
        status = _STATUS_ALIASES[key]
    return report_date, status
