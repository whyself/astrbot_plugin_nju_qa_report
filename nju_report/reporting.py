"""Render frozen nontechnical reports and deliver them idempotently by email."""

# ruff: noqa: E501 -- Embedded email HTML stays readable as a single template.

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import smtplib
import ssl
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import PluginConfig
from .models import (
    CoverageStatus,
    EvidenceItem,
    InvestigationResult,
    QuestionCluster,
    ReportArtifact,
)
from .storage import ReportStorage

_STATUS_LABELS = {
    CoverageStatus.ANSWERABLE: "可找到明确回答",
    CoverageStatus.PARTIAL: "找到部分相关资料",
    CoverageStatus.NO_USABLE_EVIDENCE: "知识库未找到可用信息",
    CoverageStatus.INCOMPLETE: "程序执行异常",
    CoverageStatus.ERROR: "程序执行异常",
}


def coverage_label(status: CoverageStatus) -> str:
    return _STATUS_LABELS[status]


def public_coverage_status(status: CoverageStatus) -> CoverageStatus:
    """Fold the legacy incomplete state into the public execution-error state."""

    return CoverageStatus.ERROR if status is CoverageStatus.INCOMPLETE else status


def coverage_counts(
    clusters: list[QuestionCluster],
    investigations: dict[str, InvestigationResult],
) -> dict[CoverageStatus, int]:
    counts = Counter(
        public_coverage_status(investigations[item.question_code].status)
        if item.question_code in investigations
        else CoverageStatus.ERROR
        for item in clusters
    )
    return {
        status: counts[status]
        for status in (
            CoverageStatus.ANSWERABLE,
            CoverageStatus.PARTIAL,
            CoverageStatus.NO_USABLE_EVIDENCE,
            CoverageStatus.ERROR,
        )
    }


def community_context_degraded_count(clusters: list[QuestionCluster]) -> int:
    return sum(item.community_context_degraded for item in clusters)


def format_coverage_counts(
    counts: dict[CoverageStatus, int],
    *,
    community_context_degraded: int = 0,
) -> str:
    text = (
        "状态统计："
        f"未找到可用信息 {counts[CoverageStatus.NO_USABLE_EVIDENCE]}｜"
        f"部分覆盖 {counts[CoverageStatus.PARTIAL]}｜"
        f"明确回答 {counts[CoverageStatus.ANSWERABLE]}｜"
        f"程序执行异常 {counts[CoverageStatus.ERROR]}"
    )
    return text + f"｜社区上下文降级 {community_context_degraded}"


def coverage_list_order(status: CoverageStatus) -> int:
    """Order public list rows as missing, partial, answerable, then execution errors."""

    public_status = public_coverage_status(status)
    return {
        CoverageStatus.NO_USABLE_EVIDENCE: 0,
        CoverageStatus.PARTIAL: 1,
        CoverageStatus.ANSWERABLE: 2,
        CoverageStatus.ERROR: 3,
    }[public_status]


@dataclass(frozen=True, slots=True)
class DeliverySummary:
    sent: int = 0
    skipped: int = 0
    failed: int = 0


def recipient_hash(recipient: str) -> str:
    """Return the stable privacy-preserving key used for mail idempotency."""

    return hashlib.sha256(recipient.casefold().encode("utf-8")).hexdigest()


class ReportService:
    def __init__(
        self,
        config: PluginConfig,
        storage: ReportStorage,
        report_dir: Path,
    ) -> None:
        self._config = config
        self._storage = storage
        self._report_dir = Path(report_dir)

    async def build(self, report_date: str) -> ReportArtifact:
        clusters = await asyncio.to_thread(self._storage.list_question_clusters, report_date)
        investigations = await asyncio.to_thread(
            self._storage.investigations_for_date,
            report_date,
        )
        window = await asyncio.to_thread(self._storage.processing_window, report_date)
        summary = _summary_payload(
            report_date,
            clusters,
            investigations,
            screening_errors=window.error_count if window else 0,
        )
        summary_json = json.dumps(
            summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        subject = _subject(self._config.mail_subject_prefix, report_date, summary)
        rendered = _render_html(report_date, clusters, investigations, summary)
        content_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
        path = self._report_dir / f"nju-report-{report_date}-{content_hash}.html"
        await asyncio.to_thread(_write_once, path, rendered)
        return await asyncio.to_thread(
            self._storage.save_report,
            report_date=report_date,
            subject=subject,
            html_path=str(path),
            summary_json=summary_json,
        )

    async def deliver(self, report: ReportArtifact, *, force: bool = False) -> DeliverySummary:
        del force  # Retries failed deliveries; successful recipients stay idempotently skipped.
        if not self._mail_configured():
            raise RuntimeError("邮件配置不完整")
        try:
            full_html = await asyncio.to_thread(
                Path(report.html_path).read_text,
                encoding="utf-8",
            )
        except OSError as exc:
            raise RuntimeError("本地日报 HTML 不存在或无法读取") from exc
        clusters = await asyncio.to_thread(
            self._storage.list_question_clusters,
            report.report_date,
        )
        investigations = await asyncio.to_thread(
            self._storage.investigations_for_date,
            report.report_date,
        )
        mail_text = _render_mail_text(report.report_date, clusters, investigations)
        mail_html = _render_mail_html(report.report_date, clusters, investigations)
        sent = skipped = failed = 0
        for recipient in self._config.mail_recipients:
            hashed_recipient = recipient_hash(recipient)
            claimed = await asyncio.to_thread(
                self._storage.begin_mail_delivery,
                report.report_id,
                hashed_recipient,
            )
            if not claimed:
                skipped += 1
                continue
            delivered = await self._send_and_record(
                report,
                recipient,
                hashed_recipient,
                mail_text,
                mail_html,
                full_html,
            )
            if delivered:
                sent += 1
            else:
                failed += 1
        return DeliverySummary(sent=sent, skipped=skipped, failed=failed)

    async def _send_and_record(
        self,
        report: ReportArtifact,
        recipient: str,
        hashed_recipient: str,
        mail_text: str,
        mail_html: str,
        full_html: str,
    ) -> bool:
        """Finish one SMTP attempt and persist its outcome before honoring cancellation."""

        cancelled = False
        send_error: Exception | None = None
        send_task = asyncio.create_task(
            asyncio.to_thread(
                self._send_one,
                recipient,
                report.subject,
                mail_text,
                mail_html,
                full_html,
                Path(report.html_path),
            )
        )
        try:
            await asyncio.shield(send_task)
        except asyncio.CancelledError:
            cancelled = True
            try:
                await send_task
            except Exception as exc:
                send_error = exc
        except Exception as exc:
            send_error = exc

        complete_task = asyncio.create_task(
            asyncio.to_thread(
                self._storage.complete_mail_delivery,
                report.report_id,
                hashed_recipient,
                error_summary=type(send_error).__name__ if send_error is not None else "",
            )
        )
        try:
            await asyncio.shield(complete_task)
        except asyncio.CancelledError:
            cancelled = True
            await complete_task

        if cancelled:
            raise asyncio.CancelledError
        return send_error is None

    def _mail_configured(self) -> bool:
        return bool(
            self._config.smtp_host
            and self._config.smtp_username
            and self._config.smtp_password
            and self._config.mail_from
            and self._config.mail_recipients
        )

    def _send_one(
        self,
        recipient: str,
        subject: str,
        text_body: str,
        html_body: str,
        full_html: str,
        path: Path,
    ) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._config.mail_from
        message["To"] = recipient
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")
        if self._config.attach_full_html:
            message.add_attachment(
                full_html.encode("utf-8"),
                maintype="text",
                subtype="html",
                filename=path.name,
            )
        timeout = min(self._config.request_timeout_seconds, 120)
        if self._config.smtp_use_ssl:
            with smtplib.SMTP_SSL(
                self._config.smtp_host,
                self._config.smtp_port,
                timeout=timeout,
                context=ssl.create_default_context(),
            ) as client:
                client.login(self._config.smtp_username, self._config.smtp_password)
                client.send_message(message)
        else:
            with smtplib.SMTP(
                self._config.smtp_host,
                self._config.smtp_port,
                timeout=timeout,
            ) as client:
                client.ehlo()
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
                client.login(self._config.smtp_username, self._config.smtp_password)
                client.send_message(message)


def format_question_detail(
    cluster: QuestionCluster,
    investigation: InvestigationResult | None,
    *,
    timezone_name: str,
) -> str:
    timezone = ZoneInfo(timezone_name)
    first = datetime.fromtimestamp(cluster.first_sent_at_utc, timezone).strftime("%Y-%m-%d %H:%M")
    last = datetime.fromtimestamp(cluster.last_sent_at_utc, timezone).strftime("%Y-%m-%d %H:%M")
    result = investigation or InvestigationResult(
        question_code=cluster.question_code,
        status=CoverageStatus.ERROR,
        summary="尚未执行知识库调查。",
        missing_information="尚无调查结果。",
        recommendation="请管理员运行该日期的完整日报处理。",
    )
    lines = [
        f"问题编号：{cluster.question_code}",
        f"聚合问题：{cluster.canonical_question}",
        f"分类：{cluster.category or '未分类'}",
        f"群聊：{'、'.join(cluster.group_aliases) or '未设置别名'}",
        f"时间范围：{first} 至 {last}",
        f"知识库覆盖：{_STATUS_LABELS[result.status]}",
        f"调查结论：{result.summary}",
        f"仍缺少：{result.missing_information}",
        f"维护建议：{result.recommendation}",
        "问题表达（AI 已归纳脱敏）：",
    ]
    if cluster.community_context_degraded:
        lines.append("社区上下文：降级（已使用安全回退）")
    lines.extend(f"- {item}" for item in cluster.representative_questions[:5])
    lines.append("群聊回答摘要（AI 已归纳脱敏，未经核实）：")
    lines.extend(f"- {item.redacted_text}" for item in cluster.answers[:5])
    if not cluster.answers:
        lines.append("- 未发现明确回答")
    lines.append("知识库引用：")
    visible_evidence = _visible_evidence(result.evidence)
    lines.extend(f"- {item.title}｜{item.source_url}" for item in visible_evidence)
    if not visible_evidence:
        lines.append("- 无")
    return "\n".join(lines)


def _summary_payload(
    report_date: str,
    clusters: list[QuestionCluster],
    investigations: dict[str, InvestigationResult],
    *,
    screening_errors: int,
) -> dict[str, object]:
    public_counts = coverage_counts(clusters, investigations)
    return {
        "report_date": report_date,
        "question_count": len(clusters),
        "status_counts": {
            status.value: public_counts.get(status, 0) for status in CoverageStatus
        },
        "groups": sorted({alias for item in clusters for alias in item.group_aliases}),
        "screening_errors": screening_errors,
        "community_context_degraded": community_context_degraded_count(clusters),
    }


def _subject(prefix: str, report_date: str, summary: dict[str, object]) -> str:
    counts = summary["status_counts"]
    assert isinstance(counts, dict)
    return (
        f"[{prefix}][{report_date}] 问题{summary['question_count']}｜"
        f"无可用知识{counts[CoverageStatus.NO_USABLE_EVIDENCE.value]}｜"
        f"部分覆盖{counts[CoverageStatus.PARTIAL.value]}｜"
        f"程序异常{counts[CoverageStatus.ERROR.value]}"
    )


def _render_mail_text(
    report_date: str,
    clusters: list[QuestionCluster],
    investigations: dict[str, InvestigationResult],
) -> str:
    counts = coverage_counts(clusters, investigations)
    degraded = community_context_degraded_count(clusters)
    lines = [
        f"南大知识缺口日报 {report_date}",
        (
            f"问题 {len(clusters)}｜"
            f"未找到 {counts[CoverageStatus.NO_USABLE_EVIDENCE]}｜"
            f"异常 {counts[CoverageStatus.ERROR]}｜"
            f"部分覆盖 {counts[CoverageStatus.PARTIAL]}｜"
            f"明确回答 {counts[CoverageStatus.ANSWERABLE]}"
        ),
        "",
    ]
    lines.insert(2, f"社区上下文降级 {degraded}")
    for cluster, status in _ordered_mail_items(clusters, investigations):
        answer = _mail_answer(cluster)
        lines.extend(
            (
                f"问题：{cluster.question_code}｜{_mail_shorten(cluster.canonical_question, 100)}",
                f"状态：{coverage_label(status)}",
                f"回答：{_mail_shorten(answer, 180)}",
                "",
            )
        )
    lines.append("完整调查、维护建议和引用见附件 HTML。")
    return "\n".join(lines)


_MAIL_STATUS_COLORS = {
    CoverageStatus.ANSWERABLE: ("#166534", "#dcfce7", "#86efac"),
    CoverageStatus.PARTIAL: ("#854d0e", "#fef9c3", "#fde047"),
    CoverageStatus.NO_USABLE_EVIDENCE: ("#991b1b", "#fee2e2", "#fca5a5"),
    CoverageStatus.ERROR: ("#991b1b", "#fee2e2", "#fca5a5"),
}

_MAIL_STATUS_ORDER = {
    CoverageStatus.NO_USABLE_EVIDENCE: 0,
    CoverageStatus.ERROR: 1,
    CoverageStatus.PARTIAL: 2,
    CoverageStatus.ANSWERABLE: 3,
}


def _render_mail_html(
    report_date: str,
    clusters: list[QuestionCluster],
    investigations: dict[str, InvestigationResult],
) -> str:
    counts = coverage_counts(clusters, investigations)
    degraded = community_context_degraded_count(clusters)
    rows: list[str] = []
    for cluster, status in _ordered_mail_items(clusters, investigations):
        foreground, background, border = _MAIL_STATUS_COLORS[status]
        rows.append(
            '<div style="padding:12px 0;border-bottom:1px solid #e5e7eb">'
            '<div style="margin:0 0 6px"><strong>问题：</strong>'
            f'{html.escape(cluster.question_code)}｜'
            f'{html.escape(_mail_shorten(cluster.canonical_question, 100))}</div>'
            '<div style="margin:0 0 6px"><strong>状态：</strong>'
            f'<span style="display:inline-block;padding:1px 8px;border-radius:999px;'
            f'color:{foreground};background:{background};border:1px solid {border};'
            f'font-weight:600">{html.escape(coverage_label(status))}</span></div>'
            '<div style="margin:0"><strong>回答：</strong>'
            f'{html.escape(_mail_shorten(_mail_answer(cluster), 180))}</div>'
            '</div>'
        )
    summary = (
        f"问题 {len(clusters)}｜"
        f"未找到 {counts[CoverageStatus.NO_USABLE_EVIDENCE]}｜"
        f"异常 {counts[CoverageStatus.ERROR]}｜"
        f"部分覆盖 {counts[CoverageStatus.PARTIAL]}｜"
        f"明确回答 {counts[CoverageStatus.ANSWERABLE]}"
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head>'
        '<body style="margin:0;background:#ffffff;color:#20242a">'
        '<div style="max-width:760px;margin:0 auto;padding:18px;'
        'font:14px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,Microsoft YaHei,sans-serif">'
        f'<div style="font-size:18px;font-weight:700;margin-bottom:4px">南大知识缺口日报 '
        f'{html.escape(report_date)}</div>'
        f'<div style="color:#4b5563;margin-bottom:8px">{html.escape(summary)}</div>'
        f'<div style="color:#4b5563;margin-bottom:8px">社区上下文降级 {degraded}</div>'
        f'{"".join(rows)}'
        '<div style="margin-top:12px;color:#6b7280">完整调查、维护建议和引用见附件 HTML。</div>'
        '</div></body></html>'
    )


def _mail_answer(cluster: QuestionCluster) -> str:
    if not cluster.answers:
        return "未发现明确回答"
    return "；".join(item.redacted_text for item in cluster.answers[:3])


def _ordered_mail_items(
    clusters: list[QuestionCluster],
    investigations: dict[str, InvestigationResult],
) -> list[tuple[QuestionCluster, CoverageStatus]]:
    items = [
        (
            cluster,
            public_coverage_status(
                investigations[cluster.question_code].status
                if cluster.question_code in investigations
                else CoverageStatus.ERROR
            ),
        )
        for cluster in clusters
    ]
    return sorted(
        items,
        key=lambda item: (
            _MAIL_STATUS_ORDER[item[1]],
            item[0].report_date,
            item[0].question_code,
        ),
    )


def _mail_shorten(value: str, maximum: int) -> str:
    compact = " ".join(value.split()).strip()
    if len(compact) <= maximum:
        return compact
    return compact[: maximum - 1].rstrip() + "…"


def _render_html(
    report_date: str,
    clusters: list[QuestionCluster],
    investigations: dict[str, InvestigationResult],
    summary: dict[str, object],
) -> str:
    counts = summary["status_counts"]
    assert isinstance(counts, dict)
    cards = []
    for cluster in clusters:
        degradation_notice = (
            '<p class="meta">社区上下文：降级（已使用安全回退）</p>'
            if cluster.community_context_degraded
            else ""
        )
        result = investigations.get(cluster.question_code)
        if result is None:
            result = InvestigationResult(
                question_code=cluster.question_code,
                status=CoverageStatus.ERROR,
                summary="尚未执行知识库调查。",
                missing_information="尚无调查结果。",
                recommendation="请管理员重新运行日报。",
            )
        answers = "".join(
            f"<li>{html.escape(item.redacted_text)}</li>" for item in cluster.answers[:5]
        )
        if not answers:
            answers = "<li>未发现明确回答</li>"
        questions = "".join(
            f"<li>{html.escape(item)}</li>" for item in cluster.representative_questions[:5]
        )
        visible_evidence = _visible_evidence(result.evidence)
        evidence = (
            "".join(
                f'<li><a href="{html.escape(item.source_url, quote=True)}">{html.escape(item.title)}</a><br><span>{html.escape(_display_excerpt(item.excerpt))}</span></li>'
                for item in visible_evidence
            )
            or "<li>无</li>"
        )
        cards.append(
            f"""
            <article>
              {degradation_notice}
              <h2>{html.escape(cluster.question_code)} · {html.escape(cluster.canonical_question)}</h2>
              <p class="meta">{html.escape(cluster.category or "未分类")} · {_STATUS_LABELS[result.status]}</p>
              <h3>问题表达（AI 已归纳脱敏）</h3><ul>{questions}</ul>
              <h3>群聊回答摘要（AI 已归纳脱敏，未经核实）</h3><ul>{answers}</ul>
              <h3>知识库调查</h3><p>{html.escape(result.summary)}</p>
              <p><strong>仍缺少：</strong>{html.escape(result.missing_information)}</p>
              <p><strong>维护建议：</strong>{html.escape(result.recommendation)}</p>
              <h3>知识库引用</h3><ul>{evidence}</ul>
              <p class="command">有查看权限的用户：/南哪日报 查看 {html.escape(cluster.question_code)}</p>
            </article>
            """
        )
    groups = "、".join(summary["groups"]) if summary["groups"] else "无"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>南大知识缺口日报 {html.escape(report_date)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;max-width:920px;margin:0 auto;padding:28px;background:#f6f7f9;color:#20242a;line-height:1.7}}
header,article{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:22px;margin-bottom:18px}}h1,h2,h3{{line-height:1.35}}h2{{font-size:20px}}h3{{font-size:15px;margin-bottom:4px}}.meta{{color:#5b6472}}.stats{{display:flex;flex-wrap:wrap;gap:10px}}.stats span{{background:#eef3ff;border-radius:999px;padding:5px 11px}}a{{color:#185abd}}.command{{background:#f3f4f6;padding:9px 12px;border-radius:7px}}footer{{color:#68707d;font-size:13px;padding:12px}}</style></head>
<body><header><h1>南大知识缺口日报</h1><p>报告日期：{html.escape(report_date)}｜群聊：{html.escape(groups)}</p>
<div class="stats"><span>问题 {summary["question_count"]}</span><span>明确回答 {counts[CoverageStatus.ANSWERABLE.value]}</span><span>部分覆盖 {counts[CoverageStatus.PARTIAL.value]}</span><span>未找到可用信息 {counts[CoverageStatus.NO_USABLE_EVIDENCE.value]}</span><span>程序执行异常 {counts[CoverageStatus.ERROR.value]}</span><span>筛选技术错误 {summary["screening_errors"]}</span><span>社区上下文降级 {summary["community_context_degraded"]}</span></div></header>
{"".join(cards) if cards else "<article><p>本日没有纳入日报的问题。</p></article>"}
<footer>本报告由非官方维护辅助插件生成。群聊回答由 AI 去除身份信息后归纳，内容未经核实；知识结论仅依据配置允许的语雀仓库。</footer></body></html>"""


def _write_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _visible_evidence(
    evidence: tuple[EvidenceItem, ...],
    *,
    maximum: int = 5,
) -> tuple[EvidenceItem, ...]:
    """Hide duplicate document copies and bound citations in the public report."""

    result: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        normalized_title = _canonical_evidence_title(item.title)
        key = (item.namespace.casefold(), normalized_title or item.document_id.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= maximum:
            break
    return tuple(result)


def _canonical_evidence_title(value: str) -> str:
    normalized = "".join(value.split()).casefold()
    for suffix in ("（副本）", "(副本)", "副本"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _display_excerpt(value: str, *, maximum: int = 420) -> str:
    compact = " ".join(value.split()).strip()
    if len(compact) <= maximum:
        return compact
    return compact[: maximum - 1].rstrip() + "…"
