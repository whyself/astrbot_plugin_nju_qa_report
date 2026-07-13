"""Operator-facing startup diagnostics without leaking configured secrets."""

from __future__ import annotations

import asyncio
import smtplib
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_agent import CommunityAnswerAgent
from .capture_writer import AsyncCaptureWriter
from .config import PluginConfig
from .models import QuestionCluster, ScopeDecision, StoredMessage
from .scope_classifier import AutoScopeReviewService
from .storage import ReportStorage


@dataclass(frozen=True, slots=True)
class StartupCheck:
    level: str
    name: str
    detail: str


class StartupCheckService:
    """Check local configuration and optionally connect to external dependencies."""

    def __init__(
        self,
        *,
        config: PluginConfig,
        storage: ReportStorage,
        capture_writer: AsyncCaptureWriter,
        scope_review_service: AutoScopeReviewService,
        answer_agent: CommunityAnswerAgent,
        astrbot_context: Any,
        export_dir: Path,
        embedding_probe: Callable[[], Awaitable[int]] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._capture_writer = capture_writer
        self._scope_review_service = scope_review_service
        self._answer_agent = answer_agent
        self._context = astrbot_context
        self._export_dir = Path(export_dir)
        self._embedding_probe = embedding_probe

    async def run(self, *, live: bool = False) -> list[StartupCheck]:
        checks = await asyncio.to_thread(self._local_checks)
        if live:
            checks.append(await self._check_llm_live())
            checks.append(await self._check_answer_agent_live())
            checks.append(await self._check_embedding_live())
            checks.append(await asyncio.to_thread(self._check_yuque_live))
            checks.append(await asyncio.to_thread(self._check_smtp_live))
        return checks

    def _local_checks(self) -> list[StartupCheck]:
        checks: list[StartupCheck] = []
        checks.append(
            StartupCheck(
                "PASS" if self._storage.initialized else "FAIL",
                "本地数据库",
                "已初始化" if self._storage.initialized else "未初始化",
            )
        )
        if self._storage.initialized:
            sqlite_ok = (
                str(self._storage.pragma("journal_mode")).lower() == "wal"
                and self._storage.pragma("foreign_keys") == 1
            )
            checks.append(
                StartupCheck(
                    "PASS" if sqlite_ok else "FAIL",
                    "SQLite 安全参数",
                    "WAL 和外键已启用" if sqlite_ok else "WAL 或外键未启用",
                )
            )

        checks.append(
            StartupCheck(
                "PASS" if self._capture_writer.running else "FAIL",
                "消息后台写入",
                (
                    f"运行中，待写 {self._capture_writer.pending_count} 条"
                    if self._capture_writer.running
                    else "写入任务未运行"
                ),
            )
        )
        groups = "、".join(self._config.target_group_ids) or "未配置"
        checks.append(
            StartupCheck(
                "PASS" if self._config.target_group_ids else "FAIL",
                "目标群",
                groups,
            )
        )
        checks.append(
            StartupCheck(
                "PASS" if self._config.capture_enabled else "WARN",
                "消息采集开关",
                "已启用" if self._config.capture_enabled else "尚未启用，不会记录新消息",
            )
        )
        checks.append(
            StartupCheck(
                "PASS" if self._config.daily_report_enabled else "WARN",
                "日报计划",
                (
                    f"每天 {self._config.daily_report_time}（{self._config.timezone}）"
                    if self._config.daily_report_enabled
                    else f"未启用；配置时间为 {self._config.daily_report_time}"
                ),
            )
        )
        checks.append(self._check_llm_config())
        checks.append(self._check_embedding_config())
        checks.append(self._check_yuque_config())
        checks.append(self._check_email_config())
        checks.append(self._check_export_path())
        return checks

    def _check_llm_config(self) -> StartupCheck:
        if self._config.llm_provider_id:
            return StartupCheck("PASS", "对话模型", "已指定独立 Provider ID")
        try:
            provider = self._context.get_using_provider()
        except Exception as exc:
            return StartupCheck("FAIL", "对话模型", f"读取默认 Provider 失败：{type(exc).__name__}")
        if provider is None:
            return StartupCheck("FAIL", "对话模型", "未指定且 AstrBot 没有默认 Provider")
        return StartupCheck("PASS", "对话模型", "使用 AstrBot 当前默认 Provider")

    def _check_embedding_config(self) -> StartupCheck:
        if not self._config.enable_vector_search:
            return StartupCheck("PASS", "向量检索", "已关闭，将使用本地关键词/grep")
        if self._config.embedding_api_key and self._config.embedding_base_url:
            return StartupCheck(
                "PASS",
                "向量检索",
                f"OpenAI-compatible / {self._config.embedding_model}",
            )
        if self._config.embedding_api_key or self._config.embedding_base_url:
            return StartupCheck(
                "WARN",
                "向量检索",
                "Embedding API Key 与 Base URL 只填写了一项，将回退关键词检索",
            )
        return StartupCheck(
            "WARN",
            "向量检索",
            "未配置 Embedding，将使用本地关键词/grep",
        )

    def _check_yuque_config(self) -> StartupCheck:
        if not self._config.yuque_token:
            return StartupCheck("WARN", "语雀调查源", "未填写 Token")
        if not self._config.approved_repositories:
            return StartupCheck("WARN", "语雀调查源", "已有 Token，但允许仓库列表为空")
        return StartupCheck(
            "PASS",
            "语雀调查源",
            (
                f"允许 {len(self._config.approved_repositories)} 个仓库；"
                f"排除 {len(self._config.excluded_repositories)} 个仓库"
            ),
        )

    def _check_email_config(self) -> StartupCheck:
        missing: list[str] = []
        if not self._config.mail_recipients:
            missing.append("收件人")
        if not self._config.smtp_host:
            missing.append("SMTP 主机")
        if not self._config.mail_from:
            missing.append("发件地址")
        if self._config.smtp_username and not self._config.smtp_password:
            missing.append("SMTP 密码/授权码")
        if missing:
            level = "FAIL" if self._config.daily_report_enabled else "WARN"
            return StartupCheck(level, "邮件日报", "缺少：" + "、".join(missing))
        return StartupCheck(
            "PASS",
            "邮件日报",
            f"独立配置了 {len(self._config.mail_recipients)} 个邮箱收件人",
        )

    def _check_export_path(self) -> StartupCheck:
        try:
            self._export_dir.mkdir(parents=True, exist_ok=True)
            probe = self._export_dir / ".startup-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return StartupCheck("FAIL", "总表导出目录", type(exc).__name__)
        return StartupCheck("PASS", "总表导出目录", str(self._export_dir.resolve()))

    async def _check_llm_live(self) -> StartupCheck:
        resolution = await self._scope_review_service.resolve("南京大学校园卡丢失后如何补办？")
        if resolution.assessment.decision is ScopeDecision.AUTO_REVIEW_ERROR:
            detail = resolution.error_summary or "模型未返回有效结构"
            return StartupCheck("FAIL", "实连：对话模型", detail)
        return StartupCheck("PASS", "实连：对话模型", "调用和结构化结果正常")

    async def _check_answer_agent_live(self) -> StartupCheck:
        cluster = QuestionCluster(
            question_code="STARTUP-Q001",
            report_date="2000-01-01",
            canonical_question="校园卡丢失后如何处理？",
            category="校园服务/校园卡",
            candidate_source_keys=("message:qq:bot:startup-q",),
            representative_questions=("校园卡丢了怎么办？",),
            group_aliases=("启动测试群",),
            first_sent_at_utc=1,
            last_sent_at_utc=1,
        )
        messages = [
            _startup_message("startup-q", 1, "校园卡丢了怎么办？"),
            _startup_message(
                "startup-water",
                2,
                "今天天气不错",
            ),
            _startup_message(
                "startup-a",
                3,
                "先在信息门户挂失，再按学校说明前往服务点处理。",
                reply_to_message_id="startup-q",
            ),
        ]
        try:
            discovery = await self._answer_agent.collect(cluster, messages)
        except Exception as exc:
            return StartupCheck("FAIL", "实连：群答上下文判断", type(exc).__name__)
        if discovery.question_message_ids != ("startup-q",) or not discovery.answers:
            return StartupCheck("FAIL", "实连：群答上下文判断", "未识别测试回答")
        return StartupCheck("PASS", "实连：群答上下文判断", "定长上下文和结果解析正常")

    async def _check_embedding_live(self) -> StartupCheck:
        if not self._config.enable_vector_search:
            return StartupCheck("WARN", "实连：Embedding", "向量检索已关闭，跳过")
        if not self._config.embedding_api_key or not self._config.embedding_base_url:
            return StartupCheck("WARN", "实连：Embedding", "未完整配置，已跳过")
        if self._embedding_probe is None:
            return StartupCheck("FAIL", "实连：Embedding", "插件未注册向量探针")
        try:
            dimensions = await self._embedding_probe()
        except Exception as exc:
            return StartupCheck("FAIL", "实连：Embedding", type(exc).__name__)
        return StartupCheck("PASS", "实连：Embedding", f"调用正常，返回 {dimensions} 维向量")

    def _check_yuque_live(self) -> StartupCheck:
        if not self._config.yuque_token:
            return StartupCheck("WARN", "实连：语雀", "未配置，已跳过")
        request = urllib.request.Request(
            f"{self._config.yuque_api_base}/user",
            headers={
                "X-Auth-Token": self._config.yuque_token,
                "User-Agent": "astrbot-plugin-nju-qa-report/startup-check",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=min(self._config.request_timeout_seconds, 30),
            ) as response:
                status = int(response.status)
        except Exception as exc:
            return StartupCheck("FAIL", "实连：语雀", type(exc).__name__)
        return StartupCheck(
            "PASS" if 200 <= status < 300 else "FAIL",
            "实连：语雀",
            f"API 状态 {status}（未下载任何仓库正文）",
        )

    def _check_smtp_live(self) -> StartupCheck:
        config_check = self._check_email_config()
        if config_check.level != "PASS":
            return StartupCheck("WARN", "实连：SMTP", "邮件配置不完整，已跳过")
        timeout = min(self._config.request_timeout_seconds, 30)
        try:
            if self._config.smtp_use_ssl:
                client: smtplib.SMTP = smtplib.SMTP_SSL(
                    self._config.smtp_host,
                    self._config.smtp_port,
                    timeout=timeout,
                )
            else:
                client = smtplib.SMTP(
                    self._config.smtp_host,
                    self._config.smtp_port,
                    timeout=timeout,
                )
            with client:
                client.ehlo()
                if not self._config.smtp_use_ssl:
                    client.starttls()
                    client.ehlo()
                if self._config.smtp_username:
                    client.login(
                        self._config.smtp_username,
                        self._config.smtp_password,
                    )
                code, _ = client.noop()
        except Exception as exc:
            return StartupCheck("FAIL", "实连：SMTP", type(exc).__name__)
        return StartupCheck(
            "PASS" if 200 <= code < 300 else "FAIL",
            "实连：SMTP",
            "连接和登录正常，未发送邮件",
        )


def format_startup_checks(checks: list[StartupCheck], *, live: bool) -> str:
    title = "完整启动实连测试" if live else "启动配置检查"
    icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    lines = [title]
    lines.extend(f"{icons.get(item.level, '•')} {item.name}：{item.detail}" for item in checks)
    failures = sum(item.level == "FAIL" for item in checks)
    warnings = sum(item.level == "WARN" for item in checks)
    lines.append(f"汇总：失败 {failures}，提醒 {warnings}")
    if not live:
        lines.append("需要实连检查时使用：/nju_collect test startup live")
    return "\n".join(lines)


def _startup_message(
    external_id: str,
    sent_at_utc: int,
    text: str,
    *,
    reply_to_message_id: str = "",
) -> StoredMessage:
    return StoredMessage(
        platform_id="qq",
        bot_self_id="bot",
        external_message_id=external_id,
        message_fingerprint=external_id,
        session_id="group:startup",
        group_id="startup",
        group_alias="启动测试群",
        sender_id=f"sender-{external_id}",
        sender_name="",
        sent_at_utc=sent_at_utc,
        text=text,
        outline="",
        reply_to_message_id=reply_to_message_id,
        analyzable=True,
    )
