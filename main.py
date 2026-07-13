"""AstrBot integration entry point for the NJU knowledge-gap report plugin."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import time
from zoneinfo import ZoneInfo

from astrbot.api import logger
from astrbot.api import message_components as Comp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .nju_report.capture_writer import AsyncCaptureWriter
from .nju_report.config import PluginConfig
from .nju_report.message_capture import MessageCaptureService
from .nju_report.models import MessageEnvelope
from .nju_report.permissions import PermissionAction, PermissionService
from .nju_report.scope_ai import AstrBotScopeAiClient
from .nju_report.scope_classifier import AutoScopeReviewService
from .nju_report.storage import ReportStorage

PLUGIN_NAME = "astrbot_plugin_nju_qa_report"
REPOSITORY_URL = "https://github.com/whyself/astrbot_plugin_nju_qa_report"


@register(
    PLUGIN_NAME,
    "whyself",
    "南京大学迎新问答采集与知识缺口日报（非官方）",
    "0.1.0",
)
class NjuQaReportPlugin(Star):
    """Assemble services and isolate passive capture from AstrBot's reply flow."""

    def __init__(self, context: Context, config=None):
        # Keep the older public Star constructor contract used by AstrBot 4.16.
        super().__init__(context)
        self.context = context
        self.runtime_config = PluginConfig.from_mapping(config or {})
        self._data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.storage = ReportStorage(self._data_dir / "nju_report.sqlite3")
        self.permissions = PermissionService(self.runtime_config)
        self.capture_service = MessageCaptureService(
            self.runtime_config,
            self.storage,
        )
        self.capture_writer = AsyncCaptureWriter(
            self.capture_service,
            max_queue_size=self.runtime_config.capture_queue_size,
            on_error=self._log_capture_write_error,
        )
        scope_ai = AstrBotScopeAiClient(
            context,
            provider_id=self.runtime_config.llm_provider_id,
            timeout_seconds=self.runtime_config.request_timeout_seconds,
            max_retries=self.runtime_config.max_retries,
        )
        self.scope_review_service = AutoScopeReviewService(
            scope_ai,
            scope_ai,
            enabled=self.runtime_config.scope_auto_review_enabled,
            max_rounds=self.runtime_config.scope_auto_review_max_rounds,
        )

    async def initialize(self) -> None:
        """Open local storage after AstrBot activates the plugin."""

        self.storage.initialize()
        retention_cutoff = int(time()) - (self.runtime_config.raw_message_retention_days * 86400)
        deleted = self.storage.delete_expired_messages(retention_cutoff)
        if deleted:
            logger.info("NJU report pruned %s expired raw messages", deleted)
        self.capture_writer.start()
        capture_state = "已启用" if self.runtime_config.capture_enabled else "未启用"
        logger.info("NJU QA report plugin loaded; message capture %s", capture_state)

    @filter.event_message_type(
        filter.EventMessageType.GROUP_MESSAGE,
        priority=100,
    )
    async def capture_group_message(self, event: AstrMessageEvent) -> None:
        """Persist a group event and never affect other AstrBot handlers."""

        if not self.runtime_config.capture_enabled:
            return
        group_id = str(event.get_group_id() or "")
        if (
            self.runtime_config.capture_mode == "selected_groups"
            and group_id not in self.runtime_config.target_group_ids
        ):
            return
        try:
            message_object = event.message_obj
            sender_id = str(event.get_sender_id() or "")
            self_id = str(event.get_self_id() or "")
            envelope = MessageEnvelope(
                platform_id=str(event.get_platform_id() or event.get_platform_name() or "unknown"),
                bot_self_id=self_id,
                external_message_id=str(getattr(message_object, "message_id", "") or ""),
                session_id=str(event.get_session_id() or ""),
                group_id=group_id,
                sender_id=sender_id,
                sender_name=str(event.get_sender_name() or ""),
                sent_at_utc=_safe_timestamp(
                    getattr(message_object, "timestamp", None),
                    event.created_at,
                ),
                text=str(getattr(message_object, "message_str", "") or ""),
                outline=_safe_outline(event),
                reply_to_message_id=_reply_to_message_id(event),
                is_group_message=True,
                is_self_message=bool(sender_id and sender_id == self_id),
                is_system_message=not bool(sender_id),
            )
            if not self.capture_writer.submit(envelope):
                logger.warning("NJU report capture queue rejected a message")
        except Exception:
            # AstrBot stops the current event when a plugin handler raises. This
            # collector must never block the existing nju_qa plugin or any reply.
            logger.exception("NJU report message capture failed")

    @filter.command_group("南哪日报")
    def nju_report(self):
        """Nontechnical, Chinese report commands."""

    @nju_report.command("帮助")
    async def report_help(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.VIEW_REPORT,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        yield event.plain_result(
            "南哪日报查询指令：\n"
            "/南哪日报 列表 YYYY-MM-DD\n"
            "/南哪日报 查看 YYYYMMDD-QNNN\n"
            "/南哪日报 关于\n\n"
            "当前版本正在实现消息采集与日报处理基础。"
        )

    @nju_report.command("关于")
    async def report_about(self, event: AstrMessageEvent):
        yield event.plain_result(
            f"南大知识缺口日报插件（非官方）\n源代码：{REPOSITORY_URL}\n许可证：AGPL-3.0-or-later"
        )

    @filter.command_group("nju_collect")
    def nju_collect(self):
        """Technical operator commands."""

    @nju_collect.command("status")
    async def operator_status(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        count = self.storage.message_count() if self.storage.initialized else 0
        latest = self.storage.latest_message_timestamp() if count else None
        latest_text = "无"
        if latest is not None:
            latest_text = datetime.fromtimestamp(
                latest,
                tz=ZoneInfo(self.runtime_config.timezone),
            ).strftime("%Y-%m-%d %H:%M:%S")
        capture_state = "已启用" if self.runtime_config.capture_enabled else "未启用"
        auto_review_state = "已启用" if self.runtime_config.scope_auto_review_enabled else "未启用"
        yield event.plain_result(
            "NJU 日报插件状态\n"
            f"消息采集：{capture_state}\n"
            f"已存消息：{count}\n"
            f"待写入消息：{self.capture_writer.pending_count}\n"
            f"丢弃消息：{self.capture_writer.dropped_count}\n"
            f"最后消息：{latest_text}\n"
            f"AI 自动复核：{auto_review_state}"
        )

    @nju_collect.group("test")
    def nju_collect_test(self):
        """Side-effect-free operator checks."""

    @nju_collect_test.command("scope")
    async def test_scope(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return

        message = _command_tail(event.get_message_str(), "nju_collect test scope")
        if not message:
            yield event.plain_result(
                "请提供要测试的问题，例如：\n/nju_collect test scope 一卡通丢了去哪里补办？"
            )
            return

        result = await self.scope_review_service.resolve(message)
        assessment = result.assessment
        lines = [
            "AI 范围审核测试",
            f"结果：{assessment.decision.value}",
            f"理由：{assessment.reason}",
            f"自动复核轮数：{result.review_rounds}",
        ]
        if assessment.canonical_question:
            lines.append(f"聚合问题：{assessment.canonical_question}")
        if assessment.category:
            lines.append(f"分类：{assessment.category}")
        if result.error_summary:
            lines.append(f"技术错误类型：{result.error_summary}")
        yield event.plain_result("\n".join(lines))

    async def terminate(self) -> None:
        """Close local resources during disable or hot reload."""

        await self.capture_writer.close()
        self.storage.close()
        logger.info("NJU QA report plugin unloaded")

    @staticmethod
    def _log_capture_write_error(error: Exception) -> None:
        logger.error(
            "NJU report background capture write failed: %s",
            type(error).__name__,
        )


def _reply_to_message_id(event: AstrMessageEvent) -> str:
    for component in event.get_messages():
        if isinstance(component, Comp.Reply):
            return str(component.id or "")
    return ""


def _safe_outline(event: AstrMessageEvent) -> str:
    """Store fixed component placeholders, never media paths, URLs, or quoted text."""

    labels = {
        "image": "[图片]",
        "record": "[语音]",
        "audio": "[语音]",
        "video": "[视频]",
        "file": "[文件]",
        "forward": "[转发消息]",
        "reply": "[回复消息]",
        "face": "[表情]",
        "at": "[提及用户]",
        "atall": "[提及全体成员]",
    }
    placeholders: list[str] = []
    for component in event.get_messages():
        component_name = type(component).__name__.lower()
        if component_name == "plain":
            continue
        placeholders.append(labels.get(component_name, "[其他消息]"))
    return " ".join(placeholders)


def _safe_timestamp(value: object, fallback: float) -> int:
    try:
        timestamp = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        timestamp = int(fallback)
    return timestamp if timestamp > 0 else int(fallback)


def _command_tail(message: str, command: str) -> str:
    normalized = " ".join(str(message).strip().lstrip("/").split())
    if normalized == command:
        return ""
    prefix = f"{command} "
    if normalized.startswith(prefix):
        return normalized[len(prefix) :].strip()
    return ""
