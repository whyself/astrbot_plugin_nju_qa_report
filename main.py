"""AstrBot integration entry point for the NJU knowledge-gap report plugin."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from time import time
from zoneinfo import ZoneInfo

from astrbot.api import logger
from astrbot.api import message_components as Comp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .nju_report.aggregation import QuestionAggregationService
from .nju_report.capture_writer import AsyncCaptureWriter
from .nju_report.config import PluginConfig
from .nju_report.investigation import AstrBotInvestigationAiClient, InvestigationService
from .nju_report.knowledge import KnowledgeError, KnowledgeService
from .nju_report.message_capture import MessageCaptureService
from .nju_report.models import MessageEnvelope
from .nju_report.permissions import PermissionAction, PermissionService
from .nju_report.qce_import import QceHistoryImporter, QceImportError
from .nju_report.question_export import DECISION_LABELS, QuestionCsvExporter
from .nju_report.question_processor import (
    DailyQuestionProcessor,
    today_in_timezone,
)
from .nju_report.reporting import ReportService, coverage_label, format_question_detail
from .nju_report.scope_ai import AstrBotScopeAiClient
from .nju_report.scope_classifier import AutoScopeReviewService
from .nju_report.startup_checks import StartupCheckService, format_startup_checks
from .nju_report.storage import ReportStorage
from .nju_report.workflow import DailyReportWorkflow, DailyScheduler, FullReportRunResult

PLUGIN_NAME = "astrbot_plugin_nju_qa_report"
REPOSITORY_URL = "https://github.com/whyself/astrbot_plugin_nju_qa_report"


@register(
    PLUGIN_NAME,
    "whyself",
    "南京大学迎新问答采集与知识缺口日报（非官方）",
    "0.2.1",
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
        self.question_processor = DailyQuestionProcessor(
            self.storage,
            self.scope_review_service,
            timezone_name=self.runtime_config.timezone,
            concurrency=self.runtime_config.batch_concurrency,
        )
        self.question_exporter = QuestionCsvExporter(
            self.storage,
            self._data_dir / "exports",
            timezone_name=self.runtime_config.timezone,
        )
        self.history_importer = QceHistoryImporter(
            self.runtime_config,
            self.storage,
            base_dir=self._data_dir,
        )
        self.knowledge_service = KnowledgeService(self.runtime_config, self.storage)
        self.aggregation_service = QuestionAggregationService(
            self.storage,
            timezone_name=self.runtime_config.timezone,
        )
        investigation_ai = AstrBotInvestigationAiClient(
            context,
            provider_id=self.runtime_config.llm_provider_id,
            timeout_seconds=self.runtime_config.request_timeout_seconds,
            max_retries=self.runtime_config.max_retries,
        )
        self.investigation_service = InvestigationService(
            self.runtime_config,
            self.storage,
            self.knowledge_service,
            investigation_ai,
        )
        self.report_service = ReportService(
            self.runtime_config,
            self.storage,
            self._data_dir / "reports",
        )
        self.workflow = DailyReportWorkflow(
            self.storage,
            self.question_processor,
            self.aggregation_service,
            self.investigation_service,
            self.report_service,
            self.knowledge_service,
            timezone_name=self.runtime_config.timezone,
        )
        self.scheduler = DailyScheduler(
            self.workflow,
            timezone_name=self.runtime_config.timezone,
            report_time=self.runtime_config.daily_report_time,
            enabled=self.runtime_config.daily_report_enabled,
        )
        self.startup_checks = StartupCheckService(
            config=self.runtime_config,
            storage=self.storage,
            capture_writer=self.capture_writer,
            scope_review_service=self.scope_review_service,
            astrbot_context=context,
            export_dir=self._data_dir / "exports",
            embedding_probe=self.knowledge_service.test_embedding,
        )

    async def initialize(self) -> None:
        """Open local storage after AstrBot activates the plugin."""

        self.storage.initialize()
        retention_cutoff = int(time()) - (self.runtime_config.raw_message_retention_days * 86400)
        deleted = self.storage.delete_expired_messages(retention_cutoff)
        if deleted:
            logger.info("NJU report pruned %s expired raw messages", deleted)
        self.capture_writer.start()
        self.scheduler.start()
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
            "/南哪日报 列表 [YYYY-MM-DD|全部] [页码]\n"
            "/南哪日报 查看 YYYYMMDD-QNNN\n"
            "/南哪日报 导出\n"
            "/南哪日报 关于\n\n"
            "列表包含已纳入、已排除和技术错误的全部筛选记录；请私聊机器人使用。"
        )

    @nju_report.command("列表")
    async def report_list(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.VIEW_REPORT,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "南哪日报 列表")
        try:
            report_date, page = _parse_list_arguments(tail)
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        page_size = 20
        if report_date is not None:
            clusters = await asyncio.to_thread(
                self.storage.list_question_clusters,
                report_date,
            )
            if clusters:
                start = (page - 1) * page_size
                shown = clusters[start : start + page_size]
                if not shown:
                    yield event.plain_result("没有符合该日期和页码的聚合问题。")
                    return
                investigations = await asyncio.to_thread(
                    self.storage.investigations_for_date,
                    report_date,
                )
                total_pages = max(1, (len(clusters) + page_size - 1) // page_size)
                lines = [f"日报问题（第 {page}/{total_pages} 页，共 {len(clusters)} 条）"]
                for cluster in shown:
                    result = investigations.get(cluster.question_code)
                    label = coverage_label(result.status) if result else "尚未调查"
                    lines.append(
                        f"{cluster.question_code}｜{label}｜出现 {cluster.occurrence_count} 次\n"
                        f"{_shorten(cluster.canonical_question, 80)}"
                    )
                lines.append("私聊发送 /南哪日报 查看 <问题编号> 可看详细报告。")
                yield event.plain_result("\n\n".join(lines))
                return
        candidates, total = await asyncio.to_thread(
            self.storage.list_question_candidates,
            report_date=report_date,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        if not candidates:
            yield event.plain_result("没有符合该日期和页码的本地问题记录。")
            return
        total_pages = max(1, (total + page_size - 1) // page_size)
        lines = [f"问题列表（第 {page}/{total_pages} 页，共 {total} 条）"]
        for candidate in candidates:
            question = (
                candidate.canonical_question or candidate.original_question or "未形成明确问题"
            )
            label = DECISION_LABELS.get(candidate.final_decision, candidate.final_decision)
            lines.append(f"{candidate.question_code}｜{label}\n{_shorten(question, 80)}")
        lines.append("私聊发送 /南哪日报 查看 <问题编号> 可看详细记录。")
        yield event.plain_result("\n\n".join(lines))

    @nju_report.command("查看")
    async def report_show(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.VIEW_REPORT,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        question_code = _command_tail(event.get_message_str(), "南哪日报 查看").upper()
        if not question_code:
            yield event.plain_result("请提供问题编号，例如：/南哪日报 查看 20260712-Q001")
            return
        cluster = await asyncio.to_thread(self.storage.get_question_cluster, question_code)
        if cluster is not None:
            investigation = await asyncio.to_thread(
                self.storage.latest_investigation,
                question_code,
            )
            yield event.plain_result(
                format_question_detail(
                    cluster,
                    investigation,
                    timezone_name=self.runtime_config.timezone,
                )
            )
            return
        candidate = await asyncio.to_thread(self.storage.get_question_candidate, question_code)
        if candidate is None:
            yield event.plain_result("没有找到该问题编号。")
            return
        sent_at = "未知"
        if candidate.sent_at_utc > 0:
            sent_at = datetime.fromtimestamp(
                candidate.sent_at_utc,
                tz=ZoneInfo(self.runtime_config.timezone),
            ).strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"问题编号：{candidate.question_code}",
            f"筛选结果：{DECISION_LABELS.get(candidate.final_decision, candidate.final_decision)}",
            f"AI 聚合问题：{candidate.canonical_question or '未形成'}",
            f"原始问题（已脱敏）：{candidate.original_question or '无文本'}",
            f"分类：{candidate.category or '未分类'}",
            f"筛选理由：{candidate.reason}",
            f"群聊：{candidate.group_alias or '未设置别名'}",
            f"时间：{sent_at}",
        ]
        yield event.plain_result("\n".join(lines))

    @nju_report.command("导出")
    async def report_export(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.VIEW_REPORT,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        path, total = await asyncio.to_thread(self.question_exporter.export_all)
        yield event.plain_result(f"已导出全部 {total} 条问题简表。")
        yield event.chain_result([Comp.File(name=path.name, file=str(path))])

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
        candidate_count = self.storage.question_candidate_count() if self.storage.initialized else 0
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
            f"已留档筛选问题：{candidate_count}\n"
            f"待写入消息：{self.capture_writer.pending_count}\n"
            f"丢弃消息：{self.capture_writer.dropped_count}\n"
            f"最后消息：{latest_text}\n"
            f"AI 自动复核：{auto_review_state}"
        )

    @nju_collect.group("import")
    def nju_collect_import(self):
        """Import QQ Chat Exporter history files."""

    @nju_collect.group("repo")
    def nju_collect_repo(self):
        """Manage allowlisted Yuque knowledge repositories."""

    @nju_collect_repo.command("sync")
    async def operator_repo_sync(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        if self.knowledge_service.syncing:
            yield event.plain_result("已有语雀同步任务正在运行。")
            return
        try:
            result = await self.knowledge_service.sync_all()
        except KnowledgeError as exc:
            yield event.plain_result(f"语雀同步失败：{exc}")
            return
        lines = ["语雀允许仓库同步完成"]
        for item in result.repositories:
            lines.append(
                f"{item.namespace}\n"
                f"文档 {item.documents_seen}；更新 {item.documents_changed}；"
                f"未变 {item.documents_unchanged}；删除 {item.documents_deleted}；"
                f"分块 {item.chunks_written}；向量 {item.embeddings_written}；"
                f"失败 {len(item.failures)}"
            )
        if result.excluded_purged:
            lines.append(
                "排除仓库清理："
                + "、".join(
                    f"{namespace} {count} 篇"
                    for namespace, count in sorted(result.excluded_purged.items())
                )
            )
        yield event.plain_result("\n\n".join(lines))

    @nju_collect_repo.command("status")
    async def operator_repo_status(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        records = await asyncio.to_thread(self.storage.repository_records)
        documents, chunks = await asyncio.to_thread(self.storage.knowledge_counts)
        lines = [f"本地知识库：文档 {documents}，分块 {chunks}"]
        if not records:
            lines.append("尚未运行语雀同步。")
        for item in records:
            lines.append(
                f"{item['namespace']}｜{item['status']}｜{item['display_name'] or '未同步名称'}"
            )
        yield event.plain_result("\n".join(lines))

    @nju_collect_repo.command("search")
    async def operator_repo_search(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        query = _command_tail(event.get_message_str(), "nju_collect repo search")
        if not query:
            yield event.plain_result("用法：/nju_collect repo search 校园卡补办")
            return
        hits = await self.knowledge_service.search(query, limit=5)
        if not hits:
            yield event.plain_result("允许仓库中没有检索到结果；请先运行 repo sync。")
            return
        lines = [f"本地混合检索：{query}"]
        for index, hit in enumerate(hits, start=1):
            lines.append(
                f"{index}. {hit.chunk.title}｜{hit.chunk.namespace}｜{hit.score:.3f}\n"
                f"{_shorten(hit.chunk.content, 180)}\n{hit.chunk.source_url}"
            )
        yield event.plain_result("\n\n".join(lines))

    @nju_collect_import.command("inspect")
    async def operator_import_inspect(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        try:
            sources = await asyncio.to_thread(self.history_importer.inspect_all)
        except QceImportError as exc:
            yield event.plain_result(f"历史记录检查失败：{exc}")
            return
        lines = [f"QCE 历史记录检查通过，共 {len(sources)} 个文件"]
        for source in sources:
            date_range = _timestamp_range(
                source.first_sent_at_utc,
                source.last_sent_at_utc,
                self.runtime_config.timezone,
            )
            lines.append(
                f"{Path(source.path).name}\n"
                f"群聊：{source.chat_name}（{source.group_id}）\n"
                f"消息：{source.message_count} 条；范围：{date_range}"
            )
        lines.append("确认无误后运行：/nju_collect import run")
        yield event.plain_result("\n\n".join(lines))

    @nju_collect_import.command("run")
    async def operator_import_run(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        try:
            await self.capture_writer.flush(timeout_seconds=30)
            results = await asyncio.to_thread(self.history_importer.import_all)
        except (QceImportError, TimeoutError) as exc:
            yield event.plain_result(f"历史记录导入失败：{exc}")
            return
        lines = [f"QCE 历史记录导入完成，共 {len(results)} 个文件"]
        for result in results:
            skipped = sum(result.skipped.values())
            skip_text = "、".join(f"{key} {count}" for key, count in sorted(result.skipped.items()))
            lines.append(
                f"{Path(result.path).name}\n"
                f"扫描 {result.scanned}；新增 {result.imported}；"
                f"重复 {result.duplicates}；跳过 {skipped}"
                + (f"（{skip_text}）" if skip_text else "")
            )
        lines.append("下一步：/nju_collect report run all")
        yield event.plain_result("\n\n".join(lines))

    @nju_collect.group("report")
    def nju_collect_report(self):
        """Idempotent historical report processing."""

    @nju_collect_report.command("run")
    async def operator_report_run(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "nju_collect report run")
        if not tail:
            yield event.plain_result(
                "请指定历史日期或全部，例如：\n"
                "/nju_collect report run 2026-07-12\n"
                "/nju_collect report run all"
            )
            return
        if self.workflow.running:
            yield event.plain_result("已有日报批处理正在运行，请稍后查询状态。")
            return
        try:
            await self.capture_writer.flush(timeout_seconds=30)
        except TimeoutError:
            logger.warning("NJU report run started with capture messages still pending")

        try:
            await self.workflow.sync_knowledge()
        except KnowledgeError as exc:
            yield event.plain_result(f"知识库同步失败，日报未开始：{exc}")
            return

        current_date = today_in_timezone(self.runtime_config.timezone)
        normalized = tail.lower()
        try:
            if normalized in {"all", "全部"}:
                results = await self.workflow.run_all_history(before_date=current_date)
            else:
                requested_date = date.fromisoformat(tail)
                if requested_date >= current_date:
                    yield event.plain_result("只能处理已经结束的自然日，不能锁定今天或未来日期。")
                    return
                results = [await self.workflow.run_date(requested_date)]
        except ValueError:
            yield event.plain_result("日期必须使用 YYYY-MM-DD，或填写 all/全部。")
            return
        except Exception as exc:
            logger.exception("NJU full report processing failed")
            yield event.plain_result(f"完整日报处理失败：{type(exc).__name__}")
            return

        if not results:
            yield event.plain_result("本地没有可处理的历史聊天日期。")
            return
        await asyncio.to_thread(self.question_exporter.export_all)
        yield event.plain_result(_format_full_run_results(results))

    @nju_collect_report.command("preview")
    async def operator_report_preview(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "nju_collect report preview")
        try:
            requested_date = date.fromisoformat(tail)
        except ValueError:
            yield event.plain_result("请提供日期，例如：/nju_collect report preview 2026-07-12")
            return
        report = await asyncio.to_thread(self.storage.latest_report, requested_date.isoformat())
        if report is None:
            yield event.plain_result("该日期尚未生成完整日报。")
            return
        path = Path(report.html_path)
        if not path.is_file():
            yield event.plain_result("该日报的本地 HTML 文件已丢失，请重新运行日报处理。")
            return
        yield event.plain_result(f"日报预览：{report.report_date} / 版本 {report.version}")
        yield event.chain_result([Comp.File(name=path.name, file=str(path))])

    @nju_collect_report.command("send")
    async def operator_report_send(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "nju_collect report send")
        try:
            requested_date = date.fromisoformat(tail)
            delivery = await self.workflow.deliver_latest(requested_date.isoformat())
        except ValueError:
            yield event.plain_result("请提供日期，例如：/nju_collect report send 2026-07-12")
            return
        except RuntimeError as exc:
            yield event.plain_result(f"日报发送失败：{exc}")
            return
        yield event.plain_result(
            f"日报邮件处理完成：发送 {delivery.sent}，"
            f"已发送跳过 {delivery.skipped}，失败 {delivery.failed}"
        )

    @nju_collect.command("investigate")
    async def operator_investigate(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        question_code = _command_tail(event.get_message_str(), "nju_collect investigate").upper()
        cluster = await asyncio.to_thread(self.storage.get_question_cluster, question_code)
        if cluster is None:
            yield event.plain_result("没有找到该聚合问题；请先运行对应日期的日报处理。")
            return
        result = await self.investigation_service.investigate(cluster)
        try:
            await self.report_service.build(cluster.report_date)
        except Exception as exc:
            logger.exception("NJU report rebuild after investigation failed")
            yield event.plain_result(f"调查已留档，但 HTML 更新失败：{type(exc).__name__}")
            return
        yield event.plain_result(
            f"调查完成：{question_code} / {result.status.value}\n{result.summary}"
        )

    @nju_collect_report.command("status")
    async def operator_report_status(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "nju_collect report status")
        try:
            requested_date = date.fromisoformat(tail)
        except ValueError:
            yield event.plain_result("请提供日期，例如：/nju_collect report status 2026-07-12")
            return
        window = await asyncio.to_thread(
            self.storage.processing_window,
            requested_date.isoformat(),
        )
        if window is None:
            yield event.plain_result("该日期尚未运行。")
            return
        clusters = await asyncio.to_thread(
            self.storage.list_question_clusters,
            requested_date.isoformat(),
        )
        investigations = await asyncio.to_thread(
            self.storage.investigations_for_date,
            requested_date.isoformat(),
        )
        report = await asyncio.to_thread(
            self.storage.latest_report,
            requested_date.isoformat(),
        )
        yield event.plain_result(
            f"{window.report_date} 处理状态\n"
            f"状态：{window.status}\n"
            f"扫描消息：{window.messages_scanned}\n"
            f"留档候选：{window.candidates_saved}\n"
            f"纳入：{window.included_count}\n"
            f"排除：{window.dropped_count}\n"
            f"技术错误：{window.error_count}\n"
            f"聚合问题：{len(clusters)}\n"
            f"已调查：{len(investigations)}\n"
            f"HTML 日报：{('版本 ' + str(report.version)) if report else '未生成'}\n"
            f"运行中：{'是' if self.workflow.running else '否'}"
        )

    @nju_collect.command("export")
    async def operator_export(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "nju_collect export").lower()
        if tail not in {"questions", "all", "全部问题", ""}:
            yield event.plain_result("用法：/nju_collect export questions")
            return
        path, total = await asyncio.to_thread(self.question_exporter.export_all)
        yield event.plain_result(f"累计问题总表已刷新，共 {total} 条。")
        yield event.chain_result([Comp.File(name=path.name, file=str(path))])

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

    @nju_collect_test.command("startup")
    async def test_startup(self, event: AstrMessageEvent):
        authorization = self.permissions.authorize(
            sender_id=event.get_sender_id(),
            action=PermissionAction.OPERATE,
            is_private=event.is_private_chat(),
            is_astrbot_admin=event.is_admin(),
        )
        if not authorization.allowed:
            yield event.plain_result(authorization.user_message)
            return
        tail = _command_tail(event.get_message_str(), "nju_collect test startup").lower()
        if tail not in {"", "live"}:
            yield event.plain_result(
                "用法：/nju_collect test startup [live]\n"
                "live 会实连模型、语雀和 SMTP，但不会下载仓库正文或发送邮件。"
            )
            return
        live = tail == "live"
        checks = await self.startup_checks.run(live=live)
        yield event.plain_result(format_startup_checks(checks, live=live))

    async def terminate(self) -> None:
        """Close local resources during disable or hot reload."""

        await self.scheduler.close()
        await self.capture_writer.close()
        await self.knowledge_service.close()
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


def _parse_list_arguments(tail: str) -> tuple[str | None, int]:
    parts = tail.split()
    if len(parts) > 2:
        raise ValueError("用法：/南哪日报 列表 [YYYY-MM-DD|全部] [页码]")
    report_date: str | None = None
    page = 1
    if parts and parts[0] not in {"全部", "all"}:
        try:
            report_date = date.fromisoformat(parts[0]).isoformat()
        except ValueError as exc:
            raise ValueError("日期必须使用 YYYY-MM-DD，或填写 全部。") from exc
    if len(parts) == 2:
        try:
            page = int(parts[1])
        except ValueError as exc:
            raise ValueError("页码必须是正整数。") from exc
        if page < 1:
            raise ValueError("页码必须是正整数。")
    return report_date, page


def _shorten(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _format_full_run_results(results: list[FullReportRunResult]) -> str:
    completed = [item for item in results if item.screening.status == "COMPLETED"]
    retry_pending = [item for item in results if item.screening.status == "RETRY_PENDING"]
    processed = completed + retry_pending
    skipped = [item for item in results if item.screening.skipped]
    failed = [item for item in results if item.screening.status == "FAILED"]
    lines = [
        "历史聊天完整日报处理完成（尚未发送邮件）",
        f"新处理日期：{len(completed)}",
        f"已处理而跳过：{len(skipped)}",
        f"失败日期：{len(failed)}",
        f"需重试日期：{len(retry_pending)}",
        f"扫描消息：{sum(item.screening.messages_scanned for item in processed)}",
        f"本次留档：{sum(item.screening.candidates_saved for item in processed)}",
        f"聚合问题：{sum(item.cluster_count for item in results)}",
        f"生成报告：{sum(item.report is not None for item in results)}",
        f"技术错误：{sum(item.screening.error_count for item in processed)}",
    ]
    if skipped:
        lines.append(
            "筛选跳过日期：" + _compact_dates([item.screening.report_date for item in skipped])
        )
    if failed:
        lines.append("失败日期：" + _compact_dates([item.screening.report_date for item in failed]))
    if retry_pending:
        lines.append(
            "存在筛选技术错误："
            + _compact_dates([item.screening.report_date for item in retry_pending])
            + "；再次运行相同命令会重试。"
        )
    if len(results) == 1:
        item = results[0]
        report_text = f"报告版本 {item.report.version}" if item.report else "未生成报告"
        lines.append(
            f"日期明细：{item.screening.report_date} / {item.screening.status} / "
            f"聚合 {item.cluster_count} / {report_text}"
        )
    lines.append("先用 report preview 检查 HTML；确认后再用 report send 发送邮件。")
    return "\n".join(lines)


def _compact_dates(values: list[str], limit: int = 20) -> str:
    shown = values[:limit]
    suffix = f" 等共 {len(values)} 天" if len(values) > limit else ""
    return "、".join(shown) + suffix


def _timestamp_range(start: int | None, end: int | None, timezone_name: str) -> str:
    if start is None or end is None:
        return "无有效时间"
    timezone = ZoneInfo(timezone_name)
    start_text = datetime.fromtimestamp(start, tz=timezone).strftime("%Y-%m-%d %H:%M:%S")
    end_text = datetime.fromtimestamp(end, tz=timezone).strftime("%Y-%m-%d %H:%M:%S")
    return f"{start_text} ～ {end_text}"
