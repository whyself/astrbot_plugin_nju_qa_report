from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from nju_report.config import PluginConfig
from nju_report.investigation import (
    AstrBotInvestigationAiClient,
    InvestigationService,
    _automatic_read_calls,
    _evidence_from_hits,
)
from nju_report.models import (
    CommunityContextAudit,
    CommunityContextDegradationReason,
    CoverageStatus,
    EvidenceItem,
    InvestigationResult,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSearchHit,
    QuestionCluster,
    ScopeAssessment,
    ScopeDecision,
)
from nju_report.reporting import (
    ReportService,
    _display_excerpt,
    _render_mail_html,
    _render_mail_text,
    _summary_payload,
    _visible_evidence,
    community_context_degradation_event_count,
    coverage_counts,
    coverage_list_order,
    format_coverage_counts,
    format_question_detail,
    report_delivery_quality_issues,
)
from nju_report.storage import ReportStorage
from nju_report.time_windows import natural_day_window


class FakeKnowledge:
    def __init__(self, hits: list[KnowledgeSearchHit]) -> None:
        self.hits = hits
        self.searches: list[str] = []
        self.greps: list[str] = []
        self.reads: list[tuple[str, str]] = []

    async def search(self, query: str, *, limit: int = 8):
        self.searches.append(query)
        return self.hits[:limit]

    async def grep(self, text: str, *, limit: int = 20):
        self.greps.append(text)
        return self.hits[:limit] if text in "校园卡补办" else []

    async def read_document(self, namespace: str, document_id: str):
        self.reads.append((namespace, document_id))
        for hit in self.hits:
            if hit.chunk.namespace == namespace and hit.chunk.document_id == document_id:
                return KnowledgeDocument(
                    namespace=namespace,
                    yuque_id=document_id,
                    title=hit.chunk.title,
                    slug=document_id,
                    url=hit.chunk.source_url,
                    updated_at=hit.chunk.updated_at,
                    body=hit.chunk.content,
                    body_hash=hit.chunk.content_hash,
                )
        return None


class FakeAi:
    def __init__(self, *, fail: bool = False, status: str = "PARTIAL") -> None:
        self.fail = fail
        self.status = status

    async def next_step(self, cluster, evidence, tool_history, *, must_finish):
        del must_finish
        if self.fail:
            raise RuntimeError("provider down")
        searches = [item for item in tool_history if item.get("tool") == "search"]
        greps = [item for item in tool_history if item.get("tool") == "grep"]
        reads = [item for item in tool_history if item.get("tool") == "read"]
        if len(searches) < 2 or not greps:
            return {
                "action": "tools",
                "tool_calls": [
                    {"tool": "search", "query": cluster.canonical_question},
                    {"tool": "search", "query": f"{cluster.category} 办理说明"},
                    {"tool": "grep", "text": "校园卡"},
                ],
            }
        if evidence and not reads:
            return {
                "action": "tools",
                "tool_calls": [
                    {
                        "tool": "read",
                        "namespace": evidence[0].namespace,
                        "document_id": evidence[0].document_id,
                        "offset": 0,
                        "focus": "校园卡",
                    }
                ],
            }
        no_evidence = self.status == "NO_USABLE_EVIDENCE" or not evidence
        final_status = "NO_USABLE_EVIDENCE" if no_evidence else self.status
        return {
            "action": "final",
            "status": final_status,
            "summary": (
                "候选资料不能支持回答。"
                if no_evidence
                else "已有挂失说明，但补办地点不完整。"
            ),
            "missing_information": "缺少正式资料。" if no_evidence else "缺少各校区补办地点。",
            "recommendation": (
                "核实后补充知识库。"
                if no_evidence
                else "补充各校区服务点和开放时间。"
            ),
            "evidence_indices": [] if no_evidence else [1],
            "flags": ["TIME_SENSITIVE"],
        }


def test_real_agent_adapter_receives_tools_evidence_and_history() -> None:
    class Response:
        completion_text = '{"action":"tools","tool_calls":[{"tool":"grep","text":"校园卡"}]}'

    class Context:
        prompt = ""
        system_prompt = ""

        async def llm_generate(self, **kwargs):
            self.prompt = kwargs["prompt"]
            self.system_prompt = kwargs["system_prompt"]
            return Response()

    cluster = QuestionCluster(
        question_code="20260712-Q001",
        report_date="2026-07-12",
        canonical_question="校园卡丢失后如何补办？",
        category="校园服务",
        candidate_source_keys=(),
        representative_questions=(),
        group_aliases=(),
        first_sent_at_utc=1,
        last_sent_at_utc=1,
    )
    context = Context()
    client = AstrBotInvestigationAiClient(context, provider_id="provider")

    result = asyncio.run(
        client.next_step(
            cluster,
            (),
            ({"tool": "search", "input": "校园卡补办", "result_count": 0},),
            must_finish=False,
        )
    )

    assert result["action"] == "tools"
    assert '"tool_history"' in context.prompt
    assert '"report_date": "2026-07-12"' in context.prompt
    assert "校园卡补办" in context.prompt
    assert "search" in context.system_prompt
    assert "grep" in context.system_prompt
    assert "read" in context.system_prompt
    assert "相对时间只能根据 report_date" in context.system_prompt


def test_investigation_uses_search_and_grep_and_persists_evidence(tmp_path: Path) -> None:
    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        knowledge = FakeKnowledge([hit])
        service = InvestigationService(config, storage, knowledge, FakeAi())  # type: ignore[arg-type]

        result = await service.investigate(cluster)

        assert result.status is CoverageStatus.PARTIAL
        assert result.evidence[0].title == "校园卡补办"
        assert len(knowledge.searches) >= 2
        assert knowledge.greps
        assert knowledge.reads == [("qc19gt/guide", "card")]
        assert result.queries[0].startswith("search:")
        assert any(item.startswith("grep:") for item in result.queries)
        assert any(item.startswith("read:") for item in result.queries)
        assert storage.latest_investigation(cluster.question_code) == result
        storage.close()

    asyncio.run(run())


def test_twenty_rounds_then_auto_reads_top_candidate_before_finalization(
    tmp_path: Path,
) -> None:
    class LongRunningAi(FakeAi):
        def __init__(self) -> None:
            super().__init__()
            self.rounds = 0

        async def next_step(self, cluster, evidence, tool_history, *, must_finish):
            self.rounds += 1
            if self.rounds == 1:
                return {
                    "action": "tools",
                    "tool_calls": [
                        {"tool": "search", "query": "校园卡补办"},
                        {"tool": "search", "query": "校园卡挂失办理"},
                        {"tool": "grep", "text": "校园卡"},
                    ],
                }
            if self.rounds <= 20:
                assert must_finish is False
                return {
                    "action": "tools",
                    "tool_calls": [
                        {"tool": "search", "query": f"补办流程 改写 {self.rounds}"}
                    ],
                }
            assert self.rounds == 21
            assert must_finish is True
            automatic_reads = [
                item
                for item in tool_history
                if item.get("tool") == "read" and item.get("automatic") is True
            ]
            assert len(automatic_reads) == 1
            assert "继续阅读" in evidence[0].excerpt
            return await super().next_step(
                cluster,
                evidence,
                tool_history,
                must_finish=must_finish,
            )

    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        ai = LongRunningAi()
        top_chunk = replace(
            hit.chunk,
            chunk_id="qc19gt/guide:top:0",
            document_id="top",
            title="校园卡服务总览",
            content="校园卡丢失后应先挂失，并按校区查询对应服务点。",
            content_hash="top",
        )
        top_hit = replace(hit, chunk=top_chunk, score=0.99)
        knowledge = FakeKnowledge([hit, top_hit])
        service = InvestigationService(  # type: ignore[arg-type]
            config,
            storage,
            knowledge,
            ai,
        )

        result = await service.investigate(cluster)

        assert ai.rounds == 21
        assert knowledge.reads == [("qc19gt/guide", "top")]
        assert any(item.startswith("read:auto:") for item in result.queries)
        assert result.status is CoverageStatus.PARTIAL
        storage.close()

    asyncio.run(run())


def test_twenty_rounds_and_two_final_failures_are_saved_as_error(
    tmp_path: Path,
) -> None:
    class NeverFinalAi:
        def __init__(self) -> None:
            self.rounds = 0

        async def next_step(self, cluster, evidence, tool_history, *, must_finish):
            del evidence, tool_history
            self.rounds += 1
            if self.rounds == 1:
                return {
                    "action": "tools",
                    "tool_calls": [
                        {"tool": "search", "query": cluster.canonical_question},
                        {"tool": "grep", "text": "校园卡"},
                    ],
                }
            return {
                "action": "tools",
                "tool_calls": [
                    {"tool": "search", "query": f"继续检索 {self.rounds} {must_finish}"}
                ],
            }

    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        ai = NeverFinalAi()
        knowledge = FakeKnowledge([hit])
        service = InvestigationService(config, storage, knowledge, ai)  # type: ignore[arg-type]

        result = await service.investigate(cluster)

        assert ai.rounds == 44
        assert knowledge.reads == [
            ("qc19gt/guide", "card"),
            ("qc19gt/guide", "card"),
        ]
        assert result.status is CoverageStatus.ERROR
        assert "20 轮" in result.error_summary
        assert "最终结论" in result.error_summary
        assert result.attempts == 2
        assert len(result.retry_errors) == 2
        assert storage.latest_investigation(cluster.question_code) == result
        storage.close()

    asyncio.run(run())


def test_failed_investigation_is_automatically_retried_and_recovers(tmp_path: Path) -> None:
    class TransientFailureAi(FakeAi):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def next_step(self, cluster, evidence, tool_history, *, must_finish):
            if not self.failed:
                self.failed = True
                raise RuntimeError("temporary provider failure")
            return await super().next_step(
                cluster,
                evidence,
                tool_history,
                must_finish=must_finish,
            )

    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        service = InvestigationService(  # type: ignore[arg-type]
            config,
            storage,
            FakeKnowledge([hit]),
            TransientFailureAi(),
        )

        result = await service.investigate(cluster)

        assert result.status is CoverageStatus.PARTIAL
        assert result.attempts == 2
        assert result.retry_errors == ("RuntimeError: temporary provider failure",)
        assert storage.latest_investigation(cluster.question_code) == result
        summary = _summary_payload(
            cluster.report_date,
            storage.list_question_clusters(cluster.report_date),
            storage.investigations_for_date(cluster.report_date),
            screening_errors=0,
        )
        assert summary["investigation_auto_retries"] == 1
        assert summary["investigation_retry_recovered"] == 1
        assert summary["investigation_retry_failed"] == 0
        detail = format_question_detail(cluster, result, timezone_name="Asia/Shanghai")
        assert "调查自动重跑：1 次；已恢复" in detail
        assert "RuntimeError: temporary provider failure" in detail
        storage.close()

    asyncio.run(run())


def test_automatic_read_only_selects_documents_visible_to_model() -> None:
    chunk = KnowledgeChunk(
        chunk_id="visible:0",
        namespace="repo",
        document_id="visible",
        title="可见文档",
        source_url="https://www.yuque.com/repo/visible",
        updated_at="2026-07-18",
        chunk_index=0,
        content="可见正文",
        content_hash="visible",
    )
    visible_hit = KnowledgeSearchHit(
        chunk=chunk,
        score=1.0,
        keyword_score=1.0,
        vector_score=0.0,
        methods=("keyword",),
    )
    hidden_hit = replace(
        visible_hit,
        chunk=replace(
            chunk,
            chunk_id="hidden:0",
            document_id="aaa-hidden",
            title="不可见文档",
            content_hash="hidden",
        ),
    )
    evidence = _evidence_from_hits((visible_hit,))

    calls = _automatic_read_calls(
        {hidden_hit.chunk.chunk_id: hidden_hit, visible_hit.chunk.chunk_id: visible_hit},
        evidence,
    )

    assert [item["document_id"] for item in calls] == ["visible"]


def test_finalization_contract_error_is_retried_once_with_existing_evidence(
    tmp_path: Path,
) -> None:
    class FinalRetryAi(FakeAi):
        def __init__(self) -> None:
            super().__init__()
            self.rounds = 0

        async def next_step(self, cluster, evidence, tool_history, *, must_finish):
            self.rounds += 1
            if self.rounds == 1:
                return {
                    "action": "tools",
                    "tool_calls": [
                        {"tool": "search", "query": cluster.canonical_question},
                        {"tool": "search", "query": "校园卡补办说明"},
                        {"tool": "grep", "text": "校园卡"},
                    ],
                }
            if self.rounds <= 20:
                return {
                    "action": "tools",
                    "tool_calls": [
                        {"tool": "search", "query": f"继续调查 {self.rounds}"}
                    ],
                }
            assert must_finish is True
            if self.rounds == 21:
                return {"action": "tools", "tool_calls": []}
            assert any(
                item.get("tool") == "contract_error"
                and "最终结论" in item.get("message", "")
                for item in tool_history
            )
            return await super().next_step(
                cluster,
                evidence,
                tool_history,
                must_finish=must_finish,
            )

    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        ai = FinalRetryAi()
        service = InvestigationService(  # type: ignore[arg-type]
            config,
            storage,
            FakeKnowledge([hit]),
            ai,
        )

        result = await service.investigate(cluster)

        assert ai.rounds == 22
        assert result.status is CoverageStatus.PARTIAL
        storage.close()

    asyncio.run(run())


def test_no_evidence_requires_all_repositories_ready(tmp_path: Path) -> None:
    async def run(status: str) -> CoverageStatus:
        case_dir = tmp_path / status
        case_dir.mkdir()
        storage, config, cluster, _ = _prepared_case(case_dir, repository_status=status)
        service = InvestigationService(  # type: ignore[arg-type]
            config,
            storage,
            FakeKnowledge([]),
            FakeAi(),
        )
        result = await service.investigate(cluster)
        storage.close()
        return result.status

    assert asyncio.run(run("READY")) is CoverageStatus.NO_USABLE_EVIDENCE
    assert asyncio.run(run("PARTIAL")) is CoverageStatus.ERROR


def test_irrelevant_hits_can_be_classified_as_no_usable_evidence(tmp_path: Path) -> None:
    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        service = InvestigationService(  # type: ignore[arg-type]
            config,
            storage,
            FakeKnowledge([hit]),
            FakeAi(status="NO_USABLE_EVIDENCE"),
        )

        result = await service.investigate(cluster)

        assert result.status is CoverageStatus.NO_USABLE_EVIDENCE
        assert result.evidence == ()
        storage.close()

    asyncio.run(run())


def test_agent_controls_domain_queries_without_programmed_dorm_rules(tmp_path: Path) -> None:
    class DormAi(FakeAi):
        async def next_step(self, cluster, evidence, tool_history, *, must_finish):
            calls = [item.get("tool") for item in tool_history]
            if "search" not in calls:
                return {
                    "action": "tools",
                    "tool_calls": [
                        {"tool": "search", "query": "仙林宿舍 2号楼 4号楼 房型"},
                        {"tool": "grep", "text": "一组团"},
                    ],
                }
            if "read" not in calls:
                return {
                    "action": "tools",
                    "tool_calls": [
                        {
                            "tool": "read",
                            "namespace": evidence[0].namespace,
                            "document_id": evidence[0].document_id,
                            "offset": 0,
                            "focus": "2号楼",
                        }
                    ],
                }
            return await super().next_step(
                cluster,
                evidence,
                tool_history,
                must_finish=must_finish,
            )

    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        dorm_cluster = replace(
            cluster,
            canonical_question="仙林校区2栋和4栋宿舍的房型、结构和设施配置",
            category="住宿食堂",
        )
        knowledge = FakeKnowledge([hit])
        service = InvestigationService(config, storage, knowledge, DormAi())  # type: ignore[arg-type]

        result = await service.investigate(dorm_cluster)

        assert "仙林宿舍 2号楼 4号楼 房型" in knowledge.searches
        assert "一组团" in knowledge.greps
        assert knowledge.reads
        assert result.status is CoverageStatus.PARTIAL
        storage.close()

    asyncio.run(run())


def test_evidence_combines_multiple_relevant_chunks_from_same_document(tmp_path: Path) -> None:
    storage, _, _, first_hit = _prepared_case(tmp_path, repository_status="READY")
    second_chunk = replace(
        first_hit.chunk,
        chunk_id="qc19gt/guide:card:1",
        chunk_index=1,
        content="南园二舍分为若干房型，其中一部分为套间结构。",
        content_hash="second",
    )
    second_hit = replace(first_hit, chunk=second_chunk, score=0.85)

    evidence = _evidence_from_hits([first_hit, second_hit])

    assert len(evidence) == 1
    assert "校园卡丢失后" in evidence[0].excerpt
    assert "南园二舍" in evidence[0].excerpt
    assert "相关段落 2" in evidence[0].excerpt
    storage.close()


def test_evidence_deduplicates_document_titles_marked_as_copy(tmp_path: Path) -> None:
    storage, _, _, first_hit = _prepared_case(tmp_path, repository_status="READY")
    copied_chunk = replace(
        first_hit.chunk,
        chunk_id="qc19gt/guide:copy:0",
        document_id="copy",
        title="校园卡补办 副本",
        content="校园卡丢失后应当先挂失。",
        content_hash="copy",
    )

    evidence = _evidence_from_hits(
        [first_hit, replace(first_hit, chunk=copied_chunk, score=0.85)]
    )

    assert len(evidence) == 1
    storage.close()


def test_public_report_evidence_is_deduplicated_and_excerpt_is_bounded() -> None:
    first = EvidenceItem(
        namespace="qc19gt/guide",
        document_id="one",
        title="宿舍总览",
        source_url="https://example.test/one",
        updated_at="2025-08-20",
        excerpt="甲" * 500,
    )
    duplicate_copy = replace(
        first,
        document_id="two",
        title="宿舍总览 副本",
        source_url="https://example.test/two",
    )

    assert _visible_evidence((first, duplicate_copy)) == (first,)
    excerpt = _display_excerpt(first.excerpt)
    assert len(excerpt) == 420
    assert excerpt.endswith("…")


def test_model_failure_is_error_not_knowledge_gap(tmp_path: Path) -> None:
    async def run() -> None:
        storage, config, cluster, hit = _prepared_case(tmp_path, repository_status="READY")
        service = InvestigationService(  # type: ignore[arg-type]
            config,
            storage,
            FakeKnowledge([hit]),
            FakeAi(fail=True),
        )
        result = await service.investigate(cluster)
        assert result.status is CoverageStatus.ERROR
        storage.close()

    asyncio.run(run())


def test_report_versions_and_mail_delivery_are_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def run() -> None:
        storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
        window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")
        assert storage.begin_processing_window(window, run_id="report-run")
        storage.complete_processing_window(
            cluster.report_date,
            run_id="report-run",
            messages_scanned=1,
            candidates_saved=1,
            included_count=1,
            dropped_count=0,
            error_count=0,
        )
        storage.save_investigation(
            InvestigationResult(
                question_code=cluster.question_code,
                status=CoverageStatus.NO_USABLE_EVIDENCE,
                summary="暂未找到可用资料。",
                missing_information="缺少正式说明。",
                recommendation="核实后补充。",
            )
        )
        config = PluginConfig.from_mapping(
            {
                "smtp_host": "smtp.qq.com",
                "smtp_username": "sender@qq.com",
                "smtp_password": "secret",
                "mail_from": "sender@qq.com",
                "mail_recipients": ["reader@example.com"],
            }
        )
        reports = ReportService(config, storage, tmp_path / "reports")
        sent: list[str] = []

        def fake_send(
            recipient: str,
            subject: str,
            text_body: str,
            html_body: str,
            full_html: str,
            path: Path,
        ) -> None:
            del subject, path
            sent.append(recipient)
            assert "问题 1｜未找到 1｜异常 0｜部分覆盖 0｜明确回答 0" in text_body
            assert "问题：20260712-Q001｜校园卡丢失后如何补办？" in text_body
            assert "状态：知识库未找到可用信息" in text_body
            assert "回答：未发现明确回答" in text_body
            assert text_body.index("状态：") < text_body.index("回答：")
            assert "知识库调查" not in html_body
            assert "问题：</strong>20260712-Q001" in html_body
            assert "状态：</strong>" in html_body
            assert "回答：</strong>未发现明确回答" in html_body
            assert "color:#991b1b" in html_body
            assert html_body.index("状态：</strong>") < html_body.index("回答：</strong>")
            assert "知识库调查" in full_html

        monkeypatch.setattr(
            reports,
            "_send_one",
            fake_send,
        )

        first = await reports.build(cluster.report_date)
        second = await reports.build(cluster.report_date)
        first_delivery = await reports.deliver(first)
        second_delivery = await reports.deliver(second)

        assert first.report_id == second.report_id
        assert first.version == 1
        assert first_delivery.sent == 1
        assert second_delivery.skipped == 1
        assert sent == ["reader@example.com"]
        rendered = Path(first.html_path).read_text(encoding="utf-8")
        assert "校园卡丢了怎么补办" in rendered
        assert "问题表达（AI 已归纳脱敏）" in rendered
        assert "群聊回答摘要（AI 已归纳脱敏，未经核实）" in rendered
        assert "message:" not in rendered
        assert "出现 1 次" not in rendered
        assert "出现次数" not in format_question_detail(
            cluster,
            storage.latest_investigation(cluster.question_code),
            timezone_name="Asia/Shanghai",
        )
        storage.close()

    asyncio.run(run())


def test_delivery_blocks_unsafe_legacy_anchor_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def run() -> None:
        storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
        unsafe = replace(
            cluster,
            canonical_question="[提及用户] 某群名片([编号]) 还没出结果呢",
            community_context_degraded=True,
            community_context_degradation_reason=(
                CommunityContextDegradationReason.VALIDATION_UNRESOLVED
            ),
            community_context_audit=CommunityContextAudit(
                degraded_question_ids=("Q1",),
                fallback_actions=("SAFE_QUESTION_FROM_UNCOVERED_ANCHOR",),
                event_id="ctx:unsafe",
            ),
        )
        storage.save_question_clusters(cluster.report_date, [unsafe])
        config = PluginConfig.from_mapping(
            {
                "smtp_host": "smtp.qq.com",
                "smtp_username": "sender@qq.com",
                "smtp_password": "secret",
                "mail_from": "sender@qq.com",
                "mail_recipients": ["reader@example.com"],
            }
        )
        reports = ReportService(config, storage, tmp_path / "reports")
        sent: list[str] = []
        monkeypatch.setattr(
            reports,
            "_send_one",
            lambda *args, **kwargs: sent.append("sent"),
        )
        report = await reports.build(cluster.report_date)

        issues = report_delivery_quality_issues([unsafe])
        assert len(issues) == 2
        with pytest.raises(RuntimeError, match="日报质量检查未通过，禁止发送"):
            await reports.deliver(report)
        assert sent == []
        assert storage.mail_deliveries(report.report_id) == []
        storage.close()

    asyncio.run(run())


def test_delivery_blocks_context_dependent_question_title(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def run() -> None:
        storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
        unsafe = replace(
            cluster,
            canonical_question="那么接替这玩意的是什么呢",
        )
        storage.save_question_clusters(cluster.report_date, [unsafe])
        config = PluginConfig.from_mapping(
            {
                "smtp_host": "smtp.qq.com",
                "smtp_username": "sender@qq.com",
                "smtp_password": "secret",
                "mail_from": "sender@qq.com",
                "mail_recipients": ["reader@example.com"],
            }
        )
        reports = ReportService(config, storage, tmp_path / "reports")
        sent: list[str] = []
        monkeypatch.setattr(
            reports,
            "_send_one",
            lambda *args, **kwargs: sent.append("sent"),
        )
        report = await reports.build(cluster.report_date)

        issues = report_delivery_quality_issues([unsafe])
        assert issues == (f"{unsafe.question_code}: 问题标题依赖缺失上下文",)
        with pytest.raises(RuntimeError, match="日报质量检查未通过，禁止发送"):
            await reports.deliver(report)
        assert sent == []
        assert storage.mail_deliveries(report.report_id) == []
        storage.close()

    asyncio.run(run())


def test_restored_screening_title_is_counted_and_shown_in_audit(
    tmp_path: Path,
) -> None:
    storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
    corrected = replace(
        cluster,
        community_context_audit=CommunityContextAudit(
            fallback_actions=("CANONICAL_QUESTION_RESTORED_FROM_SCREENING",),
        ),
    )
    investigations = storage.investigations_for_date(cluster.report_date)

    summary = _summary_payload(
        cluster.report_date,
        [corrected],
        investigations,
        screening_errors=0,
    )
    detail = format_question_detail(
        corrected,
        investigations.get(corrected.question_code),
        timezone_name="Asia/Shanghai",
    )

    assert summary["canonical_question_restored"] == 1
    assert "社区上下文校正动作：CANONICAL_QUESTION_RESTORED_FROM_SCREENING" in detail
    storage.close()


def test_cancelling_delivery_waits_for_smtp_and_persists_before_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def run() -> None:
        storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
        config = PluginConfig.from_mapping(
            {
                "smtp_host": "smtp.qq.com",
                "smtp_username": "sender@qq.com",
                "smtp_password": "secret",
                "mail_from": "sender@qq.com",
                "mail_recipients": ["reader@example.com"],
            }
        )
        reports = ReportService(config, storage, tmp_path / "reports")
        started = threading.Event()
        release = threading.Event()
        sent: list[str] = []

        def delayed_send(
            recipient: str,
            subject: str,
            text_body: str,
            html_body: str,
            full_html: str,
            path: Path,
        ) -> None:
            del subject, text_body, html_body, full_html, path
            started.set()
            assert release.wait(timeout=5)
            sent.append(recipient)

        monkeypatch.setattr(reports, "_send_one", delayed_send)
        report = await reports.build(cluster.report_date)
        delivery_task = asyncio.create_task(reports.deliver(report))
        assert await asyncio.to_thread(started.wait, 2)

        delivery_task.cancel()
        await asyncio.sleep(0)
        assert not delivery_task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await delivery_task

        deliveries = storage.mail_deliveries(report.report_id)
        assert len(deliveries) == 1
        assert deliveries[0].status == "SENT"
        retry = await reports.deliver(report)
        assert retry.skipped == 1
        assert sent == ["reader@example.com"]
        storage.close()

    asyncio.run(run())


def test_public_counts_fold_legacy_incomplete_into_execution_error(tmp_path: Path) -> None:
    storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
    storage.save_investigation(
        InvestigationResult(
            question_code=cluster.question_code,
            status=CoverageStatus.INCOMPLETE,
            summary="旧状态",
            missing_information="无",
            recommendation="重试",
        )
    )
    clusters = storage.list_question_clusters(None)
    investigations = storage.investigations_for_date(None)

    counts = coverage_counts(clusters, investigations)

    assert counts[CoverageStatus.ERROR] == 1
    assert "程序执行异常 1" in format_coverage_counts(counts)
    assert [item.question_code for item in clusters] == [cluster.question_code]
    storage.close()


def test_community_context_degradation_has_separate_public_count(tmp_path: Path) -> None:
    storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
    degraded = replace(
        cluster,
        community_context_degraded=True,
        community_context_degradation_reason=(
            CommunityContextDegradationReason.VALIDATION_UNRESOLVED
        ),
        community_context_audit=CommunityContextAudit(
            initial_errors=("invalid Q id",),
            retry_errors=("invalid Q id",),
            degraded_question_ids=("Q1",),
            fallback_actions=("SAFE_QUESTION_FROM_UNCOVERED_ANCHOR",),
            event_id="ctx:shared-event",
        ),
    )
    storage.save_question_clusters(cluster.report_date, [degraded])
    stored = storage.list_question_clusters(cluster.report_date)
    assert stored[0].community_context_audit.event_id == "ctx:shared-event"
    clusters = [stored[0], replace(stored[0], question_code="20260712-Q002")]
    investigations = storage.investigations_for_date(cluster.report_date)
    counts = coverage_counts(clusters, investigations)

    summary = _summary_payload(
        cluster.report_date,
        clusters,
        investigations,
        screening_errors=0,
    )
    text = format_coverage_counts(
        counts,
        community_context_degraded=2,
        community_context_degradation_events=1,
    )
    mail_text = _render_mail_text(cluster.report_date, clusters, investigations)
    mail_html = _render_mail_html(cluster.report_date, clusters, investigations)
    detail = format_question_detail(
        clusters[0],
        InvestigationResult(
            question_code=cluster.question_code,
            status=CoverageStatus.NO_USABLE_EVIDENCE,
            summary="未找到可用信息。",
            missing_information="缺少正式资料。",
            recommendation="补充知识库。",
        ),
        timezone_name="Asia/Shanghai",
    )

    assert summary["community_context_degraded"] == 2
    assert summary["community_context_degradation_events"] == 1
    assert community_context_degradation_event_count(clusters) == 1
    assert summary["community_context_degradation_reasons"] == {
        "VALIDATION_UNRESOLVED": 2
    }
    assert "社区上下文降级 2（独立事件 1）" in text
    assert "社区上下文降级 2（独立事件 1）" in mail_text
    assert "社区上下文降级 2（独立事件 1）" in mail_html
    assert "VALIDATION_UNRESOLVED" in detail
    assert "SAFE_QUESTION_FROM_UNCOVERED_ANCHOR" in detail
    storage.close()


def test_public_list_order_is_missing_partial_answerable_then_error() -> None:
    statuses = [
        CoverageStatus.ERROR,
        CoverageStatus.ANSWERABLE,
        CoverageStatus.NO_USABLE_EVIDENCE,
        CoverageStatus.PARTIAL,
    ]

    ordered = sorted(statuses, key=coverage_list_order)

    assert ordered == [
        CoverageStatus.NO_USABLE_EVIDENCE,
        CoverageStatus.PARTIAL,
        CoverageStatus.ANSWERABLE,
        CoverageStatus.ERROR,
    ]
    counts = {status: 1 for status in ordered}
    text = format_coverage_counts(counts)
    assert text.index("未找到可用信息") < text.index("部分覆盖")
    assert text.index("部分覆盖") < text.index("明确回答")


def test_mail_groups_questions_in_red_yellow_green_order() -> None:
    def cluster(code: str, question: str) -> QuestionCluster:
        return QuestionCluster(
            question_code=code,
            report_date="2026-07-12",
            canonical_question=question,
            category="测试",
            candidate_source_keys=(),
            representative_questions=(question,),
            group_aliases=("测试群",),
            first_sent_at_utc=1,
            last_sent_at_utc=1,
        )

    green = cluster("20260712-Q001", "已有答案的问题")
    red = cluster("20260712-Q002", "没有资料的问题")
    yellow = cluster("20260712-Q003", "部分覆盖的问题")
    clusters = [green, red, yellow]
    investigations = {
        green.question_code: InvestigationResult(
            question_code=green.question_code,
            status=CoverageStatus.ANSWERABLE,
            summary="可回答",
            missing_information="无",
            recommendation="无",
        ),
        red.question_code: InvestigationResult(
            question_code=red.question_code,
            status=CoverageStatus.NO_USABLE_EVIDENCE,
            summary="未找到",
            missing_information="全部",
            recommendation="补充",
        ),
        yellow.question_code: InvestigationResult(
            question_code=yellow.question_code,
            status=CoverageStatus.PARTIAL,
            summary="部分资料",
            missing_information="一部分",
            recommendation="补充",
        ),
    }

    text = _render_mail_text("2026-07-12", clusters, investigations)
    rendered = _render_mail_html("2026-07-12", clusters, investigations)

    assert text.index(red.question_code) < text.index(yellow.question_code)
    assert text.index(yellow.question_code) < text.index(green.question_code)
    assert rendered.index(red.question_code) < rendered.index(yellow.question_code)
    assert rendered.index(yellow.question_code) < rendered.index(green.question_code)
    assert "color:#991b1b" in rendered
    assert "color:#854d0e" in rendered
    assert "color:#166534" in rendered


def _prepared_case(
    tmp_path: Path,
    *,
    repository_status: str,
) -> tuple[ReportStorage, PluginConfig, QuestionCluster, KnowledgeSearchHit]:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    config = PluginConfig.from_mapping({"approved_repositories": ["qc19gt/guide"]})
    body = "校园卡丢失后可以先挂失，再到服务点办理。"
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    document = KnowledgeDocument(
        namespace="qc19gt/guide",
        yuque_id="card",
        title="校园卡补办",
        slug="card",
        url="https://www.yuque.com/qc19gt/guide/card",
        updated_at="2026-07-01",
        body=body,
        body_hash=body_hash,
    )
    chunk = KnowledgeChunk(
        chunk_id="qc19gt/guide:card:0",
        namespace=document.namespace,
        document_id=document.yuque_id,
        title=document.title,
        source_url=document.url,
        updated_at=document.updated_at,
        chunk_index=0,
        content=body,
        content_hash=body_hash,
    )
    storage.replace_knowledge_document(document, [chunk])
    storage.upsert_repository("qc19gt/guide", status=repository_status)
    source_key = "message:qq:bot:m1"
    storage.upsert_scope_candidate(
        source_key=source_key,
        report_date="2026-07-12",
        initial=ScopeAssessment(
            decision=ScopeDecision.INCLUDE,
            reason="校园服务问题",
            confidence=0.98,
            canonical_question="校园卡丢失后如何补办？",
            category="校园服务/校园卡",
        ),
        original_question="校园卡丢了怎么补办",
        group_alias="南京大学迎新群",
        sent_at_utc=1_752_300_000,
    )
    candidate = storage.get_question_candidate("20260712-Q001")
    assert candidate is not None
    cluster = QuestionCluster(
        question_code=candidate.question_code,
        report_date=candidate.report_date,
        canonical_question=candidate.canonical_question,
        category=candidate.category,
        candidate_source_keys=(source_key,),
        representative_questions=(candidate.original_question,),
        group_aliases=(candidate.group_alias,),
        first_sent_at_utc=candidate.sent_at_utc,
        last_sent_at_utc=candidate.sent_at_utc,
    )
    storage.save_question_clusters(cluster.report_date, [cluster])
    hit = KnowledgeSearchHit(
        chunk=chunk,
        score=0.9,
        keyword_score=0.9,
        vector_score=0.0,
        methods=("keyword",),
    )
    return storage, config, cluster, hit
