from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

from nju_report.config import PluginConfig
from nju_report.investigation import (
    InvestigationService,
    _evidence_from_hits,
    _grep_terms,
    _queries_for,
)
from nju_report.models import (
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
    _visible_evidence,
    coverage_counts,
    format_coverage_counts,
    format_question_detail,
)
from nju_report.storage import ReportStorage


class FakeKnowledge:
    def __init__(self, hits: list[KnowledgeSearchHit]) -> None:
        self.hits = hits
        self.searches: list[str] = []
        self.greps: list[str] = []

    async def search(self, query: str, *, limit: int = 8):
        self.searches.append(query)
        return self.hits[:limit]

    async def grep(self, text: str, *, limit: int = 20):
        self.greps.append(text)
        return self.hits[:limit] if text in "校园卡补办" else []


class FakeAi:
    def __init__(self, *, fail: bool = False, status: str = "PARTIAL") -> None:
        self.fail = fail
        self.status = status

    async def assess(self, cluster, evidence):
        if self.fail:
            raise RuntimeError("provider down")
        no_evidence = self.status == "NO_USABLE_EVIDENCE"
        return {
            "status": self.status,
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
        assert storage.latest_investigation(cluster.question_code) == result
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


def test_dorm_query_expands_aliases_and_specific_grep_terms(tmp_path: Path) -> None:
    storage, _, cluster, _ = _prepared_case(tmp_path, repository_status="READY")
    dorm_cluster = replace(
        cluster,
        canonical_question="南京大学南园二舍的宿舍结构是怎样的，是否有套间？",
        category="住宿食堂",
    )

    queries = _queries_for(dorm_cluster)
    terms = _grep_terms(dorm_cluster.canonical_question)

    assert any("南二" in item for item in queries)
    assert "南园二舍" in terms
    assert "南二" in terms
    assert "套间" in terms
    storage.close()


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
