from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from nju_report.config import PluginConfig
from nju_report.investigation import InvestigationService
from nju_report.models import (
    CoverageStatus,
    InvestigationResult,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSearchHit,
    QuestionCluster,
    ScopeAssessment,
    ScopeDecision,
)
from nju_report.reporting import ReportService
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
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def assess(self, cluster, evidence):
        if self.fail:
            raise RuntimeError("provider down")
        return {
            "status": "PARTIAL",
            "summary": "已有挂失说明，但补办地点不完整。",
            "missing_information": "缺少各校区补办地点。",
            "recommendation": "补充各校区服务点和开放时间。",
            "evidence_indices": [1],
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
    assert asyncio.run(run("PARTIAL")) is CoverageStatus.INCOMPLETE


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
        monkeypatch.setattr(
            reports,
            "_send_one",
            lambda recipient, subject, body, path: sent.append(recipient),
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
        assert "message:" not in rendered
        storage.close()

    asyncio.run(run())


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
