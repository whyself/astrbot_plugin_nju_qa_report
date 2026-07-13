from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from nju_report.config import PluginConfig
from nju_report.knowledge import KnowledgeService
from nju_report.models import KnowledgeChunk, KnowledgeDocument
from nju_report.storage import ReportStorage


class FakeYuqueClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def close(self) -> None:
        return None

    async def get_repository(self, namespace: str):
        self.calls.append(f"repo:{namespace}")
        return {"name": "新生指南"}

    async def get_toc(self, namespace: str):
        self.calls.append(f"toc:{namespace}")
        return [{"id": 7, "type": "DOC", "url": "campus-card", "title": "校园卡"}]

    async def get_document(self, namespace: str, slug: str):
        self.calls.append(f"doc:{namespace}:{slug}")
        return {
            "id": 7,
            "slug": slug,
            "title": "校园卡补办",
            "url": f"https://www.yuque.com/{namespace}/{slug}",
            "updated_at": "2026-07-01T00:00:00Z",
            "body": "# 校园卡补办\n\n校园卡丢失后，先在信息门户挂失，再前往服务点补办。",
        }


class FakeEmbeddingClient:
    configured = True

    async def close(self) -> None:
        return None

    async def embed_one(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0) if "校园卡" in text else (0.0, 1.0)

    async def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [await self.embed_one(text) for text in texts]


def test_allowlisted_sync_purges_excluded_and_builds_hybrid_search(tmp_path: Path) -> None:
    asyncio.run(_sync_and_search_case(tmp_path))


async def _sync_and_search_case(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    excluded_body = "不应进入检索"
    excluded = KnowledgeDocument(
        namespace="qc19gt/ogaye8",
        yuque_id="old",
        title="已排除问答",
        slug="old",
        url="https://www.yuque.com/qc19gt/ogaye8/old",
        updated_at="",
        body=excluded_body,
        body_hash=hashlib.sha256(excluded_body.encode()).hexdigest(),
    )
    storage.replace_knowledge_document(
        excluded,
        [
            KnowledgeChunk(
                chunk_id="excluded:old:0",
                namespace=excluded.namespace,
                document_id=excluded.yuque_id,
                title=excluded.title,
                source_url=excluded.url,
                updated_at="",
                chunk_index=0,
                content=excluded_body,
                content_hash=excluded.body_hash,
            )
        ],
    )
    config = PluginConfig.from_mapping(
        {
            "yuque_token": "secret",
            "approved_repositories": ["qc19gt/guide"],
            "embedding_api_key": "secret",
            "embedding_base_url": "https://embedding.test/v1",
            "embedding_model": "test-model",
        }
    )
    yuque = FakeYuqueClient()
    service = KnowledgeService(
        config,
        storage,
        yuque_client=yuque,
        embedding_client=FakeEmbeddingClient(),
    )

    first = await service.sync_all()
    second = await service.sync_all()
    hits = await service.search("校园卡怎么补办")
    document = await service.read_document("qc19gt/guide", "7")
    blocked_document = await service.read_document("qc19gt/ogaye8", "old")

    assert first.excluded_purged == {"qc19gt/ogaye8": 1}
    assert first.repositories[0].documents_changed == 1
    assert first.repositories[0].embeddings_written >= 1
    assert second.repositories[0].documents_unchanged == 1
    assert hits
    assert hits[0].chunk.namespace == "qc19gt/guide"
    assert "vector" in hits[0].methods
    assert document is not None and "校园卡丢失后" in document.body
    assert blocked_document is None
    assert all("ogaye8" not in call for call in yuque.calls)
    assert storage.knowledge_counts()[0] == 1
    records = {str(item["namespace"]): item for item in storage.repository_records()}
    assert records["qc19gt/guide"]["status"] == "READY"
    assert records["qc19gt/ogaye8"]["status"] == "EXCLUDED"
    progress = service.progress()
    assert progress.syncing is False
    assert progress.repository_index == progress.repository_total == 1
    assert progress.document_completed == progress.document_total == 1
    await service.close()
    storage.close()
