"""Allowlisted Yuque synchronization and local hybrid retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import html
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from .config import PluginConfig
from .models import KnowledgeChunk, KnowledgeDocument, KnowledgeSearchHit
from .storage import ReportStorage

_CHUNK_SIZE = 1200
_CHUNK_OVERLAP = 180
_MAX_DOCUMENT_CHARS = 2_000_000


class KnowledgeError(RuntimeError):
    """Raised when an approved knowledge source cannot be synchronized."""


@dataclass(slots=True)
class RepositorySyncResult:
    namespace: str
    documents_seen: int = 0
    documents_changed: int = 0
    documents_unchanged: int = 0
    documents_deleted: int = 0
    chunks_written: int = 0
    embeddings_written: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class KnowledgeSyncResult:
    repositories: list[RepositorySyncResult] = field(default_factory=list)
    excluded_purged: dict[str, int] = field(default_factory=dict)

    @property
    def failure_count(self) -> int:
        return sum(len(item.failures) for item in self.repositories)


@dataclass(frozen=True, slots=True)
class KnowledgeSyncProgress:
    syncing: bool
    namespace: str = ""
    repository_index: int = 0
    repository_total: int = 0
    document_completed: int = 0
    document_total: int = 0


class YuqueClient:
    """Small async client for the subset of Yuque v2 used by the report."""

    def __init__(
        self,
        token: str,
        base_url: str,
        *,
        timeout_seconds: int = 120,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = min(timeout_seconds, 120)
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None
        self._headers = {
            "X-Auth-Token": token,
            "User-Agent": "astrbot-plugin-nju-qa-report/0.2",
        }

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def get_repository(self, namespace: str) -> dict[str, Any]:
        return _object(await self._get(f"/repos/{quote(namespace, safe='/-')}"))

    async def get_toc(self, namespace: str) -> list[dict[str, Any]]:
        value = await self._get(f"/repos/{quote(namespace, safe='/-')}/toc")
        if not isinstance(value, list):
            raise KnowledgeError("语雀 TOC 不是数组")
        return [_object(item) for item in value]

    async def get_document(self, namespace: str, slug: str) -> dict[str, Any]:
        return _object(
            await self._get(
                f"/repos/{quote(namespace, safe='/-')}/docs/{quote(slug, safe='')}",
                params={"include_content": "true"},
            )
        )

    async def _get(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        client = await self._http()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.get(path, params=params)
                if (
                    response.status_code == 429 or response.status_code >= 500
                ) and attempt < self._max_retries:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else 0.5 * (2**attempt)
                    await asyncio.sleep(min(delay, 8.0))
                    continue
                response.raise_for_status()
                payload = response.json()
                return payload.get("data", payload) if isinstance(payload, dict) else payload
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
        raise KnowledgeError(f"语雀请求失败：{type(last_error).__name__}") from last_error

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout, connect=min(15, self._timeout)),
                follow_redirects=True,
            )
        return self._client


class EmbeddingClient:
    """OpenAI-compatible embedding adapter with bounded DashScope-sized batches."""

    def __init__(self, config: PluginConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = config.embedding_api_key
        self._base_url = config.embedding_base_url.rstrip("/")
        self._model = config.embedding_model
        self._timeout = min(config.request_timeout_seconds, 120)
        self._max_retries = config.max_retries
        self._client = client
        self._owns_client = client is None

    @property
    def configured(self) -> bool:
        return bool(self._api_key and self._base_url)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def embed_one(self, text: str) -> tuple[float, ...]:
        values = await self.embed_many([text])
        return values[0] if values else ()

    async def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts or not self.configured:
            return [() for _ in texts]
        result: list[tuple[float, ...]] = []
        for start in range(0, len(texts), 10):
            batch = [item[:16000] for item in texts[start : start + 10]]
            payload = await self._post(batch)
            data = payload.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise KnowledgeError("Embedding 返回数量与请求不一致")
            ordered = sorted(
                data,
                key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
            )
            for item in ordered:
                if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                    raise KnowledgeError("Embedding 返回缺少向量")
                result.append(tuple(float(value) for value in item["embedding"]))
        return result

    async def _post(self, texts: list[str]) -> dict[str, Any]:
        client = await self._http()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    self._base_url + "/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "input": texts},
                )
                if (
                    response.status_code == 429 or response.status_code >= 500
                ) and attempt < self._max_retries:
                    await asyncio.sleep(min(0.5 * (2**attempt), 8.0))
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise KnowledgeError("Embedding 返回不是 JSON 对象")
                return payload
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, ValueError, KnowledgeError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
        raise KnowledgeError(f"Embedding 请求失败：{type(last_error).__name__}") from last_error

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client


class KnowledgeService:
    """Synchronize current policy and expose allowlist-filtered hybrid search."""

    def __init__(
        self,
        config: PluginConfig,
        storage: ReportStorage,
        *,
        yuque_client: YuqueClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._yuque = yuque_client or YuqueClient(
            config.yuque_token,
            config.yuque_api_base,
            timeout_seconds=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )
        self._embedding = embedding_client or EmbeddingClient(config)
        self._sync_lock = asyncio.Lock()
        self._progress_namespace = ""
        self._progress_repository_index = 0
        self._progress_repository_total = 0
        self._progress_document_completed = 0
        self._progress_document_total = 0

    @property
    def syncing(self) -> bool:
        return self._sync_lock.locked()

    def progress(self) -> KnowledgeSyncProgress:
        return KnowledgeSyncProgress(
            syncing=self.syncing,
            namespace=self._progress_namespace,
            repository_index=self._progress_repository_index,
            repository_total=self._progress_repository_total,
            document_completed=self._progress_document_completed,
            document_total=self._progress_document_total,
        )

    async def close(self) -> None:
        await self._yuque.close()
        await self._embedding.close()

    async def test_embedding(self) -> int:
        if not self._embedding.configured:
            raise KnowledgeError("Embedding 未配置")
        vector = await self._embedding.embed_one("南京大学校园卡补办")
        if not vector:
            raise KnowledgeError("Embedding 返回空向量")
        return len(vector)

    async def sync_all(self) -> KnowledgeSyncResult:
        if not self._config.yuque_token:
            raise KnowledgeError("未配置语雀 Token")
        if not self._config.approved_repositories:
            raise KnowledgeError("允许仓库列表为空")
        async with self._sync_lock:
            result = KnowledgeSyncResult()
            await asyncio.to_thread(self._apply_exclusions, result)
            namespaces = self._allowed_namespaces()
            self._progress_repository_total = len(namespaces)
            for index, namespace in enumerate(namespaces, start=1):
                self._progress_namespace = namespace
                self._progress_repository_index = index
                self._progress_document_completed = 0
                self._progress_document_total = 0
                result.repositories.append(await self._sync_repository(namespace))
            return result

    async def search(self, query: str, *, limit: int = 8) -> list[KnowledgeSearchHit]:
        query = " ".join(query.split()).strip()
        if not query or limit < 1:
            return []
        chunks = await asyncio.to_thread(
            self._storage.knowledge_chunks,
            self._allowed_namespaces(),
        )
        if not chunks:
            return []
        query_terms = _search_terms(query)
        query_vector: tuple[float, ...] = ()
        if self._config.enable_vector_search and self._embedding.configured:
            try:
                query_vector = await self._embedding.embed_one(query)
            except KnowledgeError:
                query_vector = ()
        hits: list[KnowledgeSearchHit] = []
        for chunk in chunks:
            keyword_score = _keyword_score(
                query_terms, _search_terms(chunk.title + " " + chunk.content)
            )
            vector_score = _cosine(query_vector, chunk.embedding)
            if keyword_score <= 0 and vector_score <= 0:
                continue
            methods: list[str] = []
            if keyword_score > 0:
                methods.append("keyword")
            if vector_score > 0:
                methods.append("vector")
            combined = max(keyword_score, vector_score) + min(keyword_score, vector_score) * 0.25
            hits.append(
                KnowledgeSearchHit(
                    chunk=chunk,
                    score=min(1.0, combined),
                    keyword_score=keyword_score,
                    vector_score=vector_score,
                    methods=tuple(methods),
                )
            )
        hits.sort(key=lambda item: (-item.score, item.chunk.namespace, item.chunk.chunk_id))
        return hits[:limit]

    async def grep(self, text: str, *, limit: int = 20) -> list[KnowledgeSearchHit]:
        needle = text.strip().casefold()
        if not needle:
            return []
        chunks = await asyncio.to_thread(
            self._storage.knowledge_chunks,
            self._allowed_namespaces(),
        )
        result: list[KnowledgeSearchHit] = []
        for chunk in chunks:
            if needle not in (chunk.title + "\n" + chunk.content).casefold():
                continue
            result.append(
                KnowledgeSearchHit(
                    chunk=chunk,
                    score=1.0,
                    keyword_score=1.0,
                    vector_score=0.0,
                    methods=("grep",),
                )
            )
            if len(result) >= limit:
                break
        return result

    async def read_document(
        self,
        namespace: str,
        document_id: str,
    ) -> KnowledgeDocument | None:
        """Read a full cached document, limited to the current repository allowlist."""

        namespace = namespace.strip()
        document_id = document_id.strip()
        if not namespace or not document_id or namespace not in self._allowed_namespaces():
            return None
        return await asyncio.to_thread(
            self._storage.knowledge_document,
            namespace,
            document_id,
        )

    def _allowed_namespaces(self) -> tuple[str, ...]:
        excluded = {item.namespace for item in self._config.excluded_repositories}
        return tuple(item for item in self._config.approved_repositories if item not in excluded)

    def _apply_exclusions(self, result: KnowledgeSyncResult) -> None:
        for exclusion in self._config.excluded_repositories:
            namespace = exclusion.namespace
            reason = exclusion.reason
            self._storage.upsert_repository(
                namespace,
                status="EXCLUDED",
                excluded_reason=reason,
            )
            if self._config.purge_excluded_repository_data:
                result.excluded_purged[namespace] = self._storage.purge_knowledge_repository(
                    namespace
                )

    async def _sync_repository(self, namespace: str) -> RepositorySyncResult:
        result = RepositorySyncResult(namespace=namespace)
        try:
            repository = await self._yuque.get_repository(namespace)
            display_name = _text(repository.get("name")) or namespace
            self._storage.upsert_repository(
                namespace,
                display_name=display_name,
                status="SYNCING",
            )
            toc = await self._yuque.get_toc(namespace)
            document_nodes = [
                node
                for node in toc
                if _text(node.get("type")).upper() in {"", "DOC", "SHEET"}
                and _text(node.get("id"))
                and (_text(node.get("url")) or _text(node.get("slug")))
            ]
            self._progress_document_total = len(document_nodes)
            seen: set[str] = set()
            for node in document_nodes:
                node_type = _text(node.get("type")).upper()
                if node_type and node_type not in {"DOC", "SHEET"}:
                    continue
                yuque_id = _text(node.get("id"))
                slug = _text(node.get("url")) or _text(node.get("slug"))
                if not yuque_id or not slug:
                    continue
                seen.add(yuque_id)
                result.documents_seen += 1
                try:
                    detail = await self._yuque.get_document(namespace, slug)
                    document = _knowledge_document(namespace, yuque_id, slug, node, detail)
                    old_hash = await asyncio.to_thread(
                        self._storage.knowledge_document_hash,
                        namespace,
                        yuque_id,
                    )
                    if old_hash == document.body_hash:
                        await asyncio.to_thread(
                            self._storage.replace_knowledge_document,
                            document,
                            [],
                        )
                        result.documents_unchanged += 1
                        continue
                    chunks = _split_document(document)
                    if self._embedding.configured and chunks:
                        vectors = await self._embedding.embed_many(
                            [chunk.title + "\n" + chunk.content for chunk in chunks]
                        )
                        chunks = [
                            KnowledgeChunk(
                                chunk_id=chunk.chunk_id,
                                namespace=chunk.namespace,
                                document_id=chunk.document_id,
                                title=chunk.title,
                                source_url=chunk.source_url,
                                updated_at=chunk.updated_at,
                                chunk_index=chunk.chunk_index,
                                content=chunk.content,
                                content_hash=chunk.content_hash,
                                embedding=vector,
                            )
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ]
                        result.embeddings_written += sum(bool(item.embedding) for item in chunks)
                    await asyncio.to_thread(
                        self._storage.replace_knowledge_document,
                        document,
                        chunks,
                    )
                    result.documents_changed += 1
                    result.chunks_written += len(chunks)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    result.failures.append(f"{yuque_id}:{type(exc).__name__}")
                finally:
                    self._progress_document_completed += 1
            result.documents_deleted = await asyncio.to_thread(
                self._storage.delete_missing_knowledge_documents,
                namespace,
                seen,
            )
            status = "READY" if not result.failures else "PARTIAL"
            self._storage.upsert_repository(
                namespace,
                display_name=display_name,
                status=status,
                last_error="; ".join(result.failures[:10]),
                synced_at_utc=int(time.time()),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result.failures.append(type(exc).__name__)
            await asyncio.to_thread(
                self._storage.upsert_repository,
                namespace,
                status="ERROR",
                last_error=type(exc).__name__,
            )
        return result


def _knowledge_document(
    namespace: str,
    yuque_id: str,
    slug: str,
    node: dict[str, Any],
    detail: dict[str, Any],
) -> KnowledgeDocument:
    title = _text(detail.get("title")) or _text(node.get("title")) or slug
    body = _clean_body(_text(detail.get("body")) or _text(detail.get("content")))
    if len(body) > _MAX_DOCUMENT_CHARS:
        raise KnowledgeError("语雀正文超过单文档安全上限")
    actual_slug = _text(detail.get("slug")) or slug
    url = _text(detail.get("url")) or f"https://www.yuque.com/{namespace}/{actual_slug}"
    return KnowledgeDocument(
        namespace=namespace,
        yuque_id=yuque_id,
        title=title,
        slug=actual_slug,
        url=url,
        updated_at=_text(detail.get("updated_at")),
        body=body,
        body_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def _split_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    blocks = [
        item.strip()
        for item in re.split(r"\n\s*\n|(?=^#{1,6}\s)", document.body, flags=re.MULTILINE)
        if item.strip()
    ]
    pieces: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > _CHUNK_SIZE:
            if current:
                pieces.append(current)
                current = ""
            step = _CHUNK_SIZE - _CHUNK_OVERLAP
            for start in range(0, len(block), step):
                piece = block[start : start + _CHUNK_SIZE].strip()
                if piece:
                    pieces.append(piece)
            continue
        combined = f"{current}\n\n{block}".strip() if current else block
        if len(combined) <= _CHUNK_SIZE:
            current = combined
        else:
            pieces.append(current)
            current = block
    if current:
        pieces.append(current)
    result: list[KnowledgeChunk] = []
    for index, content in enumerate(pieces):
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunk_id = f"{document.namespace}:{document.yuque_id}:{index}:{content_hash[:12]}"
        result.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                namespace=document.namespace,
                document_id=document.yuque_id,
                title=document.title,
                source_url=document.url,
                updated_at=document.updated_at,
                chunk_index=index,
                content=content,
                content_hash=content_hash,
            )
        )
    return result


def _clean_body(value: str) -> str:
    value = re.sub(r"<script\b[^>]*>.*?</script>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<style\b[^>]*>.*?</style>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"!\[[^]]*]\([^)]+\)", "", value)
    value = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"\1 (\2)", value)
    value = re.sub(r"<[^>]+>", "\n", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _search_terms(text: str) -> set[str]:
    normalized = text.casefold()
    words = set(re.findall(r"[a-z0-9_.-]{2,}", normalized))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    words.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    return {item for item in words if item}


def _keyword_score(query: set[str], content: set[str]) -> float:
    if not query or not content:
        return 0.0
    overlap = len(query & content)
    if not overlap:
        return 0.0
    return min(1.0, overlap / max(1, len(query)))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, min(1.0, similarity))


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise KnowledgeError("语雀 API 返回不是对象")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value).strip()
    return ""
