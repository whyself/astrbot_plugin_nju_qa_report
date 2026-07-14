"""SQLite persistence for captured messages and processing audit records."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import (
    CommunityAnswer,
    CoverageStatus,
    EvidenceItem,
    InvestigationResult,
    KnowledgeChunk,
    KnowledgeDocument,
    MailDelivery,
    ProcessingWindowRecord,
    QuestionCandidate,
    QuestionCluster,
    ReportArtifact,
    ScheduledReportRun,
    ScopeAssessment,
    ScopeResolution,
    ScreeningVersion,
    StoredMessage,
)
from .time_windows import TimeWindow

_SCHEMA_VERSION = 8


class StorageError(RuntimeError):
    """Raised when the local persistent store cannot complete an operation."""


class ReportStorage:
    """One authoritative SQLite store for the report plugin."""

    def __init__(self, database_path: Path, *, busy_timeout_ms: int = 5000) -> None:
        self.database_path = Path(database_path)
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    @property
    def initialized(self) -> bool:
        return self._connection is not None

    def initialize(self) -> None:
        """Open the database, configure safety pragmas, and apply migrations."""

        with self._lock:
            if self._connection is not None:
                return
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.database_path,
                timeout=self.busy_timeout_ms / 1000,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            try:
                connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA journal_mode = WAL")
                self._connection = connection
                self._apply_migrations()
            except Exception:
                connection.close()
                self._connection = None
                raise

    def close(self) -> None:
        """Close the database; repeated calls are safe."""

        with self._lock:
            if self._connection is None:
                return
            self._connection.close()
            self._connection = None

    def pragma(self, name: str) -> str | int | None:
        """Expose selected pragma values for health checks and tests."""

        if name not in {"journal_mode", "foreign_keys", "busy_timeout"}:
            raise ValueError("不允许查询该 SQLite PRAGMA")
        with self._lock:
            row = self._conn.execute(f"PRAGMA {name}").fetchone()
            return row[0] if row else None

    def insert_message(self, message: StoredMessage) -> bool:
        """Insert one message; return ``False`` when it is a duplicate event."""

        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (
                    platform_id,
                    bot_self_id,
                    external_message_id,
                    message_fingerprint,
                    session_id,
                    group_id,
                    group_alias,
                    sender_id,
                    sender_name,
                    sent_at_utc,
                    text,
                    outline,
                    reply_to_message_id,
                    analyzable,
                    captured_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(platform_id, bot_self_id, external_message_id) DO NOTHING
                """,
                (
                    message.platform_id,
                    message.bot_self_id,
                    message.external_message_id,
                    message.message_fingerprint,
                    message.session_id,
                    message.group_id,
                    message.group_alias,
                    message.sender_id,
                    message.sender_name,
                    message.sent_at_utc,
                    message.text,
                    message.outline,
                    message.reply_to_message_id,
                    int(message.analyzable),
                    int(time.time()),
                ),
            )
            return cursor.rowcount == 1

    def insert_messages(self, messages: list[StoredMessage]) -> tuple[int, int]:
        """Bulk insert prepared messages and return ``(inserted, duplicates)``."""

        if not messages:
            return 0, 0
        rows = [
            (
                message.platform_id,
                message.bot_self_id,
                message.external_message_id,
                message.message_fingerprint,
                message.session_id,
                message.group_id,
                message.group_alias,
                message.sender_id,
                message.sender_name,
                message.sent_at_utc,
                message.text,
                message.outline,
                message.reply_to_message_id,
                int(message.analyzable),
                int(time.time()),
            )
            for message in messages
        ]
        with self._transaction() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT INTO messages (
                    platform_id,
                    bot_self_id,
                    external_message_id,
                    message_fingerprint,
                    session_id,
                    group_id,
                    group_alias,
                    sender_id,
                    sender_name,
                    sent_at_utc,
                    text,
                    outline,
                    reply_to_message_id,
                    analyzable,
                    captured_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(platform_id, bot_self_id, external_message_id) DO NOTHING
                """,
                rows,
            )
            inserted = connection.total_changes - before
        return inserted, len(messages) - inserted

    def message_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            return int(row[0])

    def latest_message_timestamp(self) -> int | None:
        with self._lock:
            row = self._conn.execute("SELECT MAX(sent_at_utc) FROM messages").fetchone()
            if not row or row[0] is None:
                return None
            return int(row[0])

    def question_candidate_count(self, *, report_date: str | None = None) -> int:
        """Return the number of retained scope-screening records."""

        with self._lock:
            if report_date is None:
                row = self._conn.execute("SELECT COUNT(*) FROM question_candidates").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM question_candidates WHERE report_date = ?",
                    (report_date,),
                ).fetchone()
            return int(row[0])

    def message_local_dates(self, timezone_name: str) -> list[str]:
        """List local dates that currently have captured messages."""

        local_timezone = ZoneInfo(timezone_name)
        dates: set[str] = set()
        with self._lock:
            cursor = self._conn.execute("SELECT sent_at_utc FROM messages ORDER BY sent_at_utc ASC")
            while rows := cursor.fetchmany(1000):
                for row in rows:
                    dates.add(
                        datetime.fromtimestamp(int(row[0]), tz=local_timezone).date().isoformat()
                    )
        return sorted(dates)

    def delete_expired_messages(self, cutoff_utc: int) -> int:
        """Delete raw messages older than the cutoff unless a live window needs them."""

        with self._transaction() as connection:
            protected = connection.execute(
                """
                SELECT MIN(start_utc)
                FROM processing_windows
                WHERE status NOT IN ('COMPLETED', 'FAILED_PERMANENT')
                """
            ).fetchone()
            effective_cutoff = int(cutoff_utc)
            if protected and protected[0] is not None:
                effective_cutoff = min(effective_cutoff, int(protected[0]))
            cursor = connection.execute(
                "DELETE FROM messages WHERE sent_at_utc < ?",
                (effective_cutoff,),
            )
            return max(0, cursor.rowcount)

    def upsert_repository(
        self,
        namespace: str,
        *,
        display_name: str = "",
        status: str,
        excluded_reason: str = "",
        last_error: str = "",
        synced_at_utc: int | None = None,
    ) -> None:
        """Persist repository policy and synchronization state."""

        now = int(time.time())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO repositories (
                    namespace, display_name, status, excluded_reason,
                    last_error, synced_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    display_name = CASE
                        WHEN excluded.display_name = '' THEN repositories.display_name
                        ELSE excluded.display_name
                    END,
                    status = excluded.status,
                    excluded_reason = excluded.excluded_reason,
                    last_error = excluded.last_error,
                    synced_at_utc = COALESCE(excluded.synced_at_utc, repositories.synced_at_utc),
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    namespace,
                    display_name,
                    status,
                    excluded_reason,
                    last_error[:1000],
                    synced_at_utc,
                    now,
                ),
            )

    def repository_records(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM repositories ORDER BY namespace").fetchall()
        return [dict(row) for row in rows]

    def replace_knowledge_document(
        self,
        document: KnowledgeDocument,
        chunks: list[KnowledgeChunk],
    ) -> bool:
        """Upsert one document and replace chunks only when its body changed."""

        now = int(time.time())
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT id, body_hash FROM knowledge_documents
                WHERE namespace = ? AND yuque_id = ?
                """,
                (document.namespace, document.yuque_id),
            ).fetchone()
            changed = existing is None or str(existing["body_hash"]) != document.body_hash
            connection.execute(
                """
                INSERT INTO knowledge_documents (
                    namespace, yuque_id, title, slug, url, updated_at,
                    body, body_hash, synced_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, yuque_id) DO UPDATE SET
                    title = excluded.title,
                    slug = excluded.slug,
                    url = excluded.url,
                    updated_at = excluded.updated_at,
                    body = excluded.body,
                    body_hash = excluded.body_hash,
                    synced_at_utc = excluded.synced_at_utc
                """,
                (
                    document.namespace,
                    document.yuque_id,
                    document.title,
                    document.slug,
                    document.url,
                    document.updated_at,
                    document.body,
                    document.body_hash,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT id FROM knowledge_documents
                WHERE namespace = ? AND yuque_id = ?
                """,
                (document.namespace, document.yuque_id),
            ).fetchone()
            if row is None:
                raise StorageError("无法保存语雀文档")
            document_id = int(row[0])
            if changed:
                connection.execute(
                    "DELETE FROM knowledge_chunks WHERE document_row_id = ?",
                    (document_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO knowledge_chunks (
                        chunk_id, document_row_id, chunk_index, content,
                        content_hash, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            document_id,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.content_hash,
                            json.dumps(chunk.embedding) if chunk.embedding else "",
                        )
                        for chunk in chunks
                    ],
                )
            return changed

    def knowledge_document_hash(self, namespace: str, yuque_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT body_hash FROM knowledge_documents
                WHERE namespace = ? AND yuque_id = ?
                """,
                (namespace, yuque_id),
            ).fetchone()
        return str(row[0]) if row is not None else ""

    def knowledge_document(self, namespace: str, yuque_id: str) -> KnowledgeDocument | None:
        """Read one locally cached knowledge document by its stable Yuque ID."""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT namespace, yuque_id, title, slug, url, updated_at, body, body_hash
                FROM knowledge_documents
                WHERE namespace = ? AND yuque_id = ?
                """,
                (namespace.strip(), yuque_id.strip()),
            ).fetchone()
        if row is None:
            return None
        return KnowledgeDocument(
            namespace=str(row["namespace"]),
            yuque_id=str(row["yuque_id"]),
            title=str(row["title"]),
            slug=str(row["slug"]),
            url=str(row["url"]),
            updated_at=str(row["updated_at"]),
            body=str(row["body"]),
            body_hash=str(row["body_hash"]),
        )

    def delete_missing_knowledge_documents(self, namespace: str, seen_ids: set[str]) -> int:
        """Delete local documents no longer present in one approved repository."""

        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT id, yuque_id FROM knowledge_documents WHERE namespace = ?",
                (namespace,),
            ).fetchall()
            doomed = [int(row["id"]) for row in rows if str(row["yuque_id"]) not in seen_ids]
            if doomed:
                connection.executemany(
                    "DELETE FROM knowledge_documents WHERE id = ?",
                    [(item,) for item in doomed],
                )
            return len(doomed)

    def purge_knowledge_repository(self, namespace: str) -> int:
        """Remove all locally stored bodies and chunks for one repository."""

        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM knowledge_documents WHERE namespace = ?",
                (namespace,),
            )
            return max(0, cursor.rowcount)

    def knowledge_chunks(self, allowed_namespaces: tuple[str, ...]) -> list[KnowledgeChunk]:
        """Return searchable chunks, always filtered by the current allowlist."""

        if not allowed_namespaces:
            return []
        placeholders = ",".join("?" for _ in allowed_namespaces)
        sql = f"""
            SELECT
                c.chunk_id, c.chunk_index, c.content, c.content_hash,
                c.embedding_json, d.namespace, d.yuque_id, d.title,
                d.url, d.updated_at
            FROM knowledge_chunks c
            JOIN knowledge_documents d ON d.id = c.document_row_id
            WHERE d.namespace IN ({placeholders})
            ORDER BY d.namespace, d.id, c.chunk_index
        """  # noqa: S608
        with self._lock:
            rows = self._conn.execute(sql, allowed_namespaces).fetchall()
        result: list[KnowledgeChunk] = []
        for row in rows:
            embedding: tuple[float, ...] = ()
            if row["embedding_json"]:
                try:
                    values = json.loads(row["embedding_json"])
                    embedding = tuple(float(item) for item in values)
                except (TypeError, ValueError, json.JSONDecodeError):
                    embedding = ()
            result.append(
                KnowledgeChunk(
                    chunk_id=row["chunk_id"],
                    namespace=row["namespace"],
                    document_id=row["yuque_id"],
                    title=row["title"],
                    source_url=row["url"],
                    updated_at=row["updated_at"],
                    chunk_index=int(row["chunk_index"]),
                    content=row["content"],
                    content_hash=row["content_hash"],
                    embedding=embedding,
                )
            )
        return result

    def knowledge_counts(
        self,
        allowed_namespaces: tuple[str, ...] | None = None,
    ) -> tuple[int, int]:
        where = ""
        params: tuple[str, ...] = ()
        if allowed_namespaces is not None:
            if not allowed_namespaces:
                return 0, 0
            placeholders = ",".join("?" for _ in allowed_namespaces)
            where = f" WHERE namespace IN ({placeholders})"  # noqa: S608
            params = allowed_namespaces
        with self._lock:
            documents = self._conn.execute(
                f"SELECT COUNT(*) FROM knowledge_documents{where}",  # noqa: S608
                params,
            ).fetchone()[0]
            chunks = self._conn.execute(
                f"""
                SELECT COUNT(*) FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.id = c.document_row_id
                {where.replace("namespace", "d.namespace")}
                """,  # noqa: S608
                params,
            ).fetchone()[0]
        return int(documents), int(chunks)

    def save_question_clusters(self, report_date: str, clusters: list[QuestionCluster]) -> None:
        """Replace one date's deterministic aggregation and answer associations."""

        now = int(time.time())
        active_codes = {item.question_code for item in clusters}
        with self._transaction() as connection:
            # A forced rerun may merge or split clusters. Clear every old link for
            # this date first so a candidate can move to a different cluster without
            # violating cluster_candidates.candidate_id's UNIQUE constraint.
            connection.execute(
                """
                DELETE FROM cluster_candidates
                WHERE cluster_id IN (
                    SELECT id FROM question_clusters WHERE report_date = ?
                )
                """,
                (report_date,),
            )
            for cluster in clusters:
                connection.execute(
                    """
                    INSERT INTO question_clusters (
                        report_date, question_code, canonical_question, category,
                        occurrence_count, group_aliases_json,
                        representative_questions_json, first_sent_at_utc,
                        last_sent_at_utc, community_context_degraded,
                        updated_at_utc, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(question_code) DO UPDATE SET
                        canonical_question = excluded.canonical_question,
                        category = excluded.category,
                        occurrence_count = excluded.occurrence_count,
                        group_aliases_json = excluded.group_aliases_json,
                        representative_questions_json = excluded.representative_questions_json,
                        first_sent_at_utc = excluded.first_sent_at_utc,
                        last_sent_at_utc = excluded.last_sent_at_utc,
                        community_context_degraded = excluded.community_context_degraded,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (
                        cluster.report_date,
                        cluster.question_code,
                        cluster.canonical_question,
                        cluster.category,
                        cluster.occurrence_count,
                        json.dumps(cluster.group_aliases, ensure_ascii=False),
                        json.dumps(cluster.representative_questions, ensure_ascii=False),
                        cluster.first_sent_at_utc,
                        cluster.last_sent_at_utc,
                        int(cluster.community_context_degraded),
                        now,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT id FROM question_clusters WHERE question_code = ?",
                    (cluster.question_code,),
                ).fetchone()
                if row is None:
                    raise StorageError("无法保存聚合问题")
                cluster_id = int(row[0])
                for source_key in cluster.candidate_source_keys:
                    candidate = connection.execute(
                        "SELECT id FROM question_candidates WHERE source_key = ?",
                        (source_key,),
                    ).fetchone()
                    if candidate is None:
                        raise StorageError("聚合问题引用了不存在的候选记录")
                    connection.execute(
                        """
                        INSERT INTO cluster_candidates(cluster_id, candidate_id)
                        VALUES (?, ?)
                        """,
                        (cluster_id, int(candidate[0])),
                    )
                connection.execute(
                    "DELETE FROM community_answers WHERE cluster_id = ?",
                    (cluster_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO community_answers (
                        cluster_id, external_message_id, redacted_text,
                        sent_at_utc, confidence, direct_reply
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            cluster_id,
                            answer.external_message_id,
                            answer.redacted_text,
                            answer.sent_at_utc,
                            answer.confidence,
                            int(answer.direct_reply),
                        )
                        for answer in cluster.answers
                    ],
                )
            stale = connection.execute(
                "SELECT id, question_code FROM question_clusters WHERE report_date = ?",
                (report_date,),
            ).fetchall()
            connection.executemany(
                "DELETE FROM question_clusters WHERE id = ?",
                [
                    (int(row["id"]),)
                    for row in stale
                    if str(row["question_code"]) not in active_codes
                ],
            )

    def list_question_clusters(self, report_date: str | None) -> list[QuestionCluster]:
        with self._lock:
            if report_date is None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM question_clusters
                    ORDER BY report_date DESC, question_code
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM question_clusters
                    WHERE report_date = ? ORDER BY question_code
                    """,
                    (report_date,),
                ).fetchall()
            return [self._cluster_from_row(row) for row in rows]

    def get_question_cluster(self, question_code: str) -> QuestionCluster | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM question_clusters WHERE question_code = ?",
                (question_code.strip().upper(),),
            ).fetchone()
            return self._cluster_from_row(row) if row is not None else None

    def _cluster_from_row(self, row: sqlite3.Row) -> QuestionCluster:
        cluster_id = int(row["id"])
        source_rows = self._conn.execute(
            """
            SELECT q.source_key
            FROM cluster_candidates cc
            JOIN question_candidates q ON q.id = cc.candidate_id
            WHERE cc.cluster_id = ? ORDER BY q.sent_at_utc, q.id
            """,
            (cluster_id,),
        ).fetchall()
        answer_rows = self._conn.execute(
            """
            SELECT * FROM community_answers
            WHERE cluster_id = ? ORDER BY sent_at_utc, id
            """,
            (cluster_id,),
        ).fetchall()
        return QuestionCluster(
            question_code=str(row["question_code"]),
            report_date=str(row["report_date"]),
            canonical_question=str(row["canonical_question"]),
            category=str(row["category"]),
            candidate_source_keys=tuple(str(item[0]) for item in source_rows),
            representative_questions=tuple(json.loads(row["representative_questions_json"])),
            group_aliases=tuple(json.loads(row["group_aliases_json"])),
            first_sent_at_utc=int(row["first_sent_at_utc"]),
            last_sent_at_utc=int(row["last_sent_at_utc"]),
            answers=tuple(
                CommunityAnswer(
                    external_message_id=str(item["external_message_id"]),
                    redacted_text=str(item["redacted_text"]),
                    sent_at_utc=int(item["sent_at_utc"]),
                    confidence=float(item["confidence"]),
                    direct_reply=bool(item["direct_reply"]),
                )
                for item in answer_rows
            ),
            community_context_degraded=bool(row["community_context_degraded"]),
        )

    def save_investigation(self, result: InvestigationResult) -> int:
        """Append one auditable investigation version for a cluster."""

        with self._transaction() as connection:
            cluster = connection.execute(
                "SELECT id FROM question_clusters WHERE question_code = ?",
                (result.question_code,),
            ).fetchone()
            if cluster is None:
                raise StorageError("调查结果引用了不存在的聚合问题")
            cluster_id = int(cluster[0])
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM investigations WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            version = int(row[0]) + 1
            connection.execute(
                """
                INSERT INTO investigations (
                    cluster_id, version, status, summary, missing_information,
                    recommendation, evidence_json, flags_json, queries_json,
                    error_summary, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cluster_id,
                    version,
                    result.status.value,
                    result.summary,
                    result.missing_information,
                    result.recommendation,
                    json.dumps(
                        [
                            {
                                "namespace": item.namespace,
                                "document_id": item.document_id,
                                "title": item.title,
                                "source_url": item.source_url,
                                "updated_at": item.updated_at,
                                "excerpt": item.excerpt,
                            }
                            for item in result.evidence
                        ],
                        ensure_ascii=False,
                    ),
                    json.dumps(result.flags, ensure_ascii=False),
                    json.dumps(result.queries, ensure_ascii=False),
                    result.error_summary[:1000],
                    int(time.time()),
                ),
            )
            return version

    def latest_investigation(self, question_code: str) -> InvestigationResult | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT i.*, q.question_code
                FROM investigations i
                JOIN question_clusters q ON q.id = i.cluster_id
                WHERE q.question_code = ?
                ORDER BY i.version DESC LIMIT 1
                """,
                (question_code.strip().upper(),),
            ).fetchone()
        return _investigation_from_row(row) if row is not None else None

    def investigations_for_date(
        self,
        report_date: str | None,
    ) -> dict[str, InvestigationResult]:
        with self._lock:
            where = "WHERE q.report_date = ?" if report_date is not None else ""
            params = (report_date,) if report_date is not None else ()
            rows = self._conn.execute(
                f"""
                SELECT i.*, q.question_code
                FROM investigations i
                JOIN question_clusters q ON q.id = i.cluster_id
                JOIN (
                    SELECT cluster_id, MAX(version) AS version
                    FROM investigations GROUP BY cluster_id
                ) latest ON latest.cluster_id = i.cluster_id AND latest.version = i.version
                {where}
                """,  # noqa: S608 -- where is a fixed local fragment
                params,
            ).fetchall()
        return {str(row["question_code"]): _investigation_from_row(row) for row in rows}

    def save_report(
        self,
        *,
        report_date: str,
        subject: str,
        html_path: str,
        summary_json: str,
        status: str = "READY",
    ) -> ReportArtifact:
        """Freeze a new report version, or return an identical existing version."""

        now = int(time.time())
        with self._transaction() as connection:
            identical = connection.execute(
                """
                SELECT * FROM reports
                WHERE report_date = ? AND subject = ? AND html_path = ?
                    AND summary_json = ? AND status = ?
                ORDER BY version DESC LIMIT 1
                """,
                (report_date, subject, html_path, summary_json, status),
            ).fetchone()
            if identical is not None:
                return _report_from_row(identical)
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM reports WHERE report_date = ?",
                (report_date,),
            ).fetchone()
            version = int(row[0]) + 1
            cursor = connection.execute(
                """
                INSERT INTO reports (
                    report_date, version, status, subject, html_path,
                    summary_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (report_date, version, status, subject, html_path, summary_json, now),
            )
            created = connection.execute(
                "SELECT * FROM reports WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            if created is None:
                raise StorageError("无法保存日报")
            return _report_from_row(created)

    def latest_report(self, report_date: str) -> ReportArtifact | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM reports WHERE report_date = ?
                ORDER BY version DESC LIMIT 1
                """,
                (report_date,),
            ).fetchone()
        return _report_from_row(row) if row is not None else None

    def begin_mail_delivery(self, report_id: int, recipient_hash: str) -> bool:
        """Claim a recipient delivery unless it already completed successfully."""

        now = int(time.time())
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT status FROM mail_deliveries
                WHERE report_id = ? AND recipient_hash = ?
                """,
                (report_id, recipient_hash),
            ).fetchone()
            if existing is not None and str(existing["status"]) == "SENT":
                return False
            connection.execute(
                """
                INSERT INTO mail_deliveries (
                    report_id, recipient_hash, status, attempts,
                    error_summary, sent_at_utc, updated_at_utc
                ) VALUES (?, ?, 'SENDING', 1, '', NULL, ?)
                ON CONFLICT(report_id, recipient_hash) DO UPDATE SET
                    status = 'SENDING', attempts = mail_deliveries.attempts + 1,
                    error_summary = '', updated_at_utc = excluded.updated_at_utc
                """,
                (report_id, recipient_hash, now),
            )
            return True

    def complete_mail_delivery(
        self,
        report_id: int,
        recipient_hash: str,
        *,
        error_summary: str = "",
    ) -> None:
        status = "FAILED" if error_summary else "SENT"
        sent_at = None if error_summary else int(time.time())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE mail_deliveries
                SET status = ?, error_summary = ?, sent_at_utc = ?, updated_at_utc = ?
                WHERE report_id = ? AND recipient_hash = ?
                """,
                (
                    status,
                    error_summary[:1000],
                    sent_at,
                    int(time.time()),
                    report_id,
                    recipient_hash,
                ),
            )

    def mail_deliveries(self, report_id: int) -> list[MailDelivery]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM mail_deliveries
                WHERE report_id = ? ORDER BY recipient_hash
                """,
                (report_id,),
            ).fetchall()
        return [
            MailDelivery(
                report_id=int(row["report_id"]),
                recipient_hash=str(row["recipient_hash"]),
                status=str(row["status"]),
                attempts=int(row["attempts"]),
                error_summary=str(row["error_summary"]),
                sent_at_utc=(int(row["sent_at_utc"]) if row["sent_at_utc"] is not None else None),
            )
            for row in rows
        ]

    def report_date_delivered_to(
        self,
        report_date: str,
        recipient_hashes: tuple[str, ...],
    ) -> bool:
        """Return whether every configured recipient got any version for the date."""

        required = tuple(sorted(set(recipient_hashes)))
        if not required:
            return False
        placeholders = ", ".join("?" for _ in required)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT DISTINCT deliveries.recipient_hash
                FROM mail_deliveries AS deliveries
                JOIN reports ON reports.id = deliveries.report_id
                WHERE reports.report_date = ?
                    AND deliveries.status = 'SENT'
                    AND deliveries.recipient_hash IN ({placeholders})
                """,
                (report_date, *required),
            ).fetchall()
        delivered = {str(row["recipient_hash"]) for row in rows}
        return delivered == set(required)

    def scheduled_report_run(self, scheduled_date: str) -> ScheduledReportRun | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scheduled_report_runs WHERE scheduled_date = ?",
                (scheduled_date,),
            ).fetchone()
        return _scheduled_report_run_from_row(row) if row is not None else None

    def oldest_unfinished_scheduled_report_run(self) -> ScheduledReportRun | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM scheduled_report_runs
                WHERE status != 'SENT'
                ORDER BY scheduled_date
                LIMIT 1
                """
            ).fetchone()
        return _scheduled_report_run_from_row(row) if row is not None else None

    def begin_scheduled_report_run(
        self,
        scheduled_date: str,
        report_date: str,
        *,
        now_utc: int,
        stale_before_utc: int,
    ) -> str | None:
        """Claim a due scheduled run, including a stale RUNNING attempt."""

        claim_token = uuid.uuid4().hex
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM scheduled_report_runs WHERE scheduled_date = ?",
                (scheduled_date,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO scheduled_report_runs (
                        scheduled_date, report_date, status, attempts,
                        claim_token, error_summary, next_retry_at_utc,
                        created_at_utc, updated_at_utc, sent_at_utc
                    ) VALUES (?, ?, 'RUNNING', 1, ?, '', NULL, ?, ?, NULL)
                    """,
                    (scheduled_date, report_date, claim_token, now_utc, now_utc),
                )
                return claim_token
            if str(existing["report_date"]) != report_date:
                raise StorageError("scheduled report date does not match persisted state")
            status = str(existing["status"])
            if status == "SENT":
                return None
            retry_at = existing["next_retry_at_utc"]
            if status == "RETRY_PENDING" and retry_at is not None and int(retry_at) > now_utc:
                return None
            if status == "RUNNING" and int(existing["updated_at_utc"]) > stale_before_utc:
                return None
            connection.execute(
                """
                UPDATE scheduled_report_runs
                SET status = 'RUNNING', attempts = attempts + 1,
                    claim_token = ?, error_summary = '', next_retry_at_utc = NULL,
                    updated_at_utc = ?
                WHERE scheduled_date = ?
                """,
                (claim_token, now_utc, scheduled_date),
            )
            return claim_token

    def fail_scheduled_report_run(
        self,
        scheduled_date: str,
        *,
        error_summary: str,
        next_retry_at_utc: int,
        now_utc: int,
        claim_token: str,
    ) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_report_runs
                SET status = 'RETRY_PENDING', error_summary = ?,
                    next_retry_at_utc = ?, updated_at_utc = ?, claim_token = ''
                WHERE scheduled_date = ? AND status = 'RUNNING' AND claim_token = ?
                """,
                (
                    error_summary[:1000],
                    next_retry_at_utc,
                    now_utc,
                    scheduled_date,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise StorageError("scheduled report run is not active")

    def complete_scheduled_report_run(
        self,
        scheduled_date: str,
        *,
        now_utc: int,
        claim_token: str,
    ) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_report_runs
                SET status = 'SENT', error_summary = '', next_retry_at_utc = NULL,
                    updated_at_utc = ?, sent_at_utc = ?, claim_token = ''
                WHERE scheduled_date = ? AND status = 'RUNNING' AND claim_token = ?
                """,
                (now_utc, now_utc, scheduled_date, claim_token),
            )
            if cursor.rowcount != 1:
                raise StorageError("scheduled report run is not active")

    def bootstrap_scheduled_report_sent(
        self,
        scheduled_date: str,
        report_date: str,
        *,
        now_utc: int,
    ) -> bool:
        """Record legacy successful deliveries without rerunning the workflow."""

        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scheduled_report_runs (
                    scheduled_date, report_date, status, attempts,
                    claim_token, error_summary, next_retry_at_utc,
                    created_at_utc, updated_at_utc, sent_at_utc
                ) VALUES (?, ?, 'SENT', 0, '', '', NULL, ?, ?, ?)
                ON CONFLICT(scheduled_date) DO UPDATE SET
                    status = 'SENT', claim_token = '', error_summary = '',
                    next_retry_at_utc = NULL,
                    updated_at_utc = excluded.updated_at_utc,
                    sent_at_utc = excluded.sent_at_utc
                WHERE scheduled_report_runs.status != 'SENT'
                """,
                (scheduled_date, report_date, now_utc, now_utc, now_utc),
            )
            return cursor.rowcount == 1

    def messages_in_window(
        self,
        window: TimeWindow,
        *,
        analyzable_only: bool = False,
    ) -> list[StoredMessage]:
        """Read messages using the report window's strict half-open bounds."""

        sql = """
            SELECT
                platform_id,
                bot_self_id,
                external_message_id,
                message_fingerprint,
                session_id,
                group_id,
                group_alias,
                sender_id,
                sender_name,
                sent_at_utc,
                text,
                outline,
                reply_to_message_id,
                analyzable
            FROM messages
            WHERE sent_at_utc >= ? AND sent_at_utc < ?
        """
        params: list[int] = [window.start_timestamp, window.end_timestamp]
        if analyzable_only:
            sql += " AND analyzable = 1"
        sql += " ORDER BY sent_at_utc ASC, id ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            StoredMessage(
                platform_id=row["platform_id"],
                bot_self_id=row["bot_self_id"],
                external_message_id=row["external_message_id"],
                message_fingerprint=row["message_fingerprint"],
                session_id=row["session_id"],
                group_id=row["group_id"],
                group_alias=row["group_alias"],
                sender_id=row["sender_id"],
                sender_name=row["sender_name"],
                sent_at_utc=row["sent_at_utc"],
                text=row["text"],
                outline=row["outline"],
                reply_to_message_id=row["reply_to_message_id"],
                analyzable=bool(row["analyzable"]),
            )
            for row in rows
        ]

    def begin_processing_window(
        self,
        window: TimeWindow,
        *,
        run_id: str,
        force: bool = False,
    ) -> bool:
        """Start or retry a date; return ``False`` only when it already completed."""

        if not run_id.strip():
            raise ValueError("run_id 不能为空")
        report_date = window.report_date.isoformat()
        now = int(time.time())
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT status FROM processing_windows WHERE report_date = ?",
                (report_date,),
            ).fetchone()
            if existing is not None and existing["status"] == "COMPLETED" and not force:
                return False
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO processing_windows (
                        report_date, timezone, start_utc, end_utc, status, run_id,
                        messages_scanned, candidates_saved, included_count,
                        dropped_count, error_count, error_summary,
                        created_at_utc, updated_at_utc
                    ) VALUES (?, ?, ?, ?, 'RUNNING', ?, 0, 0, 0, 0, 0, '', ?, ?)
                    """,
                    (
                        report_date,
                        window.timezone_name,
                        window.start_timestamp,
                        window.end_timestamp,
                        run_id,
                        now,
                        now,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE processing_windows
                    SET timezone = ?, start_utc = ?, end_utc = ?, status = 'RUNNING',
                        run_id = ?, messages_scanned = 0, candidates_saved = 0,
                        included_count = 0, dropped_count = 0, error_count = 0,
                        error_summary = '', updated_at_utc = ?
                    WHERE report_date = ?
                    """,
                    (
                        window.timezone_name,
                        window.start_timestamp,
                        window.end_timestamp,
                        run_id,
                        now,
                        report_date,
                    ),
                )
            active = connection.execute(
                """
                SELECT snapshot_json FROM screening_versions
                WHERE report_date = ? AND is_active = 1
                ORDER BY version DESC LIMIT 1
                """,
                (report_date,),
            ).fetchone()
            if active is not None:
                _restore_candidate_snapshot_tx(
                    connection,
                    report_date,
                    str(active["snapshot_json"]),
                )
            version_row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1
                FROM screening_versions WHERE report_date = ?
                """,
                (report_date,),
            ).fetchone()
            connection.execute(
                """
                UPDATE screening_versions
                SET status = 'SUPERSEDED', completed_at_utc = ?
                WHERE report_date = ? AND status = 'RUNNING'
                """,
                (now, report_date),
            )
            connection.execute(
                """
                INSERT INTO screening_versions (
                    report_date, version, run_id, status, is_active,
                    snapshot_json, created_at_utc
                ) VALUES (?, ?, ?, 'RUNNING', 0, '[]', ?)
                """,
                (report_date, int(version_row[0]), run_id, now),
            )
            return True

    def complete_processing_window(
        self,
        report_date: str,
        *,
        run_id: str,
        messages_scanned: int,
        candidates_saved: int,
        included_count: int,
        dropped_count: int,
        error_count: int,
    ) -> None:
        """Atomically persist successful run totals."""

        final_status = "COMPLETED" if error_count == 0 else "RETRY_PENDING"
        with self._transaction() as connection:
            now = int(time.time())
            cursor = connection.execute(
                """
                UPDATE processing_windows
                SET status = ?, messages_scanned = ?, candidates_saved = ?,
                    included_count = ?, dropped_count = ?, error_count = ?,
                    error_summary = '', updated_at_utc = ?
                WHERE report_date = ? AND status = 'RUNNING' AND run_id = ?
                """,
                (
                    final_status,
                    messages_scanned,
                    candidates_saved,
                    included_count,
                    dropped_count,
                    error_count,
                    now,
                    report_date,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StorageError("本次日报运行已被更新的任务替代，不能写入完成状态")
            snapshot_json = _candidate_snapshot_json_tx(connection, report_date)
            version_cursor = connection.execute(
                """
                UPDATE screening_versions
                SET status = ?, snapshot_json = ?, messages_scanned = ?,
                    candidates_saved = ?, included_count = ?, dropped_count = ?,
                    error_count = ?, error_summary = '', completed_at_utc = ?
                WHERE report_date = ? AND run_id = ? AND status = 'RUNNING'
                """,
                (
                    final_status,
                    snapshot_json,
                    messages_scanned,
                    candidates_saved,
                    included_count,
                    dropped_count,
                    error_count,
                    now,
                    report_date,
                    run_id,
                ),
            )
            if version_cursor.rowcount != 1:
                raise StorageError("找不到本次筛选运行对应的版本记录")
            if error_count == 0:
                connection.execute(
                    "UPDATE screening_versions SET is_active = 0 WHERE report_date = ?",
                    (report_date,),
                )
                connection.execute(
                    "UPDATE screening_versions SET is_active = 1 WHERE run_id = ?",
                    (run_id,),
                )
            else:
                active = connection.execute(
                    """
                    SELECT snapshot_json FROM screening_versions
                    WHERE report_date = ? AND is_active = 1
                    ORDER BY version DESC LIMIT 1
                    """,
                    (report_date,),
                ).fetchone()
                if active is not None:
                    _restore_candidate_snapshot_tx(
                        connection,
                        report_date,
                        str(active["snapshot_json"]),
                    )

    def fail_processing_window(
        self,
        report_date: str,
        error_summary: str,
        *,
        run_id: str,
    ) -> None:
        """Mark a run failed and restore the last completed screening snapshot."""

        with self._transaction() as connection:
            now = int(time.time())
            snapshot_json = _candidate_snapshot_json_tx(connection, report_date)
            cursor = connection.execute(
                """
                UPDATE processing_windows
                SET status = 'FAILED', error_summary = ?, updated_at_utc = ?
                WHERE report_date = ? AND status = 'RUNNING' AND run_id = ?
                """,
                (error_summary[:1000], now, report_date, run_id),
            )
            if cursor.rowcount != 1:
                return
            connection.execute(
                """
                UPDATE screening_versions
                SET status = 'FAILED', snapshot_json = ?, error_summary = ?,
                    completed_at_utc = ?
                WHERE report_date = ? AND run_id = ? AND status = 'RUNNING'
                """,
                (snapshot_json, error_summary[:1000], now, report_date, run_id),
            )
            active = connection.execute(
                """
                SELECT snapshot_json FROM screening_versions
                WHERE report_date = ? AND is_active = 1
                ORDER BY version DESC LIMIT 1
                """,
                (report_date,),
            ).fetchone()
            if active is not None:
                _restore_candidate_snapshot_tx(
                    connection,
                    report_date,
                    str(active["snapshot_json"]),
                )

    def list_screening_versions(self, report_date: str) -> list[ScreeningVersion]:
        """List immutable AI screening attempts for one date, newest first."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM screening_versions
                WHERE report_date = ? ORDER BY version DESC
                """,
                (report_date,),
            ).fetchall()
        return [_screening_version_from_row(row) for row in rows]

    def processing_window(self, report_date: str) -> ProcessingWindowRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM processing_windows WHERE report_date = ?",
                (report_date,),
            ).fetchone()
        return _processing_window_from_row(row) if row is not None else None

    def list_question_candidates(
        self,
        *,
        report_date: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> tuple[list[QuestionCandidate], int]:
        """Return all decisions, including excluded and technical-error candidates."""

        if offset < 0 or (limit is not None and limit < 1):
            raise ValueError("limit/offset 无效")
        where = ""
        params: list[object] = []
        if report_date is not None:
            where = " WHERE report_date = ?"
            params.append(report_date)
        with self._lock:
            total_row = self._conn.execute(
                f"SELECT COUNT(*) FROM question_candidates{where}",  # noqa: S608
                params,
            ).fetchone()
            sql = f"SELECT * FROM question_candidates{where} ORDER BY report_date DESC, id ASC"  # noqa: S608
            query_params = list(params)
            if limit is not None:
                sql += " LIMIT ? OFFSET ?"
                query_params.extend((limit, offset))
            rows = self._conn.execute(sql, query_params).fetchall()
        return [_candidate_from_row(row) for row in rows], int(total_row[0])

    def get_question_candidate(self, question_code: str) -> QuestionCandidate | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM question_candidates WHERE question_code = ?",
                (question_code.strip().upper(),),
            ).fetchone()
        return _candidate_from_row(row) if row is not None else None

    def upsert_scope_candidate(
        self,
        *,
        source_key: str,
        report_date: str,
        initial: ScopeAssessment,
        final: ScopeAssessment | None = None,
        status: str = "CLASSIFIED",
        original_question: str = "",
        group_alias: str = "",
        sent_at_utc: int = 0,
    ) -> int:
        """Create or update a candidate without creating a human review queue."""

        with self._transaction() as connection:
            return self._upsert_scope_candidate_tx(
                connection,
                source_key=source_key,
                report_date=report_date,
                initial=initial,
                final=final or initial,
                status=status,
                original_question=original_question,
                group_alias=group_alias,
                sent_at_utc=sent_at_utc,
            )

    def record_scope_review(
        self,
        *,
        candidate_id: int,
        review_run_id: str,
        round_no: int,
        assessment: ScopeAssessment | None,
        error_summary: str = "",
    ) -> None:
        """Persist one automatic review round idempotently."""

        if not review_run_id.strip():
            raise ValueError("review_run_id 不能为空")
        decision = assessment.decision.value if assessment else "AUTO_REVIEW_ERROR"
        reason = assessment.reason if assessment else ""
        confidence = assessment.confidence if assessment else 0.0
        with self._transaction() as connection:
            self._record_scope_review_tx(
                connection,
                candidate_id=candidate_id,
                review_run_id=review_run_id,
                round_no=round_no,
                decision=decision,
                reason=reason,
                confidence=confidence,
                error_summary=error_summary,
            )

    def save_scope_resolution(
        self,
        *,
        source_key: str,
        report_date: str,
        review_run_id: str,
        resolution: ScopeResolution,
        original_question: str = "",
        group_alias: str = "",
        sent_at_utc: int = 0,
    ) -> int:
        """Atomically save the candidate terminal state and all AI review rounds."""

        if not review_run_id.strip():
            raise ValueError("review_run_id 不能为空")
        initial = resolution.initial_assessment or resolution.assessment
        status = "AUTO_RETRY_PENDING" if resolution.retryable else "RESOLVED"
        with self._transaction() as connection:
            candidate_id = self._upsert_scope_candidate_tx(
                connection,
                source_key=source_key,
                report_date=report_date,
                initial=initial,
                final=resolution.assessment,
                status=status,
                original_question=original_question,
                group_alias=group_alias,
                sent_at_utc=sent_at_utc,
            )
            for round_no, assessment in enumerate(
                resolution.review_attempts,
                start=1,
            ):
                self._record_scope_review_tx(
                    connection,
                    candidate_id=candidate_id,
                    review_run_id=review_run_id,
                    round_no=round_no,
                    decision=assessment.decision.value,
                    reason=assessment.reason,
                    confidence=assessment.confidence,
                    error_summary="",
                )
            if resolution.retryable and resolution.review_rounds > len(resolution.review_attempts):
                self._record_scope_review_tx(
                    connection,
                    candidate_id=candidate_id,
                    review_run_id=review_run_id,
                    round_no=resolution.review_rounds,
                    decision="AUTO_REVIEW_ERROR",
                    reason=resolution.assessment.reason,
                    confidence=0.0,
                    error_summary=resolution.error_summary,
                )
            return candidate_id

    @staticmethod
    def _upsert_scope_candidate_tx(
        connection: sqlite3.Connection,
        *,
        source_key: str,
        report_date: str,
        initial: ScopeAssessment,
        final: ScopeAssessment,
        status: str,
        original_question: str,
        group_alias: str,
        sent_at_utc: int,
    ) -> int:
        now = int(time.time())
        existing = connection.execute(
            "SELECT id, question_code FROM question_candidates WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        question_code = (
            str(existing["question_code"])
            if existing is not None
            else ReportStorage._next_question_code_tx(connection, report_date)
        )
        connection.execute(
            """
            INSERT INTO question_candidates (
                source_key,
                question_code,
                report_date,
                original_question,
                canonical_question,
                category,
                initial_decision,
                final_decision,
                reason,
                confidence,
                status,
                group_alias,
                sent_at_utc,
                created_at_utc,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                original_question = excluded.original_question,
                canonical_question = excluded.canonical_question,
                category = excluded.category,
                initial_decision = excluded.initial_decision,
                final_decision = excluded.final_decision,
                reason = excluded.reason,
                confidence = excluded.confidence,
                status = excluded.status,
                group_alias = excluded.group_alias,
                sent_at_utc = excluded.sent_at_utc,
                updated_at_utc = excluded.updated_at_utc
            """,
            (
                source_key,
                question_code,
                report_date,
                original_question,
                final.canonical_question,
                final.category,
                initial.decision.value,
                final.decision.value,
                final.reason,
                final.confidence,
                status,
                group_alias,
                int(sent_at_utc),
                now,
                now,
            ),
        )
        row = connection.execute(
            "SELECT id FROM question_candidates WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        if row is None:
            raise StorageError("无法保存问题候选记录")
        return int(row[0])

    @staticmethod
    def _next_question_code_tx(connection: sqlite3.Connection, report_date: str) -> str:
        date_key = report_date.replace("-", "")
        prefix = f"{date_key}-Q"
        rows = connection.execute(
            "SELECT question_code FROM question_candidates WHERE report_date = ?",
            (report_date,),
        ).fetchall()
        maximum = 0
        for row in rows:
            code = str(row[0])
            if code.startswith(prefix) and code[len(prefix) :].isdigit():
                maximum = max(maximum, int(code[len(prefix) :]))
        return f"{prefix}{maximum + 1:03d}"

    @staticmethod
    def _record_scope_review_tx(
        connection: sqlite3.Connection,
        *,
        candidate_id: int,
        review_run_id: str,
        round_no: int,
        decision: str,
        reason: str,
        confidence: float,
        error_summary: str,
    ) -> None:
        cursor = connection.execute(
            """
                INSERT INTO scope_review_runs (
                    candidate_id,
                    review_run_id,
                    round_no,
                    decision,
                    reason,
                    confidence,
                    error_summary,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, review_run_id, round_no) DO NOTHING
                """,
            (
                candidate_id,
                review_run_id,
                round_no,
                decision,
                reason,
                confidence,
                error_summary[:1000],
                int(time.time()),
            ),
        )
        if cursor.rowcount == 0:
            existing = connection.execute(
                """
                SELECT decision, reason, confidence, error_summary
                FROM scope_review_runs
                WHERE candidate_id = ? AND review_run_id = ? AND round_no = ?
                """,
                (candidate_id, review_run_id, round_no),
            ).fetchone()
            expected = (decision, reason, confidence, error_summary[:1000])
            actual = tuple(existing) if existing is not None else None
            if actual != expected:
                raise StorageError("同一 AI 审核轮次出现不一致的审计内容")

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StorageError("数据库尚未初始化")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._conn
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _apply_migrations(self) -> None:
        connection = self._conn
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at_utc INTEGER NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
            version = int(row[0]) if row else 0
            if version > _SCHEMA_VERSION:
                raise StorageError("数据库版本高于当前插件支持的版本，请升级插件后再启动")
            if version < 1:
                self._migration_v1(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (1, int(time.time())),
                )
                version = 1
            if version < 2:
                self._migration_v2(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (2, int(time.time())),
                )
                version = 2
            if version < 3:
                self._migration_v3(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (3, int(time.time())),
                )
                version = 3
            if version < 4:
                self._migration_v4(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (4, int(time.time())),
                )
                version = 4
            if version < 5:
                self._migration_v5(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (5, int(time.time())),
                )
                version = 5
            if version < 6:
                self._migration_v6(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (6, int(time.time())),
                )
                version = 6
            if version < 7:
                self._migration_v7(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (7, int(time.time())),
                )
                version = 7
            if version < 8:
                self._migration_v8(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                    (8, int(time.time())),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _migration_v1(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                platform_id TEXT NOT NULL,
                bot_self_id TEXT NOT NULL,
                external_message_id TEXT NOT NULL,
                message_fingerprint TEXT NOT NULL,
                session_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                group_alias TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                sent_at_utc INTEGER NOT NULL,
                text TEXT NOT NULL,
                outline TEXT NOT NULL,
                reply_to_message_id TEXT NOT NULL DEFAULT '',
                analyzable INTEGER NOT NULL CHECK(analyzable IN (0, 1)),
                captured_at_utc INTEGER NOT NULL,
                UNIQUE(platform_id, bot_self_id, external_message_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_messages_window
                ON messages(sent_at_utc, group_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_messages_fingerprint
                ON messages(message_fingerprint)
            """,
            """
            CREATE TABLE IF NOT EXISTS processing_windows (
                id INTEGER PRIMARY KEY,
                report_date TEXT NOT NULL UNIQUE,
                timezone TEXT NOT NULL,
                start_utc INTEGER NOT NULL,
                end_utc INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL,
                CHECK(start_utc < end_utc)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS question_candidates (
                id INTEGER PRIMARY KEY,
                source_key TEXT NOT NULL UNIQUE,
                report_date TEXT NOT NULL,
                canonical_question TEXT NOT NULL,
                category TEXT NOT NULL,
                initial_decision TEXT NOT NULL,
                final_decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                status TEXT NOT NULL,
                created_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scope_review_runs (
                id INTEGER PRIMARY KEY,
                candidate_id INTEGER NOT NULL,
                review_run_id TEXT NOT NULL,
                round_no INTEGER NOT NULL CHECK(round_no >= 1),
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                error_summary TEXT NOT NULL,
                created_at_utc INTEGER NOT NULL,
                UNIQUE(candidate_id, review_run_id, round_no),
                FOREIGN KEY(candidate_id)
                    REFERENCES question_candidates(id)
                    ON DELETE CASCADE
            )
            """,
        )
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _migration_v2(connection: sqlite3.Connection) -> None:
        """Retain every screening result and persist idempotent run summaries."""

        statements = (
            "ALTER TABLE question_candidates ADD COLUMN question_code TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE question_candidates ADD COLUMN original_question TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE question_candidates ADD COLUMN group_alias TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE question_candidates ADD COLUMN sent_at_utc INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_windows ADD COLUMN run_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE processing_windows ADD COLUMN messages_scanned INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_windows ADD COLUMN candidates_saved INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_windows ADD COLUMN included_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_windows ADD COLUMN dropped_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_windows ADD COLUMN error_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE processing_windows ADD COLUMN error_summary TEXT NOT NULL DEFAULT ''",
        )
        for statement in statements:
            connection.execute(statement)

        counters: dict[str, int] = {}
        rows = connection.execute(
            "SELECT id, report_date FROM question_candidates ORDER BY report_date, id"
        ).fetchall()
        for row in rows:
            report_date = str(row["report_date"])
            counters[report_date] = counters.get(report_date, 0) + 1
            code = f"{report_date.replace('-', '')}-Q{counters[report_date]:03d}"
            connection.execute(
                "UPDATE question_candidates SET question_code = ? WHERE id = ?",
                (code, int(row["id"])),
            )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_question_candidates_code
                ON question_candidates(question_code)
            """
        )

    @staticmethod
    def _migration_v3(connection: sqlite3.Connection) -> None:
        """Add allowlisted Yuque repository, document, and chunk storage."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS repositories (
                namespace TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                excluded_reason TEXT NOT NULL DEFAULT '',
                last_error TEXT NOT NULL DEFAULT '',
                synced_at_utc INTEGER,
                updated_at_utc INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY,
                namespace TEXT NOT NULL,
                yuque_id TEXT NOT NULL,
                title TEXT NOT NULL,
                slug TEXT NOT NULL,
                url TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                body TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                synced_at_utc INTEGER NOT NULL,
                UNIQUE(namespace, yuque_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_documents_namespace
                ON knowledge_documents(namespace)
            """,
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                document_row_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL DEFAULT '',
                UNIQUE(document_row_id, chunk_index),
                FOREIGN KEY(document_row_id)
                    REFERENCES knowledge_documents(id)
                    ON DELETE CASCADE
            )
            """,
        )
        for statement in statements:
            connection.execute(statement)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_candidates_date
                ON question_candidates(report_date, id)
            """
        )

    @staticmethod
    def _migration_v4(connection: sqlite3.Connection) -> None:
        """Add clustered questions, community answers, and investigation history."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS question_clusters (
                id INTEGER PRIMARY KEY,
                report_date TEXT NOT NULL,
                question_code TEXT NOT NULL UNIQUE,
                canonical_question TEXT NOT NULL,
                category TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL,
                group_aliases_json TEXT NOT NULL,
                representative_questions_json TEXT NOT NULL,
                first_sent_at_utc INTEGER NOT NULL,
                last_sent_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL,
                created_at_utc INTEGER NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_question_clusters_date
                ON question_clusters(report_date, question_code)
            """,
            """
            CREATE TABLE IF NOT EXISTS cluster_candidates (
                cluster_id INTEGER NOT NULL,
                candidate_id INTEGER NOT NULL UNIQUE,
                PRIMARY KEY(cluster_id, candidate_id),
                FOREIGN KEY(cluster_id) REFERENCES question_clusters(id) ON DELETE CASCADE,
                FOREIGN KEY(candidate_id) REFERENCES question_candidates(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS community_answers (
                id INTEGER PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                external_message_id TEXT NOT NULL,
                redacted_text TEXT NOT NULL,
                sent_at_utc INTEGER NOT NULL,
                confidence REAL NOT NULL,
                direct_reply INTEGER NOT NULL CHECK(direct_reply IN (0, 1)),
                UNIQUE(cluster_id, external_message_id),
                FOREIGN KEY(cluster_id) REFERENCES question_clusters(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS investigations (
                id INTEGER PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                missing_information TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                flags_json TEXT NOT NULL,
                queries_json TEXT NOT NULL,
                error_summary TEXT NOT NULL,
                created_at_utc INTEGER NOT NULL,
                UNIQUE(cluster_id, version),
                FOREIGN KEY(cluster_id) REFERENCES question_clusters(id) ON DELETE CASCADE
            )
            """,
        )
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _migration_v5(connection: sqlite3.Connection) -> None:
        """Add frozen report versions and idempotent per-recipient delivery state."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY,
                report_date TEXT NOT NULL,
                version INTEGER NOT NULL CHECK(version >= 1),
                status TEXT NOT NULL,
                subject TEXT NOT NULL,
                html_path TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at_utc INTEGER NOT NULL,
                UNIQUE(report_date, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mail_deliveries (
                id INTEGER PRIMARY KEY,
                report_id INTEGER NOT NULL,
                recipient_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error_summary TEXT NOT NULL DEFAULT '',
                sent_at_utc INTEGER,
                updated_at_utc INTEGER NOT NULL,
                UNIQUE(report_id, recipient_hash),
                FOREIGN KEY(report_id) REFERENCES reports(id) ON DELETE CASCADE
            )
            """,
        )
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _migration_v6(connection: sqlite3.Connection) -> None:
        """Add immutable full-date screening snapshots with one active version."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS screening_versions (
                id INTEGER PRIMARY KEY,
                report_date TEXT NOT NULL,
                version INTEGER NOT NULL CHECK(version >= 1),
                run_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0 CHECK(is_active IN (0, 1)),
                snapshot_json TEXT NOT NULL DEFAULT '[]',
                messages_scanned INTEGER NOT NULL DEFAULT 0,
                candidates_saved INTEGER NOT NULL DEFAULT 0,
                included_count INTEGER NOT NULL DEFAULT 0,
                dropped_count INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                error_summary TEXT NOT NULL DEFAULT '',
                created_at_utc INTEGER NOT NULL,
                completed_at_utc INTEGER,
                UNIQUE(report_date, version)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_screening_versions_date
            ON screening_versions(report_date, version DESC)
            """
        )
        dates = connection.execute(
            "SELECT DISTINCT report_date FROM question_candidates ORDER BY report_date"
        ).fetchall()
        now = int(time.time())
        for row in dates:
            report_date = str(row[0])
            existing = connection.execute(
                "SELECT 1 FROM screening_versions WHERE report_date = ?",
                (report_date,),
            ).fetchone()
            if existing is not None:
                continue
            window = connection.execute(
                "SELECT * FROM processing_windows WHERE report_date = ?",
                (report_date,),
            ).fetchone()
            status = str(window["status"]) if window is not None else "LEGACY"
            is_active = int(status == "COMPLETED")
            connection.execute(
                """
                INSERT INTO screening_versions (
                    report_date, version, run_id, status, is_active, snapshot_json,
                    messages_scanned, candidates_saved, included_count,
                    dropped_count, error_count, error_summary,
                    created_at_utc, completed_at_utc
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_date,
                    f"legacy:{report_date}",
                    status,
                    is_active,
                    _candidate_snapshot_json_tx(connection, report_date),
                    int(window["messages_scanned"]) if window is not None else 0,
                    int(window["candidates_saved"]) if window is not None else 0,
                    int(window["included_count"]) if window is not None else 0,
                    int(window["dropped_count"]) if window is not None else 0,
                    int(window["error_count"]) if window is not None else 0,
                    str(window["error_summary"]) if window is not None else "",
                    int(window["created_at_utc"]) if window is not None else now,
                    int(window["updated_at_utc"]) if window is not None else now,
                ),
            )

    @staticmethod
    def _migration_v7(connection: sqlite3.Connection) -> None:
        """Persist automatic daily report scheduling and retry state."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_report_runs (
                scheduled_date TEXT PRIMARY KEY,
                report_date TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('RUNNING', 'RETRY_PENDING', 'SENT')),
                attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                claim_token TEXT NOT NULL DEFAULT '',
                error_summary TEXT NOT NULL DEFAULT '',
                next_retry_at_utc INTEGER,
                created_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL,
                sent_at_utc INTEGER
            )
            """
        )

    @staticmethod
    def _migration_v8(connection: sqlite3.Connection) -> None:
        """Persist per-question community-context degradation."""

        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(question_clusters)").fetchall()
        }
        if "community_context_degraded" not in columns:
            connection.execute(
                """
                ALTER TABLE question_clusters
                ADD COLUMN community_context_degraded INTEGER NOT NULL DEFAULT 0
                    CHECK(community_context_degraded IN (0, 1))
                """
            )


def _scheduled_report_run_from_row(row: sqlite3.Row) -> ScheduledReportRun:
    return ScheduledReportRun(
        scheduled_date=str(row["scheduled_date"]),
        report_date=str(row["report_date"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        claim_token=str(row["claim_token"]),
        error_summary=str(row["error_summary"]),
        next_retry_at_utc=(
            int(row["next_retry_at_utc"])
            if row["next_retry_at_utc"] is not None
            else None
        ),
        created_at_utc=int(row["created_at_utc"]),
        updated_at_utc=int(row["updated_at_utc"]),
        sent_at_utc=(int(row["sent_at_utc"]) if row["sent_at_utc"] is not None else None),
    )


def _candidate_from_row(row: sqlite3.Row) -> QuestionCandidate:
    return QuestionCandidate(
        question_code=str(row["question_code"]),
        source_key=str(row["source_key"]),
        report_date=str(row["report_date"]),
        original_question=str(row["original_question"]),
        canonical_question=str(row["canonical_question"]),
        category=str(row["category"]),
        initial_decision=str(row["initial_decision"]),
        final_decision=str(row["final_decision"]),
        reason=str(row["reason"]),
        confidence=float(row["confidence"]),
        status=str(row["status"]),
        group_alias=str(row["group_alias"]),
        sent_at_utc=int(row["sent_at_utc"]),
        created_at_utc=int(row["created_at_utc"]),
        updated_at_utc=int(row["updated_at_utc"]),
    )


_CANDIDATE_SNAPSHOT_COLUMNS = (
    "source_key",
    "question_code",
    "report_date",
    "original_question",
    "canonical_question",
    "category",
    "initial_decision",
    "final_decision",
    "reason",
    "confidence",
    "status",
    "group_alias",
    "sent_at_utc",
    "created_at_utc",
    "updated_at_utc",
)


def _candidate_snapshot_json_tx(
    connection: sqlite3.Connection,
    report_date: str,
) -> str:
    rows = connection.execute(
        "SELECT * FROM question_candidates WHERE report_date = ? ORDER BY id",
        (report_date,),
    ).fetchall()
    return json.dumps(
        [
            {column: row[column] for column in _CANDIDATE_SNAPSHOT_COLUMNS}
            for row in rows
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _restore_candidate_snapshot_tx(
    connection: sqlite3.Connection,
    report_date: str,
    snapshot_json: str,
) -> None:
    try:
        raw_items = json.loads(snapshot_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StorageError("筛选版本快照损坏，无法恢复") from exc
    if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
        raise StorageError("筛选版本快照格式无效")
    items: list[dict[str, object]] = []
    source_keys: set[str] = set()
    for raw in raw_items:
        if any(column not in raw for column in _CANDIDATE_SNAPSHOT_COLUMNS):
            raise StorageError("筛选版本快照字段不完整")
        if str(raw["report_date"]) != report_date:
            raise StorageError("筛选版本快照日期不匹配")
        source_key = str(raw["source_key"])
        if not source_key or source_key in source_keys:
            raise StorageError("筛选版本快照来源键无效")
        source_keys.add(source_key)
        items.append(raw)

    current = connection.execute(
        "SELECT id, source_key FROM question_candidates WHERE report_date = ?",
        (report_date,),
    ).fetchall()
    connection.executemany(
        "DELETE FROM question_candidates WHERE id = ?",
        [(int(row["id"]),) for row in current if str(row["source_key"]) not in source_keys],
    )
    placeholders = ", ".join("?" for _ in _CANDIDATE_SNAPSHOT_COLUMNS)
    update_columns = tuple(
        column for column in _CANDIDATE_SNAPSHOT_COLUMNS if column != "source_key"
    )
    update_sql = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
    connection.executemany(
        f"""
        INSERT INTO question_candidates ({', '.join(_CANDIDATE_SNAPSHOT_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(source_key) DO UPDATE SET {update_sql}
        """,  # noqa: S608 -- identifiers are fixed module constants
        [tuple(item[column] for column in _CANDIDATE_SNAPSHOT_COLUMNS) for item in items],
    )


def _screening_version_from_row(row: sqlite3.Row) -> ScreeningVersion:
    return ScreeningVersion(
        report_date=str(row["report_date"]),
        version=int(row["version"]),
        run_id=str(row["run_id"]),
        status=str(row["status"]),
        is_active=bool(row["is_active"]),
        messages_scanned=int(row["messages_scanned"]),
        candidates_saved=int(row["candidates_saved"]),
        included_count=int(row["included_count"]),
        dropped_count=int(row["dropped_count"]),
        error_count=int(row["error_count"]),
        error_summary=str(row["error_summary"]),
        created_at_utc=int(row["created_at_utc"]),
        completed_at_utc=(
            int(row["completed_at_utc"]) if row["completed_at_utc"] is not None else None
        ),
    )


def _investigation_from_row(row: sqlite3.Row) -> InvestigationResult:
    evidence_data = json.loads(row["evidence_json"])
    return InvestigationResult(
        question_code=str(row["question_code"]),
        status=CoverageStatus(str(row["status"])),
        summary=str(row["summary"]),
        missing_information=str(row["missing_information"]),
        recommendation=str(row["recommendation"]),
        evidence=tuple(
            EvidenceItem(
                namespace=str(item.get("namespace", "")),
                document_id=str(item.get("document_id", "")),
                title=str(item.get("title", "")),
                source_url=str(item.get("source_url", "")),
                updated_at=str(item.get("updated_at", "")),
                excerpt=str(item.get("excerpt", "")),
            )
            for item in evidence_data
            if isinstance(item, dict)
        ),
        flags=tuple(str(item) for item in json.loads(row["flags_json"])),
        queries=tuple(str(item) for item in json.loads(row["queries_json"])),
        error_summary=str(row["error_summary"]),
    )


def _report_from_row(row: sqlite3.Row) -> ReportArtifact:
    return ReportArtifact(
        report_id=int(row["id"]),
        report_date=str(row["report_date"]),
        version=int(row["version"]),
        status=str(row["status"]),
        subject=str(row["subject"]),
        html_path=str(row["html_path"]),
        summary_json=str(row["summary_json"]),
        created_at_utc=int(row["created_at_utc"]),
    )


def _processing_window_from_row(row: sqlite3.Row) -> ProcessingWindowRecord:
    return ProcessingWindowRecord(
        report_date=str(row["report_date"]),
        timezone=str(row["timezone"]),
        start_utc=int(row["start_utc"]),
        end_utc=int(row["end_utc"]),
        status=str(row["status"]),
        run_id=str(row["run_id"]),
        messages_scanned=int(row["messages_scanned"]),
        candidates_saved=int(row["candidates_saved"]),
        included_count=int(row["included_count"]),
        dropped_count=int(row["dropped_count"]),
        error_count=int(row["error_count"]),
        error_summary=str(row["error_summary"]),
        created_at_utc=int(row["created_at_utc"]),
        updated_at_utc=int(row["updated_at_utc"]),
    )
