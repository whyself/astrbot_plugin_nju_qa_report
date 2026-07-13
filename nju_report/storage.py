"""SQLite persistence for captured messages and processing audit records."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import (
    ProcessingWindowRecord,
    QuestionCandidate,
    ScopeAssessment,
    ScopeResolution,
    StoredMessage,
)
from .time_windows import TimeWindow

_SCHEMA_VERSION = 2


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

    def begin_processing_window(self, window: TimeWindow, *, run_id: str) -> bool:
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
            if existing is not None and existing["status"] == "COMPLETED":
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
            return True

    def complete_processing_window(
        self,
        report_date: str,
        *,
        messages_scanned: int,
        candidates_saved: int,
        included_count: int,
        dropped_count: int,
        error_count: int,
    ) -> None:
        """Atomically persist successful run totals."""

        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE processing_windows
                SET status = 'COMPLETED', messages_scanned = ?, candidates_saved = ?,
                    included_count = ?, dropped_count = ?, error_count = ?,
                    error_summary = '', updated_at_utc = ?
                WHERE report_date = ? AND status = 'RUNNING'
                """,
                (
                    messages_scanned,
                    candidates_saved,
                    included_count,
                    dropped_count,
                    error_count,
                    int(time.time()),
                    report_date,
                ),
            )
            if cursor.rowcount != 1:
                raise StorageError("日报处理窗口不是可完成的 RUNNING 状态")

    def fail_processing_window(self, report_date: str, error_summary: str) -> None:
        """Mark a run retryable while keeping already written candidate records."""

        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE processing_windows
                SET status = 'FAILED', error_summary = ?, updated_at_utc = ?
                WHERE report_date = ? AND status = 'RUNNING'
                """,
                (error_summary[:1000], int(time.time()), report_date),
            )

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
            "CREATE UNIQUE INDEX idx_question_candidates_code ON question_candidates(question_code)"
        )
        connection.execute(
            "CREATE INDEX idx_question_candidates_date ON question_candidates(report_date, id)"
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
