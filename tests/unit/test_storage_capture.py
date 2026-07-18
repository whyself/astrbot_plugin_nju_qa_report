from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from nju_report.config import PluginConfig
from nju_report.message_capture import MessageCaptureService
from nju_report.models import (
    CaptureOutcome,
    Clarity,
    KnowledgeValue,
    MessageEnvelope,
    ScopeAssessment,
    ScopeDecision,
)
from nju_report.storage import ReportStorage, StorageError
from nju_report.time_windows import natural_day_window


def _message(**overrides: object) -> MessageEnvelope:
    values: dict[str, object] = {
        "platform_id": "aiocqhttp:default",
        "bot_self_id": "999",
        "external_message_id": "m-1",
        "session_id": "group-session",
        "group_id": "123456",
        "sender_id": "10001",
        "sender_name": "测试用户",
        "sent_at_utc": 1783785600,
        "text": "一卡通丢了去哪里补办？",
        "outline": "一卡通丢了去哪里补办？",
    }
    values.update(overrides)
    return MessageEnvelope(**values)  # type: ignore[arg-type]


def _capture_service(tmp_path: Path) -> tuple[ReportStorage, MessageCaptureService]:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    config = PluginConfig.from_mapping(
        {
            "capture_enabled": True,
            "target_group_ids": ["123456"],
            "group_aliases": {"123456": "迎新群"},
        }
    )
    return storage, MessageCaptureService(config, storage)


def test_storage_initialization_is_idempotent_and_uses_safety_pragmas(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.sqlite3"
    storage = ReportStorage(path)
    storage.initialize()
    storage.initialize()

    assert str(storage.pragma("journal_mode")).lower() == "wal"
    assert storage.pragma("foreign_keys") == 1
    assert storage.pragma("busy_timeout") == 5000

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == {
        "schema_migrations",
        "messages",
        "processing_windows",
        "question_candidates",
        "scope_review_runs",
        "repositories",
        "knowledge_documents",
        "knowledge_chunks",
        "question_clusters",
        "cluster_candidates",
        "community_answers",
        "investigations",
        "reports",
        "mail_deliveries",
        "screening_versions",
        "scheduled_report_runs",
    }
    storage.close()


def test_superseded_run_cannot_complete_or_fail_newer_window(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")
    assert storage.begin_processing_window(window, run_id="old") is True
    assert storage.begin_processing_window(window, run_id="new", force=True) is True

    with pytest.raises(StorageError, match="更新的任务替代"):
        storage.complete_processing_window(
            window.report_date.isoformat(),
            run_id="old",
            messages_scanned=1,
            candidates_saved=1,
            included_count=1,
            dropped_count=0,
            error_count=0,
        )
    storage.fail_processing_window(
        window.report_date.isoformat(),
        "old failed",
        run_id="old",
    )
    current = storage.processing_window(window.report_date.isoformat())
    assert current is not None
    assert current.status == "RUNNING"
    assert current.run_id == "new"

    storage.complete_processing_window(
        window.report_date.isoformat(),
        run_id="new",
        messages_scanned=1,
        candidates_saved=1,
        included_count=1,
        dropped_count=0,
        error_count=0,
    )
    assert storage.processing_window(window.report_date.isoformat()).status == "COMPLETED"
    storage.close()


def test_migrations_tolerate_indexes_left_by_an_older_partial_run(tmp_path: Path) -> None:
    """A reload must recover when schema rows lag behind already-created indexes."""

    path = tmp_path / "report.sqlite3"
    storage = ReportStorage(path)
    storage.initialize()
    storage.close()

    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= 3")
        connection.commit()

    recovered = ReportStorage(path)
    recovered.initialize()
    with sqlite3.connect(path) as connection:
        version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == 9
    recovered.close()


def test_v9_migration_marks_legacy_degradation_audit_unknown(tmp_path: Path) -> None:
    path = tmp_path / "report.sqlite3"
    storage = ReportStorage(path)
    storage.initialize()
    storage.close()

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO question_clusters (
                report_date, question_code, canonical_question, category,
                occurrence_count, group_aliases_json, representative_questions_json,
                first_sent_at_utc, last_sent_at_utc, updated_at_utc, created_at_utc,
                community_context_degraded,
                community_context_degradation_reason, community_context_audit_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-07-12",
                "20260712-Q001",
                "测试问题",
                "测试",
                0,
                "[]",
                '["测试问题"]',
                1,
                1,
                1,
                1,
                1,
                "",
                "{}",
            ),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version = 9")
        connection.commit()

    recovered = ReportStorage(path)
    recovered.initialize()
    cluster = recovered.get_question_cluster("20260712-Q001")

    assert cluster is not None
    assert cluster.community_context_degradation_reason.value == "LEGACY_UNKNOWN"
    assert cluster.community_context_audit.fallback_actions == (
        "LEGACY_AUDIT_UNAVAILABLE",
    )
    recovered.close()


def test_scheduled_report_run_state_is_persistent_and_retryable(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()

    assert storage.scheduled_report_run("2026-07-14") is None
    first_claim = storage.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=100,
        stale_before_utc=0,
    )
    assert first_claim is not None
    running = storage.scheduled_report_run("2026-07-14")
    assert running is not None
    assert running.status == "RUNNING"
    assert running.attempts == 1

    storage.fail_scheduled_report_run(
        "2026-07-14",
        error_summary="smtp failed",
        next_retry_at_utc=1000,
        now_utc=101,
        claim_token=first_claim,
    )
    failed = storage.scheduled_report_run("2026-07-14")
    assert failed is not None
    assert failed.status == "RETRY_PENDING"
    assert failed.next_retry_at_utc == 1000
    assert storage.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=999,
        stale_before_utc=0,
    ) is None

    retry_claim = storage.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=1000,
        stale_before_utc=0,
    )
    assert retry_claim is not None
    retried = storage.scheduled_report_run("2026-07-14")
    assert retried is not None
    assert retried.status == "RUNNING"
    assert retried.attempts == 2

    storage.complete_scheduled_report_run(
        "2026-07-14",
        now_utc=1001,
        claim_token=retry_claim,
    )
    sent = storage.scheduled_report_run("2026-07-14")
    assert sent is not None
    assert sent.status == "SENT"
    assert sent.sent_at_utc == 1001
    assert storage.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=2000,
        stale_before_utc=0,
    ) is None
    storage.close()


def test_fresh_running_schedule_claim_is_protected_until_stale(tmp_path: Path) -> None:
    path = tmp_path / "report.sqlite3"
    first = ReportStorage(path)
    first.initialize()
    second = ReportStorage(path)
    second.initialize()

    first_claim = first.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=100,
        stale_before_utc=0,
    )
    assert first_claim is not None
    assert second.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=101,
        stale_before_utc=-1699,
    ) is None

    replacement_claim = second.begin_scheduled_report_run(
        "2026-07-14",
        "2026-07-13",
        now_utc=1900,
        stale_before_utc=100,
    )
    assert replacement_claim is not None
    assert replacement_claim != first_claim
    with pytest.raises(StorageError, match="not active"):
        first.complete_scheduled_report_run(
            "2026-07-14",
            now_utc=1901,
            claim_token=first_claim,
        )
    second.complete_scheduled_report_run(
        "2026-07-14",
        now_utc=1901,
        claim_token=replacement_claim,
    )
    first.close()
    second.close()


def test_failed_screening_version_restores_last_completed_snapshot(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    report_date = "2026-07-12"
    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")
    include = ScopeAssessment(
        decision=ScopeDecision.INCLUDE,
        reason="是可沉淀问题",
        confidence=0.9,
        canonical_question="南京大学校园卡如何补办？",
        category="校园卡",
    )
    drop = ScopeAssessment(
        decision=ScopeDecision.DROP,
        reason="未入选",
        confidence=0.8,
    )

    assert storage.begin_processing_window(window, run_id="good") is True
    storage.upsert_scope_candidate(
        source_key="m1",
        report_date=report_date,
        initial=include,
        original_question="校园卡怎么补办",
    )
    storage.complete_processing_window(
        report_date,
        run_id="good",
        messages_scanned=1,
        candidates_saved=1,
        included_count=1,
        dropped_count=0,
        error_count=0,
    )

    assert storage.begin_processing_window(window, run_id="bad", force=True) is True
    storage.upsert_scope_candidate(
        source_key="m1",
        report_date=report_date,
        initial=drop,
        original_question="校园卡怎么补办",
    )
    storage.upsert_scope_candidate(
        source_key="m2",
        report_date=report_date,
        initial=drop,
        original_question="闲聊",
    )
    storage.complete_processing_window(
        report_date,
        run_id="bad",
        messages_scanned=2,
        candidates_saved=2,
        included_count=0,
        dropped_count=1,
        error_count=1,
    )

    candidates, total = storage.list_question_candidates(report_date=report_date, limit=None)
    assert total == 1
    assert candidates[0].source_key == "m1"
    assert candidates[0].final_decision == "INCLUDE"
    versions = storage.list_screening_versions(report_date)
    assert [(item.version, item.status, item.is_active) for item in versions] == [
        (2, "RETRY_PENDING", False),
        (1, "COMPLETED", True),
    ]
    assert versions[0].error_count == 1
    storage.close()


def test_failed_migration_rolls_back_partial_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "broken.sqlite3"

    def fail_halfway(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE should_rollback(id INTEGER PRIMARY KEY)")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(
        ReportStorage,
        "_migration_v1",
        staticmethod(fail_halfway),
    )
    storage = ReportStorage(path)
    with pytest.raises(RuntimeError, match="injected"):
        storage.initialize()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "should_rollback" not in tables
    assert "schema_migrations" not in tables


def test_capture_is_idempotent_and_preserves_reply_reference(tmp_path: Path) -> None:
    storage, capture = _capture_service(tmp_path)
    message = _message(reply_to_message_id="not-yet-captured")

    assert capture.capture(message) is CaptureOutcome.CAPTURED
    assert capture.capture(message) is CaptureOutcome.DUPLICATE
    assert storage.message_count() == 1

    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")
    stored = storage.messages_in_window(window)
    assert stored[0].reply_to_message_id == "not-yet-captured"
    assert stored[0].group_alias == "迎新群"
    storage.close()


def test_capture_filters_group_bot_system_command_and_empty_messages(
    tmp_path: Path,
) -> None:
    storage, capture = _capture_service(tmp_path)

    assert capture.capture(_message(group_id="other")) is CaptureOutcome.OUT_OF_SCOPE_GROUP
    assert capture.capture(_message(sender_id="999")) is CaptureOutcome.BOT_MESSAGE
    assert capture.capture(_message(is_self_message=True)) is CaptureOutcome.BOT_MESSAGE
    assert capture.capture(_message(is_system_message=True)) is CaptureOutcome.SYSTEM_MESSAGE
    assert capture.capture(_message(text="/nju status")) is CaptureOutcome.COMMAND_MESSAGE
    assert capture.capture(_message(text="", outline="")) is CaptureOutcome.EMPTY_MESSAGE
    assert capture.capture(_message(is_group_message=False)) is CaptureOutcome.PRIVATE_MESSAGE
    assert storage.message_count() == 0
    storage.close()


def test_attachment_only_message_is_stored_and_sent_to_ai_screening(tmp_path: Path) -> None:
    storage, capture = _capture_service(tmp_path)

    assert capture.capture(_message(text="", outline="[图片]")) is CaptureOutcome.CAPTURED
    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")
    assert len(storage.messages_in_window(window)) == 1
    assert storage.messages_in_window(window)[0].analyzable is True
    assert len(storage.messages_in_window(window, analyzable_only=True)) == 1
    storage.close()


def test_window_query_includes_start_and_excludes_end(tmp_path: Path) -> None:
    storage, capture = _capture_service(tmp_path)
    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")

    capture.capture(_message(external_message_id="start", sent_at_utc=window.start_timestamp))
    capture.capture(_message(external_message_id="end", sent_at_utc=window.end_timestamp))

    assert [item.external_message_id for item in storage.messages_in_window(window)] == ["start"]
    storage.close()


def test_disabled_capture_never_writes(tmp_path: Path) -> None:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    capture = MessageCaptureService(PluginConfig.from_mapping({}), storage)
    assert capture.capture(_message()) is CaptureOutcome.DISABLED
    assert storage.message_count() == 0
    storage.close()


def test_missing_adapter_ids_preserve_identical_occurrences(tmp_path: Path) -> None:
    storage, capture = _capture_service(tmp_path)
    first = _message(external_message_id="")
    second = _message(external_message_id="")

    assert capture.capture(first) is CaptureOutcome.CAPTURED
    assert capture.capture(second) is CaptureOutcome.CAPTURED
    window = natural_day_window(date(2026, 7, 12), "Asia/Shanghai")
    stored = storage.messages_in_window(window)
    assert len(stored) == 2
    assert stored[0].external_message_id != stored[1].external_message_id
    assert stored[0].message_fingerprint == stored[1].message_fingerprint
    storage.close()


def test_raw_message_retention_uses_strict_cutoff(tmp_path: Path) -> None:
    storage, capture = _capture_service(tmp_path)
    capture.capture(_message(external_message_id="old", sent_at_utc=99))
    capture.capture(_message(external_message_id="edge", sent_at_utc=100))

    assert storage.delete_expired_messages(100) == 1
    assert storage.delete_expired_messages(100) == 0
    assert storage.message_count() == 1
    storage.close()


def test_scope_audit_round_is_idempotent_and_has_no_human_queue(tmp_path: Path) -> None:
    path = tmp_path / "report.sqlite3"
    storage = ReportStorage(path)
    storage.initialize()
    initial = ScopeAssessment(
        decision=ScopeDecision.AUTO_REVIEW,
        reason="需要独立审核",
        confidence=0.5,
        clarity=Clarity.UNCERTAIN,
        knowledge_value=KnowledgeValue.MEDIUM,
    )
    reviewed = ScopeAssessment(
        decision=ScopeDecision.INCLUDE,
        reason="属于校园卡公共办理问题",
        confidence=0.9,
        canonical_question="南京大学校园卡丢失后如何补办？",
        category="校园生活/校园卡",
        clarity=Clarity.CLEAR,
        knowledge_value=KnowledgeValue.HIGH,
    )
    candidate_id = storage.upsert_scope_candidate(
        source_key="20260712:m-1",
        report_date="2026-07-12",
        initial=initial,
        final=reviewed,
    )
    storage.record_scope_review(
        candidate_id=candidate_id,
        review_run_id="run-1",
        round_no=1,
        assessment=reviewed,
    )
    storage.record_scope_review(
        candidate_id=candidate_id,
        review_run_id="run-1",
        round_no=1,
        assessment=reviewed,
    )
    conflicting = ScopeAssessment(
        decision=ScopeDecision.DROP,
        reason="不同结果",
        confidence=0.8,
    )
    with pytest.raises(StorageError, match="不一致"):
        storage.record_scope_review(
            candidate_id=candidate_id,
            review_run_id="run-1",
            round_no=1,
            assessment=conflicting,
        )
    storage.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM scope_review_runs").fetchone()[0] == 1
        table_names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "review_queue" not in table_names
    assert "manual_reviews" not in table_names
