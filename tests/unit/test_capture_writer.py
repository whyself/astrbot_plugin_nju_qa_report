from __future__ import annotations

import asyncio
from pathlib import Path

from nju_report.capture_writer import AsyncCaptureWriter
from nju_report.config import PluginConfig
from nju_report.message_capture import MessageCaptureService
from nju_report.models import MessageEnvelope
from nju_report.storage import ReportStorage


def _message(message_id: str) -> MessageEnvelope:
    return MessageEnvelope(
        platform_id="aiocqhttp:default",
        bot_self_id="999",
        external_message_id=message_id,
        session_id="session",
        group_id="123456",
        sender_id="10001",
        sender_name="用户",
        sent_at_utc=1783785600,
        text="校园卡丢了怎么补办？",
        outline="校园卡丢了怎么补办？",
    )


def _writer(
    tmp_path: Path, *, max_queue_size: int = 10
) -> tuple[ReportStorage, AsyncCaptureWriter]:
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    config = PluginConfig.from_mapping({"capture_enabled": True, "target_group_ids": ["123456"]})
    service = MessageCaptureService(config, storage)
    return storage, AsyncCaptureWriter(service, max_queue_size=max_queue_size)


def test_writer_persists_messages_outside_the_event_loop_thread(tmp_path: Path) -> None:
    storage, writer = _writer(tmp_path)

    async def scenario() -> None:
        writer.start()
        assert writer.submit(_message("m-1")) is True
        await writer.flush()
        await writer.close()

    asyncio.run(scenario())
    assert storage.message_count() == 1
    assert writer.pending_count == 0
    assert writer.write_error_count == 0
    storage.close()


def test_full_queue_rejects_immediately_and_counts_drop(tmp_path: Path) -> None:
    storage, writer = _writer(tmp_path, max_queue_size=1)

    async def scenario() -> None:
        writer.start()
        assert writer.submit(_message("m-1")) is True
        # No await has yielded control to the writer yet, so the queue is full.
        assert writer.submit(_message("m-2")) is False
        await writer.close()

    asyncio.run(scenario())
    assert writer.dropped_count == 1
    assert storage.message_count() == 1
    storage.close()
