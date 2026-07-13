"""Fast, passive message filtering and persistence."""

from __future__ import annotations

import hashlib
import time
import uuid

from .config import PluginConfig
from .models import CaptureOutcome, MessageEnvelope, StoredMessage
from .storage import ReportStorage


class MessageCaptureService:
    """Capture configured group traffic without invoking an LLM or replying."""

    def __init__(self, config: PluginConfig, storage: ReportStorage) -> None:
        self._config = config
        self._storage = storage
        self._target_groups = frozenset(config.target_group_ids)

    def capture(self, message: MessageEnvelope) -> CaptureOutcome:
        if not self._config.capture_enabled:
            return CaptureOutcome.DISABLED
        if not message.is_group_message:
            return CaptureOutcome.PRIVATE_MESSAGE

        group_id = str(message.group_id).strip()
        if self._config.capture_mode == "selected_groups" and group_id not in self._target_groups:
            return CaptureOutcome.OUT_OF_SCOPE_GROUP

        sender_id = str(message.sender_id).strip()
        bot_self_id = str(message.bot_self_id).strip()
        if message.is_self_message or (sender_id and bot_self_id and sender_id == bot_self_id):
            return CaptureOutcome.BOT_MESSAGE
        if message.is_system_message:
            return CaptureOutcome.SYSTEM_MESSAGE

        text = message.text.strip()
        outline = message.outline.strip()
        if text and any(
            text.lstrip().startswith(prefix) for prefix in self._config.command_prefixes
        ):
            return CaptureOutcome.COMMAND_MESSAGE
        if not text and not outline:
            return CaptureOutcome.EMPTY_MESSAGE

        external_message_id = str(message.external_message_id).strip()
        fingerprint = _message_fingerprint(message)
        if not external_message_id:
            # Without an adapter ID, true duplicate delivery cannot be
            # distinguished from two identical messages in the same second.
            # Preserve both occurrences and retain a fingerprint for auditing.
            external_message_id = f"occurrence:{uuid.uuid4().hex}"

        stored = StoredMessage(
            platform_id=str(message.platform_id).strip(),
            bot_self_id=bot_self_id,
            external_message_id=external_message_id,
            message_fingerprint=fingerprint,
            session_id=str(message.session_id).strip(),
            group_id=group_id,
            group_alias=self._config.group_alias(group_id),
            sender_id=sender_id,
            sender_name=str(message.sender_name).strip(),
            sent_at_utc=(
                int(message.sent_at_utc) if int(message.sent_at_utc) > 0 else int(time.time())
            ),
            text=text,
            outline=outline,
            reply_to_message_id=str(message.reply_to_message_id).strip(),
            analyzable=bool(text),
        )
        inserted = self._storage.insert_message(stored)
        return CaptureOutcome.CAPTURED if inserted else CaptureOutcome.DUPLICATE


def _message_fingerprint(message: MessageEnvelope) -> str:
    """Hash stable content for duplicate-delivery diagnostics."""

    parts = (
        str(message.platform_id),
        str(message.bot_self_id),
        str(message.session_id),
        str(message.group_id),
        str(message.sender_id),
        str(message.sent_at_utc),
        message.text,
        message.outline,
        str(message.reply_to_message_id),
    )
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest
