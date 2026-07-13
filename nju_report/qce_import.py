"""Import QQ Chat Exporter JSON histories into the normal capture store."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO
from zoneinfo import ZoneInfo

from .config import PluginConfig
from .message_capture import prepare_message
from .models import MessageEnvelope, StoredMessage
from .storage import ReportStorage

_MAX_SINGLE_JSON_BYTES = 256 * 1024 * 1024
_MAX_MESSAGES_PER_SOURCE = 2_000_000
_BATCH_SIZE = 1000


class QceImportError(ValueError):
    """Raised when a QCE export cannot be safely identified or parsed."""


@dataclass(frozen=True, slots=True)
class QceSourceInfo:
    path: str
    chat_name: str
    group_id: str
    message_count: int
    first_sent_at_utc: int | None
    last_sent_at_utc: int | None


@dataclass(slots=True)
class QceImportResult:
    path: str
    chat_name: str = ""
    group_id: str = ""
    scanned: int = 0
    imported: int = 0
    duplicates: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    first_sent_at_utc: int | None = None
    last_sent_at_utc: int | None = None


class QceHistoryImporter:
    """Validate configured QCE files and import them idempotently in batches."""

    def __init__(self, config: PluginConfig, storage: ReportStorage) -> None:
        self._config = config
        self._storage = storage
        self._timezone = ZoneInfo(config.timezone)
        self._bot_ids = frozenset(config.history_import_bot_qq_ids)

    def inspect_all(self) -> list[QceSourceInfo]:
        if not self._config.history_import_files:
            raise QceImportError("尚未在插件配置中上传 QCE JSON/ZIP 文件")
        return [self._inspect_one(Path(item)) for item in self._config.history_import_files]

    def import_all(self) -> list[QceImportResult]:
        if not self._config.history_import_files:
            raise QceImportError("尚未在插件配置中上传 QCE JSON/ZIP 文件")
        return [self._import_one(Path(item)) for item in self._config.history_import_files]

    def _inspect_one(self, path: Path) -> QceSourceInfo:
        with _open_qce_source(path) as (chat_info, messages):
            group_id = self._validated_group_id(chat_info, path)
            count = 0
            first: int | None = None
            last: int | None = None
            for message in messages:
                count += 1
                if count > _MAX_MESSAGES_PER_SOURCE:
                    raise QceImportError("单个导出超过 200 万条消息，请拆分后再导入")
                timestamp = _timestamp_seconds(message, self._timezone)
                if timestamp > 0:
                    first = timestamp if first is None else min(first, timestamp)
                    last = timestamp if last is None else max(last, timestamp)
            return QceSourceInfo(
                path=str(path),
                chat_name=_string(chat_info.get("name")) or "未命名群聊",
                group_id=group_id,
                message_count=count,
                first_sent_at_utc=first,
                last_sent_at_utc=last,
            )

    def _import_one(self, path: Path) -> QceImportResult:
        with _open_qce_source(path) as (chat_info, messages):
            group_id = self._validated_group_id(chat_info, path)
            result = QceImportResult(
                path=str(path),
                chat_name=_string(chat_info.get("name")) or "未命名群聊",
                group_id=group_id,
            )
            batch: list[StoredMessage] = []
            for index, raw in enumerate(messages, start=1):
                result.scanned += 1
                if result.scanned > _MAX_MESSAGES_PER_SOURCE:
                    raise QceImportError("单个导出超过 200 万条消息，请拆分后再导入")
                if bool(raw.get("recalled")):
                    result.skipped["RECALLED"] += 1
                    continue
                envelope = self._to_envelope(raw, group_id=group_id, fallback_index=index)
                timestamp = envelope.sent_at_utc
                if timestamp > 0:
                    result.first_sent_at_utc = (
                        timestamp
                        if result.first_sent_at_utc is None
                        else min(result.first_sent_at_utc, timestamp)
                    )
                    result.last_sent_at_utc = (
                        timestamp
                        if result.last_sent_at_utc is None
                        else max(result.last_sent_at_utc, timestamp)
                    )
                outcome, stored = prepare_message(
                    self._config,
                    envelope,
                    respect_capture_enabled=False,
                )
                if stored is None:
                    result.skipped[outcome.value] += 1
                    continue
                batch.append(stored)
                if len(batch) >= _BATCH_SIZE:
                    self._flush_batch(batch, result)
            self._flush_batch(batch, result)
            return result

    def _flush_batch(self, batch: list[StoredMessage], result: QceImportResult) -> None:
        if not batch:
            return
        inserted, duplicates = self._storage.insert_messages(batch)
        result.imported += inserted
        result.duplicates += duplicates
        batch.clear()

    def _validated_group_id(self, chat_info: dict[str, Any], source_path: Path) -> str:
        chat_type = _string(chat_info.get("type")).lower()
        if chat_type not in {"group", "group_chat", "2"}:
            raise QceImportError("导出文件不是 QQ 群聊记录")
        candidates = {
            _string(chat_info.get(key)) for key in ("peerUin", "peerUid", "groupCode", "groupId")
        }
        candidates.discard("")
        candidates.update(_group_ids_from_qce_filename(source_path.name))
        matches = sorted(candidates & set(self._config.target_group_ids))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise QceImportError(
                "导出文件群号与 target_group_ids 不匹配；请保留 QCE 原始的 "
                "group_<群号>_*.json 文件名，并确认只上传目标群导出"
            )
        raise QceImportError("导出文件同时匹配多个目标群，无法确定归属")

    def _to_envelope(
        self,
        raw: dict[str, Any],
        *,
        group_id: str,
        fallback_index: int,
    ) -> MessageEnvelope:
        sender_raw = raw.get("sender")
        sender = sender_raw if isinstance(sender_raw, dict) else {}
        sender_id = _string(sender.get("uin")) or _string(sender.get("uid"))
        sender_name = _string(sender.get("name")) or _string(sender.get("nickname"))
        timestamp = _timestamp_seconds(raw, self._timezone)
        content_raw = raw.get("content")
        content = content_raw if isinstance(content_raw, dict) else {}
        text = _string(content.get("text"))
        elements = content.get("elements")
        elements = elements if isinstance(elements, list) else []
        message_id = _string(raw.get("id")) or _fallback_message_id(
            group_id,
            sender_id,
            timestamp,
            text,
            fallback_index,
        )
        return MessageEnvelope(
            platform_id="qce-export",
            bot_self_id="qce-history",
            external_message_id=f"{group_id}:{message_id}",
            session_id=f"qce:group:{group_id}",
            group_id=group_id,
            sender_id=sender_id,
            sender_name=sender_name,
            sent_at_utc=timestamp,
            text=text,
            outline=_element_outline(elements),
            reply_to_message_id=_prefixed_reply_id(elements, group_id),
            is_group_message=True,
            is_self_message=sender_id in self._bot_ids,
            is_system_message=bool(raw.get("system")),
        )


@contextmanager
def _open_qce_source(
    path: Path,
) -> Iterator[tuple[dict[str, Any], Iterable[dict[str, Any]]]]:
    path = path.expanduser()
    if not path.is_file():
        raise QceImportError(f"历史记录文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            chat_info, messages = _source_from_zip(archive)
            yield chat_info, messages
        return
    if suffix == ".jsonl":
        manifest = path.parent.parent / "manifest.json"
        if not manifest.is_file():
            manifest = path.parent / "manifest.json"
        if not manifest.is_file():
            raise QceImportError("单独的 JSONL 缺少 manifest.json，请上传完整分块 ZIP")
        path = manifest
    if suffix not in {".json", ".jsonl"}:
        raise QceImportError("只支持 QCE 导出的 .json、.jsonl 或 .zip")
    data = _load_json_path(path)
    chat_info = _required_object(data, "chatInfo")
    if isinstance(data.get("messages"), list):
        yield chat_info, _object_messages(data["messages"])
        return
    chunked = _required_object(data, "chunked")
    yield chat_info, _manifest_path_messages(path.parent, chunked)


def _source_from_zip(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, Any], Iterable[dict[str, Any]]]:
    names = [name for name in archive.namelist() if not name.endswith("/")]
    manifest_names = [name for name in names if PurePosixPath(name).name == "manifest.json"]
    for name in manifest_names:
        data = _load_zip_json(archive, name)
        if isinstance(data.get("chatInfo"), dict) and isinstance(data.get("chunked"), dict):
            root = str(PurePosixPath(name).parent)
            root = "" if root == "." else root
            return data["chatInfo"], _manifest_zip_messages(archive, root, data["chunked"])
    for name in names:
        if not name.lower().endswith(".json"):
            continue
        data = _load_zip_json(archive, name)
        if isinstance(data.get("chatInfo"), dict) and isinstance(data.get("messages"), list):
            return data["chatInfo"], _object_messages(data["messages"])
    raise QceImportError("ZIP 中没有找到 QCE JSON 或分块 manifest.json")


def _manifest_path_messages(base_dir: Path, chunked: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for relative in _chunk_paths(chunked):
        candidate = (base_dir / Path(*PurePosixPath(relative).parts)).resolve()
        if base_dir.resolve() not in candidate.parents:
            raise QceImportError("manifest 包含越出导出目录的 chunk 路径")
        if not candidate.is_file():
            raise QceImportError(f"缺少分块文件：{relative}")
        with candidate.open("r", encoding="utf-8-sig") as stream:
            yield from _jsonl_messages(stream, relative)


def _manifest_zip_messages(
    archive: zipfile.ZipFile,
    root: str,
    chunked: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    names = set(archive.namelist())
    for relative in _chunk_paths(chunked):
        member = str(PurePosixPath(root) / PurePosixPath(relative)) if root else relative
        if member not in names:
            raise QceImportError(f"ZIP 缺少分块文件：{relative}")
        with (
            archive.open(member) as raw_stream,
            io.TextIOWrapper(raw_stream, encoding="utf-8-sig") as stream,
        ):
            yield from _jsonl_messages(stream, relative)


def _chunk_paths(chunked: dict[str, Any]) -> list[str]:
    chunks = chunked.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise QceImportError("分块 manifest 没有 chunks 列表")
    result: list[str] = []
    for item in chunks:
        if not isinstance(item, dict):
            raise QceImportError("分块 manifest 的 chunk 条目无效")
        relative = _string(item.get("relativePath")) or _string(item.get("fileName"))
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            raise QceImportError("分块 manifest 包含不安全路径")
        result.append(relative)
    return result


def _jsonl_messages(stream: TextIO, source: str) -> Iterator[dict[str, Any]]:
    for line_no, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QceImportError(f"{source} 第 {line_no} 行不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise QceImportError(f"{source} 第 {line_no} 行不是消息对象")
        yield value


def _object_messages(values: list[Any]) -> Iterator[dict[str, Any]]:
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise QceImportError(f"messages 第 {index} 项不是对象")
        yield value


def _load_json_path(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_SINGLE_JSON_BYTES:
        raise QceImportError("单文件 JSON 超过 256MB，请用 QCE 分块 JSONL 并上传 ZIP")
    with path.open("r", encoding="utf-8-sig") as stream:
        return _load_json_object(stream, str(path))


def _load_zip_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    info = archive.getinfo(name)
    if info.file_size > _MAX_SINGLE_JSON_BYTES:
        raise QceImportError("ZIP 内单个 JSON 超过 256MB，请使用 QCE 分块 JSONL")
    with (
        archive.open(info) as raw_stream,
        io.TextIOWrapper(raw_stream, encoding="utf-8-sig") as stream,
    ):
        return _load_json_object(stream, name)


def _load_json_object(stream: TextIO, source: str) -> dict[str, Any]:
    try:
        value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise QceImportError(f"不是有效 JSON：{source}") from exc
    if not isinstance(value, dict):
        raise QceImportError(f"QCE 导出根节点必须是对象：{source}")
    return value


def _required_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise QceImportError(f"QCE 导出缺少对象字段 {key}")
    return value


def _timestamp_seconds(message: dict[str, Any], timezone: ZoneInfo) -> int:
    raw = message.get("timestamp")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        numeric = float(raw)
        return int(numeric / 1000) if numeric >= 100_000_000_000 else int(numeric)
    time_text = _string(message.get("time"))
    if time_text:
        try:
            parsed = datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone)
            return int(parsed.timestamp())
        except ValueError:
            pass
    raise QceImportError("消息缺少有效 timestamp/time")


def _element_outline(elements: list[Any]) -> str:
    labels = {
        "image": "[图片]",
        "video": "[视频]",
        "audio": "[语音]",
        "file": "[文件]",
        "face": "[表情]",
        "market_face": "[表情]",
        "reply": "[回复消息]",
        "at": "[提及用户]",
        "forward": "[转发消息]",
        "json": "[卡片消息]",
        "location": "[位置]",
    }
    result: list[str] = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        element_type = _string(item.get("type")).lower()
        if element_type == "text":
            continue
        label = labels.get(element_type, "[其他消息]")
        if label not in result:
            result.append(label)
    return " ".join(result)


def _prefixed_reply_id(elements: list[Any], group_id: str) -> str:
    for item in elements:
        if not isinstance(item, dict) or _string(item.get("type")).lower() != "reply":
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        for key in ("referencedMessageId", "replyMsgId", "sourceMsgId", "msgId"):
            reply_id = _string(data.get(key))
            if reply_id:
                return f"{group_id}:{reply_id}"
    return ""


def _fallback_message_id(
    group_id: str,
    sender_id: str,
    timestamp: int,
    text: str,
    index: int,
) -> str:
    payload = f"{group_id}\x1f{sender_id}\x1f{timestamp}\x1f{text}\x1f{index}"
    return "generated:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_ids_from_qce_filename(filename: str) -> set[str]:
    """Extract group IDs from QCE's canonical ``group_<id>_...`` filename."""

    return set(re.findall(r"(?:^|[_-])group[_-](\d+)(?=[_.-]|$)", filename, flags=re.IGNORECASE))


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value).strip()
    return ""
