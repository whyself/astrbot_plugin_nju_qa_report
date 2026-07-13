from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from nju_report.config import PluginConfig
from nju_report.qce_import import QceHistoryImporter, QceImportError
from nju_report.storage import ReportStorage
from nju_report.time_windows import natural_day_window


def _message(
    message_id: str,
    text: str,
    timestamp_ms: int,
    *,
    sender: str = "10001",
    elements: list[dict[str, object]] | None = None,
    system: bool = False,
) -> dict[str, object]:
    return {
        "id": message_id,
        "timestamp": timestamp_ms,
        "time": "2026-07-01 08:00:00",
        "sender": {"uid": f"u_{sender}", "uin": sender, "name": f"用户{sender}"},
        "type": "normal",
        "content": {
            "text": text,
            "elements": elements or ([{"type": "text", "data": {"text": text}}] if text else []),
            "resources": [],
            "mentions": [],
        },
        "recalled": False,
        "system": system,
    }


def _single_export(messages: list[dict[str, object]], *, group_id: str = "826811581"):
    return {
        "metadata": {"name": "QQChatExporter", "version": "5.5.79"},
        "chatInfo": {
            "name": "南京大学迎新群",
            "type": "group",
            "peerUid": group_id,
            "selfUin": "55555",
        },
        "messages": messages,
    }


def _config(path: Path) -> PluginConfig:
    return PluginConfig.from_mapping(
        {
            "capture_enabled": False,
            "target_group_ids": ["826811581"],
            "history_import_files": [str(path)],
            "history_import_bot_qq_ids": ["99999"],
        }
    )


def test_single_json_import_is_filtered_and_idempotent(tmp_path: Path) -> None:
    base = 1_751_324_400_000
    messages = [
        _message("m1", "校园卡丢了怎么补办？", base),
        _message("m2", "/nju help", base + 1000),
        _message("m3", "系统通知", base + 2000, system=True),
        _message("m4", "机器人回答", base + 3000, sender="99999"),
        _message(
            "m5",
            "",
            base + 4000,
            elements=[{"type": "image", "data": {"filename": "photo.jpg"}}],
        ),
        _message(
            "m6",
            "去信息门户挂失后补办",
            base + 5000,
            elements=[
                {"type": "reply", "data": {"replyMsgId": "m1"}},
                {"type": "text", "data": {"text": "去信息门户挂失后补办"}},
            ],
        ),
    ]
    export_path = tmp_path / "history.json"
    export_path.write_text(
        json.dumps(_single_export(messages), ensure_ascii=False),
        encoding="utf-8",
    )
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    importer = QceHistoryImporter(_config(export_path), storage)

    inspected = importer.inspect_all()
    assert inspected[0].message_count == 6
    assert inspected[0].group_id == "826811581"

    first = importer.import_all()[0]
    assert first.scanned == 6
    assert first.imported == 3
    assert first.duplicates == 0
    assert first.skipped["COMMAND_MESSAGE"] == 1
    assert first.skipped["SYSTEM_MESSAGE"] == 1
    assert first.skipped["BOT_MESSAGE"] == 1

    second = importer.import_all()[0]
    assert second.imported == 0
    assert second.duplicates == 3
    window = natural_day_window(date(2025, 7, 1), "Asia/Shanghai")
    stored = storage.messages_in_window(window)
    assert len(stored) == 3
    assert stored[1].outline == "[图片]"
    assert stored[1].analyzable is False
    assert stored[2].reply_to_message_id == "826811581:m1"
    storage.close()


def test_group_mismatch_is_rejected_before_import(tmp_path: Path) -> None:
    export_path = tmp_path / "wrong-group.json"
    export_path.write_text(
        json.dumps(_single_export([], group_id="123456789"), ensure_ascii=False),
        encoding="utf-8",
    )
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    importer = QceHistoryImporter(_config(export_path), storage)

    with pytest.raises(QceImportError, match="target_group_ids"):
        importer.import_all()
    assert storage.message_count() == 0
    storage.close()


def test_chunked_jsonl_zip_is_supported(tmp_path: Path) -> None:
    manifest = {
        "chatInfo": {
            "name": "南京大学迎新群",
            "type": "group",
            "peerUid": "826811581",
        },
        "chunked": {
            "format": "jsonl",
            "chunks": [{"index": 1, "relativePath": "chunks/c000001.jsonl", "count": 2}],
        },
    }
    messages = [
        _message("z1", "仙林校区快递点在哪里？", 1_751_324_400_000),
        _message("z2", "在校门附近", 1_751_324_401_000),
    ]
    zip_path = tmp_path / "history.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("export/manifest.json", json.dumps(manifest, ensure_ascii=False))
        archive.writestr(
            "export/chunks/c000001.jsonl",
            "\n".join(json.dumps(item, ensure_ascii=False) for item in messages),
        )
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()
    result = QceHistoryImporter(_config(zip_path), storage).import_all()[0]

    assert result.scanned == 2
    assert result.imported == 2
    assert storage.message_count() == 2
    storage.close()


def test_qce_5579_filename_group_id_and_reply_field_are_supported(tmp_path: Path) -> None:
    export_path = tmp_path / "group_826811581_20260713_164457.json"
    payload = _single_export(
        [
            _message("original", "校园卡在哪里补办？", 1_782_386_895_000),
            _message(
                "reply",
                "在信息门户挂失后补办",
                1_782_386_896_000,
                elements=[
                    {
                        "type": "reply",
                        "data": {"messageId": "0", "referencedMessageId": "original"},
                    },
                    {"type": "text", "data": {"text": "在信息门户挂失后补办"}},
                ],
            ),
        ]
    )
    payload["chatInfo"].pop("peerUid")
    export_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()

    result = QceHistoryImporter(_config(export_path), storage).import_all()[0]

    assert result.group_id == "826811581"
    assert result.imported == 2
    window = natural_day_window(date(2026, 6, 25), "Asia/Shanghai")
    stored = storage.messages_in_window(window)
    assert stored[1].reply_to_message_id == "826811581:original"
    storage.close()


def test_astrbot_relative_file_path_resolves_from_plugin_data_dir(tmp_path: Path) -> None:
    relative = Path("files/history_import_files/group_826811581_20260713.json")
    export_path = tmp_path / relative
    export_path.parent.mkdir(parents=True)
    export_path.write_text(
        json.dumps(
            _single_export([_message("relative", "宿舍床尺寸是多少？", 1_783_958_400_000)]),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = PluginConfig.from_mapping(
        {
            "target_group_ids": ["826811581"],
            "history_import_files": [relative.as_posix()],
        }
    )
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()

    result = QceHistoryImporter(config, storage, base_dir=tmp_path).import_all()[0]

    assert result.imported == 1
    assert storage.message_count() == 1
    storage.close()


def test_relative_file_path_cannot_escape_plugin_data_dir(tmp_path: Path) -> None:
    config = PluginConfig.from_mapping(
        {
            "target_group_ids": ["826811581"],
            "history_import_files": ["../outside.json"],
        }
    )
    storage = ReportStorage(tmp_path / "report.sqlite3")
    storage.initialize()

    with pytest.raises(QceImportError, match="越出插件数据目录"):
        QceHistoryImporter(config, storage, base_dir=tmp_path).inspect_all()
    storage.close()
