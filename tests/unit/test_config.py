from __future__ import annotations

import json
from pathlib import Path

import pytest

from nju_report.config import ConfigError, PluginConfig


def test_webui_schema_defaults_are_accepted_by_runtime_parser() -> None:
    schema_path = Path(__file__).parents[2] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    defaults = {
        name: definition["default"]
        for name, definition in schema.items()
        if "default" in definition
    }
    config = PluginConfig.from_mapping(defaults)
    assert config.capture_enabled is False


def test_defaults_are_safe_and_exclude_qa_repository() -> None:
    config = PluginConfig.from_mapping({})

    assert config.capture_enabled is False
    assert config.daily_report_enabled is False
    assert config.timezone == "Asia/Shanghai"
    assert config.target_group_ids == ("826811581",)
    assert config.group_alias("826811581") == "南京大学迎新群"
    assert config.daily_report_time == "00:00"
    assert config.scope_auto_review_enabled is True
    assert config.scope_auto_review_max_rounds == 2
    assert config.embedding_model == "text-embedding-3-small"
    assert config.enable_vector_search is True
    assert [item.namespace for item in config.excluded_repositories] == ["qc19gt/ogaye8"]


def test_enabled_selected_capture_requires_target_groups() -> None:
    with pytest.raises(ConfigError, match="target_group_ids"):
        PluginConfig.from_mapping({"capture_enabled": True, "target_group_ids": []})


def test_all_group_mode_can_be_enabled_without_target_list() -> None:
    config = PluginConfig.from_mapping(
        {
            "capture_enabled": True,
            "capture_mode": "all_group_messages",
            "target_group_ids": [],
        }
    )
    assert config.capture_enabled is True
    assert config.target_group_ids == ()


def test_ids_must_be_strings_and_are_deduplicated() -> None:
    config = PluginConfig.from_mapping({"report_viewer_qq_ids": [" 123 ", "123", "456"]})
    assert config.report_viewer_qq_ids == ("123", "456")

    with pytest.raises(ConfigError, match="字符串"):
        PluginConfig.from_mapping({"report_viewer_qq_ids": [123]})


def test_string_boolean_is_rejected() -> None:
    with pytest.raises(ConfigError, match="布尔值"):
        PluginConfig.from_mapping({"capture_enabled": "false"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_mode", "unknown"),
        ("timezone", "Not/A-Timezone"),
        ("daily_report_time", "3:5"),
        ("scope_auto_review_max_rounds", True),
    ],
)
def test_invalid_config_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ConfigError):
        PluginConfig.from_mapping({field: value})


def test_repository_allow_and_exclude_sets_cannot_overlap() -> None:
    with pytest.raises(ConfigError, match="同时"):
        PluginConfig.from_mapping(
            {
                "approved_repositories": ["qc19gt/ogaye8"],
                "excluded_repositories": ["qc19gt/ogaye8"],
            }
        )


def test_yuque_api_rejects_plain_http() -> None:
    with pytest.raises(ConfigError, match="HTTPS"):
        PluginConfig.from_mapping({"yuque_api_base": "http://nova.yuque.com/api/v2"})


def test_exclusion_is_configurable_not_hard_coded() -> None:
    config = PluginConfig.from_mapping({"excluded_repositories": []})
    assert config.excluded_repositories == ()


def test_secrets_are_not_in_dataclass_repr() -> None:
    config = PluginConfig.from_mapping(
        {
            "yuque_token": "yuque-secret",
            "smtp_password": "smtp-secret",
            "embedding_api_key": "embedding-secret",
        }
    )
    rendered = repr(config)
    assert "yuque-secret" not in rendered
    assert "smtp-secret" not in rendered
    assert "embedding-secret" not in rendered


def test_reference_plugin_embedding_configuration_is_accepted() -> None:
    config = PluginConfig.from_mapping(
        {
            "embedding_api_key": "secret",
            "embedding_base_url": "https://api.example.com/v1/",
            "embedding_model": "text-embedding-3-large",
            "enable_vector_search": True,
        }
    )
    assert config.embedding_base_url == "https://api.example.com/v1"
    assert config.embedding_model == "text-embedding-3-large"

    with pytest.raises(ConfigError, match="embedding_base_url"):
        PluginConfig.from_mapping({"embedding_base_url": "not-a-url"})


def test_group_alias_falls_back_to_masked_group_id() -> None:
    config = PluginConfig.from_mapping({"group_aliases": {"12345678": "迎新一群"}})
    assert config.group_alias("12345678") == "迎新一群"
    assert config.group_alias("87654321") == "群聊-****4321"


def test_email_recipients_are_validated_and_independent_from_qq_roles() -> None:
    config = PluginConfig.from_mapping(
        {
            "mail_recipients": ["maintainer@example.edu.cn"],
            "report_viewer_qq_ids": [],
            "operator_qq_ids": [],
        }
    )
    assert config.mail_recipients == ("maintainer@example.edu.cn",)

    with pytest.raises(ConfigError, match="邮箱"):
        PluginConfig.from_mapping({"mail_recipients": ["not-an-email"]})
