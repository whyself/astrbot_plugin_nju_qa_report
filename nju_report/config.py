"""Validated runtime configuration for the plugin."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import time
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_NAMESPACE_RE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*")
_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_DEFAULT_TARGET_GROUPS = ("826811581",)
_DEFAULT_GROUP_ALIASES = {"826811581": "南京大学迎新群"}


class ConfigError(ValueError):
    """Raised when one or more plugin settings are invalid."""


@dataclass(frozen=True, slots=True)
class RepositoryExclusion:
    """A configurable repository exclusion rule."""

    namespace: str
    reason: str = ""


_DEFAULT_EXCLUSIONS = (
    RepositoryExclusion(
        "qc19gt/ogaye8",
        "QA 成品仓库，默认不作为缺口调查源",
    ),
)


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """Strongly typed plugin configuration."""

    capture_enabled: bool = False
    capture_mode: str = "selected_groups"
    target_group_ids: tuple[str, ...] = _DEFAULT_TARGET_GROUPS
    group_aliases: Mapping[str, str] | None = None
    command_prefixes: tuple[str, ...] = ("/",)
    capture_queue_size: int = 5000
    timezone: str = "Asia/Shanghai"
    raw_message_retention_days: int = 90

    report_viewer_qq_ids: tuple[str, ...] = ()
    operator_qq_ids: tuple[str, ...] = ()
    inherit_astrbot_admins_as_viewers: bool = True
    inherit_astrbot_admins_as_operators: bool = True
    sensitive_commands_private_only: bool = True

    llm_provider_id: str = ""
    embedding_api_key: str = field(default="", repr=False)
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    enable_vector_search: bool = True
    batch_concurrency: int = 2
    request_timeout_seconds: int = 120
    max_retries: int = 3
    scope_auto_review_enabled: bool = True
    scope_auto_review_max_rounds: int = 2

    yuque_token: str = field(default="", repr=False)
    yuque_api_base: str = "https://nova.yuque.com/api/v2"
    yuque_space_login: str = "qc19gt"
    approved_repositories: tuple[str, ...] = ()
    excluded_repositories: tuple[RepositoryExclusion, ...] = _DEFAULT_EXCLUSIONS
    purge_excluded_repository_data: bool = True

    daily_report_enabled: bool = False
    daily_report_time: str = "00:00"
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_username: str = ""
    smtp_password: str = field(default="", repr=False)
    smtp_use_ssl: bool = True
    mail_from: str = ""
    mail_recipients: tuple[str, ...] = ()

    def group_alias(self, group_id: str) -> str:
        """Return a configured group alias, falling back to a masked identifier."""

        aliases = self.group_aliases or {}
        configured = str(aliases.get(group_id, "")).strip()
        if configured:
            return configured
        if len(group_id) <= 4:
            return "群聊-****"
        return f"群聊-****{group_id[-4:]}"

    @classmethod
    def from_mapping(cls, raw: Any) -> PluginConfig:
        """Parse AstrBot's mapping-like config without exposing secret values."""

        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ConfigError("插件配置必须是对象")

        capture_mode = _string(raw, "capture_mode", "selected_groups")
        if capture_mode not in {"selected_groups", "all_group_messages"}:
            raise ConfigError("capture_mode 必须是 selected_groups 或 all_group_messages")

        target_group_ids = _id_tuple(
            raw.get("target_group_ids", _DEFAULT_TARGET_GROUPS),
            "target_group_ids",
        )
        capture_enabled = _boolean(raw, "capture_enabled", False)
        if capture_enabled and capture_mode == "selected_groups" and not target_group_ids:
            raise ConfigError("启用 selected_groups 采集前必须填写 target_group_ids")
        aliases = _aliases(raw.get("group_aliases", _DEFAULT_GROUP_ALIASES))
        timezone = _string(raw, "timezone", "Asia/Shanghai")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ConfigError("timezone 不是有效的 IANA 时区") from exc

        report_time = _string(raw, "daily_report_time", "00:00")
        _parse_time(report_time)

        retention = _integer(
            raw,
            "raw_message_retention_days",
            90,
            minimum=1,
            maximum=3650,
        )
        capture_queue_size = _integer(raw, "capture_queue_size", 5000, minimum=100, maximum=100000)
        review_rounds = _integer(
            raw,
            "scope_auto_review_max_rounds",
            2,
            minimum=1,
            maximum=5,
        )
        batch_concurrency = _integer(raw, "batch_concurrency", 2, minimum=1, maximum=16)
        request_timeout = _integer(raw, "request_timeout_seconds", 120, minimum=10, maximum=1800)
        max_retries = _integer(raw, "max_retries", 3, minimum=0, maximum=10)
        smtp_port = _integer(raw, "smtp_port", 465, minimum=1, maximum=65535)

        api_base = _string(raw, "yuque_api_base", "https://nova.yuque.com/api/v2").rstrip("/")
        parsed_api_base = urlparse(api_base)
        if parsed_api_base.scheme != "https" or not parsed_api_base.netloc:
            raise ConfigError("yuque_api_base 必须是有效的 HTTPS URL")

        approved = _repository_tuple(raw.get("approved_repositories", ()), "approved_repositories")
        excluded = _repository_exclusions(raw.get("excluded_repositories"))
        excluded_namespaces = {item.namespace for item in excluded}
        overlap = sorted(set(approved) & excluded_namespaces)
        if overlap:
            raise ConfigError(
                "仓库不能同时出现在 approved_repositories 和 "
                f"excluded_repositories：{', '.join(overlap)}"
            )

        command_prefixes = _string_tuple(
            raw.get("command_prefixes", ("/",)),
            "command_prefixes",
            allow_empty_collection=False,
        )

        space_login = _string(raw, "yuque_space_login", "qc19gt")
        if not space_login:
            raise ConfigError("yuque_space_login 不能为空")

        embedding_base_url = _string(raw, "embedding_base_url", "").rstrip("/")
        if embedding_base_url:
            parsed_embedding_url = urlparse(embedding_base_url)
            if (
                parsed_embedding_url.scheme not in {"http", "https"}
                or not parsed_embedding_url.netloc
            ):
                raise ConfigError("embedding_base_url 必须是有效的 HTTP(S) URL")
        embedding_model = _string(raw, "embedding_model", "text-embedding-3-small")
        if not embedding_model:
            raise ConfigError("embedding_model 不能为空")

        return cls(
            capture_enabled=capture_enabled,
            capture_mode=capture_mode,
            target_group_ids=target_group_ids,
            group_aliases=aliases,
            command_prefixes=command_prefixes,
            capture_queue_size=capture_queue_size,
            timezone=timezone,
            raw_message_retention_days=retention,
            report_viewer_qq_ids=_id_tuple(
                raw.get("report_viewer_qq_ids", ()), "report_viewer_qq_ids"
            ),
            operator_qq_ids=_id_tuple(raw.get("operator_qq_ids", ()), "operator_qq_ids"),
            inherit_astrbot_admins_as_viewers=_boolean(
                raw, "inherit_astrbot_admins_as_viewers", True
            ),
            inherit_astrbot_admins_as_operators=_boolean(
                raw, "inherit_astrbot_admins_as_operators", True
            ),
            sensitive_commands_private_only=_boolean(
                raw,
                "sensitive_commands_private_only",
                _boolean(raw, "report_commands_private_only", True),
            ),
            llm_provider_id=_string(raw, "llm_provider_id", ""),
            embedding_api_key=_string(raw, "embedding_api_key", ""),
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            enable_vector_search=_boolean(raw, "enable_vector_search", True),
            batch_concurrency=batch_concurrency,
            request_timeout_seconds=request_timeout,
            max_retries=max_retries,
            scope_auto_review_enabled=_boolean(raw, "scope_auto_review_enabled", True),
            scope_auto_review_max_rounds=review_rounds,
            yuque_token=_string(raw, "yuque_token", ""),
            yuque_api_base=api_base,
            yuque_space_login=space_login,
            approved_repositories=approved,
            excluded_repositories=excluded,
            purge_excluded_repository_data=_boolean(raw, "purge_excluded_repository_data", True),
            daily_report_enabled=_boolean(raw, "daily_report_enabled", False),
            daily_report_time=report_time,
            smtp_host=_string(raw, "smtp_host", ""),
            smtp_port=smtp_port,
            smtp_username=_string(raw, "smtp_username", ""),
            smtp_password=_string(raw, "smtp_password", ""),
            smtp_use_ssl=_boolean(raw, "smtp_use_ssl", True),
            mail_from=_optional_email(_string(raw, "mail_from", ""), "mail_from"),
            mail_recipients=_email_tuple(
                raw.get("mail_recipients", ()),
                "mail_recipients",
            ),
        )


def _boolean(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} 必须是布尔值")
    return value


def _integer(
    raw: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} 必须是整数")
    if not minimum <= value <= maximum:
        raise ConfigError(f"{key} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _string(raw: Mapping[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} 必须是字符串")
    return value.strip()


def _string_tuple(
    value: Any,
    field: str,
    *,
    allow_empty_collection: bool,
) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ConfigError(f"{field} 必须是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise ConfigError(f"{field} 中的每一项都必须是字符串")
        normalized = item.strip()
        if not normalized:
            raise ConfigError(f"{field} 不能包含空字符串")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    if not allow_empty_collection and not result:
        raise ConfigError(f"{field} 不能为空")
    return tuple(result)


def _id_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{field} 必须是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{field} 中的 QQ 号或群号必须以字符串填写")
        normalized = item.strip()
        if not normalized:
            raise ConfigError(f"{field} 不能包含空字符串")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _aliases(value: Any) -> dict[str, str]:
    if value in (None, ""):
        return {}
    aliases: dict[str, str] = {}
    if isinstance(value, Mapping):
        entries = value.items()
    elif isinstance(value, (list, tuple)):
        parsed_entries: list[tuple[Any, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ConfigError("group_aliases 的模板条目必须是对象")
            parsed_entries.append((item.get("group_id", ""), item.get("alias", "")))
        entries = parsed_entries
    else:
        raise ConfigError("group_aliases 必须是模板列表或兼容的对象")
    for group_id, alias in entries:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ConfigError("group_aliases 的 group_id 必须是非空字符串")
        if not isinstance(alias, str) or not alias.strip():
            raise ConfigError("group_aliases 的 alias 必须是非空字符串")
        aliases[group_id.strip()] = alias.strip()
    return aliases


def _repository_tuple(value: Any, field: str) -> tuple[str, ...]:
    values = _string_tuple(value, field, allow_empty_collection=True)
    for namespace in values:
        _validate_namespace(namespace, field)
    return values


def _repository_exclusions(value: Any) -> tuple[RepositoryExclusion, ...]:
    if value is None:
        return _DEFAULT_EXCLUSIONS
    if not isinstance(value, (list, tuple)):
        raise ConfigError("excluded_repositories 必须是数组")
    result: list[RepositoryExclusion] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            namespace = item.strip()
            reason = ""
        elif isinstance(item, Mapping):
            namespace_raw = item.get("namespace", "")
            reason_raw = item.get("reason", "")
            if not isinstance(namespace_raw, str) or not isinstance(reason_raw, str):
                raise ConfigError("excluded_repositories 的 namespace/reason 必须是字符串")
            namespace = namespace_raw.strip()
            reason = reason_raw.strip()
        else:
            raise ConfigError("excluded_repositories 中的项目必须是字符串或对象")
        _validate_namespace(namespace, "excluded_repositories")
        if namespace not in seen:
            seen.add(namespace)
            result.append(RepositoryExclusion(namespace, reason))
    return tuple(result)


def _validate_namespace(namespace: str, field: str) -> None:
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ConfigError(f"{field} 包含无效的语雀 namespace")


def _parse_time(value: str) -> time:
    if not re.fullmatch(r"\d{2}:\d{2}", value):
        raise ConfigError("daily_report_time 必须使用 HH:MM 格式")
    try:
        hour_text, minute_text = value.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
        return time(hour=hour, minute=minute)
    except (ValueError, TypeError) as exc:
        raise ConfigError("daily_report_time 必须使用 HH:MM 格式") from exc


def _optional_email(value: str, field: str) -> str:
    normalized = value.strip()
    if normalized and not _EMAIL_RE.fullmatch(normalized):
        raise ConfigError(f"{field} 必须是有效邮箱地址")
    return normalized


def _email_tuple(value: Any, field: str) -> tuple[str, ...]:
    addresses = _string_tuple(value, field, allow_empty_collection=True)
    for address in addresses:
        if not _EMAIL_RE.fullmatch(address):
            raise ConfigError(f"{field} 包含无效邮箱地址")
    return addresses
