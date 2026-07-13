"""Configurable report-viewer and operator authorization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import PluginConfig


class PermissionAction(str, Enum):
    VIEW_REPORT = "VIEW_REPORT"
    OPERATE = "OPERATE"


class AuthorizationStatus(str, Enum):
    ALLOWED = "ALLOWED"
    PRIVATE_REQUIRED = "PRIVATE_REQUIRED"
    DENIED = "DENIED"


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    status: AuthorizationStatus

    @property
    def allowed(self) -> bool:
        return self.status is AuthorizationStatus.ALLOWED

    @property
    def user_message(self) -> str:
        if self.status is AuthorizationStatus.PRIVATE_REQUIRED:
            return "请私聊机器人执行该指令。"
        if self.status is AuthorizationStatus.DENIED:
            return "你没有执行该指令的权限。"
        return ""


class PermissionService:
    """Apply private-chat and role checks before any data lookup occurs."""

    def __init__(self, config: PluginConfig) -> None:
        self._config = config
        self._viewers = frozenset(config.report_viewer_qq_ids)
        self._operators = frozenset(config.operator_qq_ids)

    def authorize(
        self,
        *,
        sender_id: str,
        action: PermissionAction,
        is_private: bool,
        is_astrbot_admin: bool,
    ) -> AuthorizationResult:
        if self._config.sensitive_commands_private_only and not is_private:
            return AuthorizationResult(AuthorizationStatus.PRIVATE_REQUIRED)

        normalized_sender = str(sender_id).strip()
        if action is PermissionAction.VIEW_REPORT:
            allowed = (
                normalized_sender in self._viewers
                or normalized_sender in self._operators
                or (
                    is_astrbot_admin
                    and (
                        self._config.inherit_astrbot_admins_as_viewers
                        or self._config.inherit_astrbot_admins_as_operators
                    )
                )
            )
        elif action is PermissionAction.OPERATE:
            allowed = normalized_sender in self._operators or (
                is_astrbot_admin and self._config.inherit_astrbot_admins_as_operators
            )
        else:  # pragma: no cover - protected by the enum type
            allowed = False
        return AuthorizationResult(
            AuthorizationStatus.ALLOWED if allowed else AuthorizationStatus.DENIED
        )
