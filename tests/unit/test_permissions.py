from __future__ import annotations

from nju_report.config import PluginConfig
from nju_report.permissions import (
    AuthorizationStatus,
    PermissionAction,
    PermissionService,
)


def _service(**overrides: object) -> PermissionService:
    raw = {
        "report_viewer_qq_ids": ["viewer"],
        "operator_qq_ids": ["operator"],
        **overrides,
    }
    return PermissionService(PluginConfig.from_mapping(raw))


def test_viewer_and_operator_permissions_are_independent() -> None:
    service = _service()

    assert service.authorize(
        sender_id="viewer",
        action=PermissionAction.VIEW_REPORT,
        is_private=True,
        is_astrbot_admin=False,
    ).allowed
    assert not service.authorize(
        sender_id="viewer",
        action=PermissionAction.OPERATE,
        is_private=True,
        is_astrbot_admin=False,
    ).allowed
    assert service.authorize(
        sender_id="operator",
        action=PermissionAction.OPERATE,
        is_private=True,
        is_astrbot_admin=False,
    ).allowed
    assert not service.authorize(
        sender_id="operator",
        action=PermissionAction.VIEW_REPORT,
        is_private=True,
        is_astrbot_admin=False,
    ).allowed


def test_private_requirement_is_checked_before_identity() -> None:
    result = _service().authorize(
        sender_id="unknown",
        action=PermissionAction.VIEW_REPORT,
        is_private=False,
        is_astrbot_admin=False,
    )
    assert result.status is AuthorizationStatus.PRIVATE_REQUIRED
    assert result.user_message == "请私聊机器人执行该指令。"


def test_astrbot_admin_inheritance_is_configurable_per_action() -> None:
    service = _service(
        inherit_astrbot_admins_as_viewers=False,
        inherit_astrbot_admins_as_operators=True,
    )
    assert not service.authorize(
        sender_id="admin",
        action=PermissionAction.VIEW_REPORT,
        is_private=True,
        is_astrbot_admin=True,
    ).allowed
    assert service.authorize(
        sender_id="admin",
        action=PermissionAction.OPERATE,
        is_private=True,
        is_astrbot_admin=True,
    ).allowed


def test_group_commands_can_be_enabled_but_still_require_role() -> None:
    service = _service(sensitive_commands_private_only=False)
    assert service.authorize(
        sender_id="viewer",
        action=PermissionAction.VIEW_REPORT,
        is_private=False,
        is_astrbot_admin=False,
    ).allowed
    assert not service.authorize(
        sender_id="unknown",
        action=PermissionAction.VIEW_REPORT,
        is_private=False,
        is_astrbot_admin=False,
    ).allowed
